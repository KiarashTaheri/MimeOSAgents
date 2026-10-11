"""Run the eval test set against the live agent on mimOE and report a pass rate.

    uv run python evals/run_evals.py            # all cases
    uv run python evals/run_evals.py -k lows    # cases whose id contains "lows"

Each case is scored on:
- tools:   the expected tools were called (and none were called when none are expected)
- days:    every call used the expected `days` window
- numbers: each expected value (computed live from cgm.py, the ground truth) appears in the answer
- text:    expected words appear in the answer (e.g. the worst time of day, "healthcare provider")

Every case is a Logfire span `eval.case` with `passed` and the sub-scores, so pass rate over time can be
charted in Logfire. Results are also written to evals/results.json.
"""

import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Any

import logfire
from agents import SQLiteSession
from pydantic import BaseModel

from mimoe_agent import cgm
from mimoe_agent.agent import AgentError, CGMAssistant
from mimoe_agent.observability import setup_observability

HERE = Path(__file__).resolve().parent


class FieldRef(BaseModel):
    metric: str  # average | time_in_range | low_events | high_events
    days: int
    field: str  # dotted path into the metric's pydantic model


class Case(BaseModel):
    id: str
    question: str
    expected_tools: list[str]
    expected_days: int | None = None
    allow_tools: bool = False  # for expected_tools == []: tolerate (harmless) data lookups
    numbers: list[FieldRef] = []
    text: list[FieldRef] = []
    must_contain_any: list[str] = []


class CaseResult(BaseModel):
    id: str
    passed: bool
    tools_ok: bool
    days_ok: bool
    numbers_ok: bool
    text_ok: bool
    tools_called: list[str]
    expected_values: dict[str, Any]
    latency_ms: int | None
    answer: str
    error: str | None = None


def ground_truth(readings: list[cgm.GlucoseReading], ref: FieldRef) -> Any:
    metrics = {
        "average": lambda: cgm.average(readings, ref.days),
        "time_in_range": lambda: cgm.time_in_range(readings, ref.days),
        "low_events": lambda: cgm.find_events(readings, ref.days, "low"),
        "high_events": lambda: cgm.find_events(readings, ref.days, "high"),
    }
    value: Any = metrics[ref.metric]()
    for part in ref.field.split("."):
        value = getattr(value, part)
    return value


def numbers_in(text: str) -> list[float]:
    return [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))]


def number_present(expected: float, answer: str) -> bool:
    # Strict: the prompt tells the model to quote tool values (1 decimal). A loose tolerance would let
    # unrelated numbers ("7 days") satisfy an expected 6.9.
    return any(abs(n - expected) < 0.051 for n in numbers_in(answer))


async def run_case(assistant: CGMAssistant, case: Case) -> CaseResult:
    readings = assistant.context.readings
    expected_values = {f"{r.metric}.{r.field}@{r.days}d": ground_truth(readings, r) for r in case.numbers + case.text}

    with logfire.span("eval.case {case_id}", case_id=case.id, question=case.question) as span:
        try:
            # Fresh session per case: cases must not leak context into each other.
            result = await assistant.ask(case.question, session=SQLiteSession(f"eval-{case.id}"))
        except AgentError as e:
            span.set_attributes({"passed": False, "error": str(e)})
            return CaseResult(
                id=case.id, passed=False, tools_ok=False, days_ok=False, numbers_ok=False, text_ok=False,
                tools_called=[], expected_values=expected_values, latency_ms=None, answer="", error=str(e),
            )

        called = [c.name for c in result.tool_calls]
        if case.expected_tools:
            tools_ok = set(case.expected_tools) <= set(called)
        else:
            tools_ok = case.allow_tools or not called

        days_ok = case.expected_days is None or all(
            json.loads(c.args or "{}").get("days") == case.expected_days
            for c in result.tool_calls
            if c.name in case.expected_tools
        )
        numbers_ok = all(number_present(ground_truth(readings, r), result.answer) for r in case.numbers)
        answer_lower = result.answer.lower()
        text_ok = all(str(ground_truth(readings, r)).lower() in answer_lower for r in case.text) and (
            not case.must_contain_any or any(w.lower() in answer_lower for w in case.must_contain_any)
        )
        passed = tools_ok and days_ok and numbers_ok and text_ok
        span.set_attributes(
            {"passed": passed, "tools_ok": tools_ok, "days_ok": days_ok, "numbers_ok": numbers_ok,
             "text_ok": text_ok, "latency_ms": result.latency_ms}
        )

    return CaseResult(
        id=case.id, passed=passed, tools_ok=tools_ok, days_ok=days_ok, numbers_ok=numbers_ok, text_ok=text_ok,
        tools_called=called, expected_values=expected_values, latency_ms=result.latency_ms, answer=result.answer,
    )


def mark(ok: bool) -> str:
    return "✅" if ok else "❌"


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", default="", help="Only run cases whose id contains this string.")
    args = parser.parse_args()

    setup_observability()
    cases = [Case.model_validate_json(line) for line in (HERE / "testset.jsonl").read_text().splitlines() if line.strip()]
    cases = [c for c in cases if args.k in c.id]
    assistant = CGMAssistant()
    model = str((await assistant.agent()).model.model)

    results: list[CaseResult] = []
    with logfire.span("eval.run", model=model, n_cases=len(cases)) as run_span:
        for case in cases:  # sequential: one local model, one request at a time
            result = await run_case(assistant, case)
            results.append(result)
            print(f"{mark(result.passed)} {case.id:<16} tools={mark(result.tools_ok)} days={mark(result.days_ok)} "
                  f"numbers={mark(result.numbers_ok)} text={mark(result.text_ok)} "
                  f"{result.latency_ms or '-':>6} ms  {result.error or ''}")
        n_passed = sum(r.passed for r in results)
        pass_rate = round(100 * n_passed / len(results), 1) if results else 0.0
        run_span.set_attributes({"passed": n_passed, "pass_rate": pass_rate})

    latencies = [r.latency_ms for r in results if r.latency_ms]
    print(f"\nModel: {model}   Pass rate: {n_passed}/{len(results)} ({pass_rate}%)   "
          f"Median latency: {sorted(latencies)[len(latencies) // 2] if latencies else '-'} ms")
    (HERE / "results.json").write_text(
        json.dumps({"model": model, "pass_rate": pass_rate, "cases": [r.model_dump() for r in results]}, indent=2)
    )
    for r in results:
        if not r.passed:
            print(f"\n--- {r.id} (expected {r.expected_values}, tools {r.tools_called})\n{r.answer or r.error}")


if __name__ == "__main__":
    asyncio.run(main())
