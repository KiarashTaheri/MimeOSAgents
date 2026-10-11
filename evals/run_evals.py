"""Run the eval test set against the live agent on mimOE and report a pass rate.

    uv run python evals/run_evals.py            # all cases
    uv run python evals/run_evals.py -k lows    # cases whose id contains "lows"

Each case is scored on:
- tools:   the expected tools were tools_called (and none were tools_called when none are expected)
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

# Values are rounded to 1 decimal, so a correct quote is within 0.05; the extra 0.001 absorbs float error.
# Tight enough that "7" does not match an expected 6.9.
NUMBER_TOLERANCE = 0.051


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


def ground_truth(readings: list[cgm.GlucoseReading], field_ref: FieldRef) -> Any:
    metric_by_name = {
        "average": lambda: cgm.average(readings, field_ref.days),
        "time_in_range": lambda: cgm.time_in_range(readings, field_ref.days),
        "low_events": lambda: cgm.find_events(readings, field_ref.days, "low"),
        "high_events": lambda: cgm.find_events(readings, field_ref.days, "high"),
    }
    value: Any = metric_by_name[field_ref.metric]()
    for attribute in field_ref.field.split("."):
        value = getattr(value, attribute)
    return value


def extract_numbers(text: str) -> list[float]:
    return [float(number) for number in re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))]


def number_present(expected: float, answer: str) -> bool:
    # Strict: the prompt tells the model to quote tool values (1 decimal). A loose tolerance would let
    # unrelated numbers ("7 days") satisfy an expected 6.9.
    return any(abs(number - expected) < NUMBER_TOLERANCE for number in extract_numbers(answer))


async def run_case(assistant: CGMAssistant, case: Case) -> CaseResult:
    readings = assistant.context.readings
    expected_values = {
        f"{ref.metric}.{ref.field}@{ref.days}d": ground_truth(readings, ref) for ref in case.numbers + case.text
    }

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

        tools_called = [call.name for call in result.tool_calls]
        if case.expected_tools:
            tools_ok = set(case.expected_tools) <= set(tools_called)
        else:
            tools_ok = case.allow_tools or not tools_called

        days_ok = case.expected_days is None or all(
            json.loads(call.args or "{}").get("days") == case.expected_days
            for call in result.tool_calls
            if call.name in case.expected_tools
        )
        numbers_ok = all(number_present(ground_truth(readings, ref), result.answer) for ref in case.numbers)
        answer_lower = result.answer.lower()
        text_ok = all(str(ground_truth(readings, ref)).lower() in answer_lower for ref in case.text) and (
            not case.must_contain_any or any(word.lower() in answer_lower for word in case.must_contain_any)
        )
        passed = tools_ok and days_ok and numbers_ok and text_ok
        span.set_attributes(
            {"passed": passed, "tools_ok": tools_ok, "days_ok": days_ok, "numbers_ok": numbers_ok,
             "text_ok": text_ok, "latency_ms": result.latency_ms}
        )

    return CaseResult(
        id=case.id, passed=passed, tools_ok=tools_ok, days_ok=days_ok, numbers_ok=numbers_ok, text_ok=text_ok,
        tools_called=tools_called, expected_values=expected_values, latency_ms=result.latency_ms, answer=result.answer,
    )


def status_icon(ok: bool) -> str:
    return "✅" if ok else "❌"


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", default="", help="Only run cases whose id contains this string.")
    args = parser.parse_args()

    setup_observability()
    cases = [Case.model_validate_json(line) for line in (HERE / "testset.jsonl").read_text().splitlines() if line.strip()]
    cases = [case for case in cases if args.k in case.id]
    assistant = CGMAssistant()
    model = str((await assistant.agent()).model.model)

    results: list[CaseResult] = []
    with logfire.span("eval.run", model=model, n_cases=len(cases)) as run_span:
        for case in cases:  # sequential: one local model, one request at a time
            result = await run_case(assistant, case)
            results.append(result)
            print(f"{status_icon(result.passed)} {case.id:<16} tools={status_icon(result.tools_ok)} "
                  f"days={status_icon(result.days_ok)} numbers={status_icon(result.numbers_ok)} text={status_icon(result.text_ok)} "
                  f"{result.latency_ms or '-':>6} ms  {result.error or ''}")
        n_passed = sum(result.passed for result in results)
        pass_rate = round(100 * n_passed / len(results), 1) if results else 0.0
        run_span.set_attributes({"passed": n_passed, "pass_rate": pass_rate})

    latencies = [result.latency_ms for result in results if result.latency_ms]
    print(f"\nModel: {model}   Pass rate: {n_passed}/{len(results)} ({pass_rate}%)   "
          f"Median latency: {sorted(latencies)[len(latencies) // 2] if latencies else '-'} ms")
    (HERE / "results.json").write_text(
        json.dumps(
            {"model": model, "pass_rate": pass_rate, "cases": [result.model_dump() for result in results]}, indent=2
        )
    )
    for result in results:
        if not result.passed:
            print(f"\n--- {result.id} (expected {result.expected_values}, tools {result.tools_called})\n"
                  f"{result.answer or result.error}")


if __name__ == "__main__":
    asyncio.run(main())
