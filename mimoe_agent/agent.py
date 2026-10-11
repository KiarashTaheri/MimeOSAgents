"""The CGM agent: OpenAI Agents SDK as the framework, mimOE as the (only) inference engine.

    user msg -> Runner -> LLM on mimOE -> tool_calls (run concurrently, locally) -> results -> ... -> answer

The coupling to mimOE is a single object: an AsyncOpenAI client whose base_url is mimOE's
OpenAI-compatible endpoint, wrapped in OpenAIChatCompletionsModel. That is the "bring your own
framework" contract; swapping the framework would not change anything on the mimOE side.
"""

import re
import time
from typing import Any

import logfire
from agents import (
    Agent,
    MaxTurnsExceeded,
    ModelBehaviorError,
    ModelSettings,
    OpenAIChatCompletionsModel,
    RunContextWrapper,
    Runner,
    Session,
    ToolCallItem,
    ToolCallOutputItem,
)
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, NotFoundError
from pydantic import BaseModel

from . import cgm
from .config import Settings, get_settings
from .tools import CGM_TOOLS, CGMContext

INSTRUCTIONS = """\
You are a friendly wellness assistant that helps the user understand their own continuous glucose \
monitor (CGM) data. All glucose values are in mmol/L. The target range is 3.9-10.0 mmol/L.

Rules:
- For ANY question about the user's glucose numbers, call a tool first. Never guess or calculate numbers yourself; \
quote the values the tool returns, with units.
- If a question needs several tools, call them together in one step.
- Report GMI / estimated A1c as a percentage (gmi_pct, e.g. "6.3%"); you may add the mmol/mol value after it.
- Pick `days` from the question: today/yesterday/last 24h = 1, this week = 7, two weeks = 14. Default to 7.
- Keep answers short: 2-4 sentences in plain language, then at most one practical, non-prescriptive observation.
- You are not a clinician. Do not diagnose conditions or recommend medication or insulin doses; for those, \
suggest talking to their healthcare provider.

The most recent CGM reading is from {latest:%A %Y-%m-%d %H:%M}. Treat that as "now".\
"""

NUDGE = "Please answer my previous question directly, using a tool if you need data."
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class AgentError(Exception):
    """The agent could not produce an answer (bad model output, turn limit, server error)."""


class MimoeUnavailableError(AgentError):
    """mimOE could not be reached or has no usable model loaded."""


class ToolCallRecord(BaseModel):
    name: str
    args: str
    ok: bool


class AgentResult(BaseModel):
    answer: str
    tool_calls: list[ToolCallRecord]
    latency_ms: int
    model: str


def strip_think(text: str | None) -> str:
    """Remove Qwen3-style reasoning (<think>...</think>), which mimOE returns inside `content`."""
    if not text:
        return ""
    text = _THINK_RE.sub("", text)
    if "</think>" in text:  # opening tag consumed by the chat template
        text = text.split("</think>", 1)[1]
    return text.strip()


def create_client(settings: Settings) -> AsyncOpenAI:
    """OpenAI client aimed at mimOE. Retry logic lives here: the SDK retries connection errors,
    timeouts, 408/409/429 and 5xx with exponential backoff and jitter, up to `max_retries` times."""
    return AsyncOpenAI(
        base_url=settings.base_url,
        api_key=settings.api_key,
        max_retries=settings.max_retries,
        timeout=settings.timeout_s,
    )


async def resolve_model(client: AsyncOpenAI, settings: Settings) -> str:
    """Use the configured model, or whichever model is currently loaded in mimOE Studio."""
    try:
        available = [m.id for m in (await client.models.list()).data]
    except (APIConnectionError, APITimeoutError) as e:
        raise MimoeUnavailableError(
            f"mimOE is not reachable at {settings.base_url}. Is mimOE Studio running with a model loaded?"
        ) from e
    if settings.model:
        if available and settings.model not in available:
            raise MimoeUnavailableError(f"Model '{settings.model}' is not loaded in mimOE. Available: {available}")
        return settings.model
    if not available:
        raise MimoeUnavailableError("mimOE has no model loaded. Load one in mimOE Studio's Model view.")
    return available[0]


def _instructions(settings: Settings):
    def render(ctx: RunContextWrapper[CGMContext], agent: Agent[CGMContext]) -> str:
        prompt = INSTRUCTIONS.format(latest=ctx.context.readings[-1].timestamp)
        # Qwen3 soft switch: skip the reasoning phase for ~5x lower latency on-device. Other models ignore it.
        return prompt if settings.thinking else prompt + "\n/no_think"

    return render


def build_agent(client: AsyncOpenAI, model: str, settings: Settings) -> Agent[CGMContext]:
    return Agent[CGMContext](
        name="CGM Wellness Assistant",
        instructions=_instructions(settings),
        model=OpenAIChatCompletionsModel(model=model, openai_client=client),
        model_settings=ModelSettings(temperature=settings.temperature, parallel_tool_calls=True),
        tools=CGM_TOOLS,
    )


def _tool_records(items: list[Any]) -> list[ToolCallRecord]:
    calls = [i.raw_item for i in items if isinstance(i, ToolCallItem)]
    outputs = {
        _get(i.raw_item, "call_id"): str(i.output) for i in items if isinstance(i, ToolCallOutputItem)
    }
    return [
        ToolCallRecord(
            name=_get(c, "name"),
            args=_get(c, "arguments"),
            ok=not outputs.get(_get(c, "call_id"), "").startswith("Tool error"),
        )
        for c in calls
    ]


def _get(raw: Any, key: str) -> Any:
    return raw.get(key) if isinstance(raw, dict) else getattr(raw, key, None)


async def run_agent(
    agent: Agent[CGMContext],
    message: str,
    context: CGMContext,
    settings: Settings,
    session: Session | None = None,
) -> AgentResult:
    """Answer one user turn, with error mapping and a Logfire span carrying success metrics."""
    started = time.perf_counter()
    model_name = str(getattr(agent.model, "model", agent.model))
    with logfire.span("agent.run", model=model_name, question=message) as span:
        try:
            result = await Runner.run(agent, message, context=context, session=session, max_turns=settings.max_steps)
            answer = strip_think(result.final_output)
            items = list(result.new_items)
            if not answer:
                # Small models occasionally end a turn with nothing but an empty <think> block.
                logfire.warn("empty model response, nudging once")
                result = await Runner.run(agent, NUDGE, context=context, session=session, max_turns=settings.max_steps)
                answer = strip_think(result.final_output)
                items += result.new_items
            if not answer:
                raise AgentError("The model returned an empty response twice.")
        except (APIConnectionError, APITimeoutError) as e:
            span.set_attributes({"success": False, "error": "mimoe_unreachable"})
            raise MimoeUnavailableError(
                f"mimOE is not reachable at {settings.base_url} (after {settings.max_retries} retries)."
            ) from e
        except NotFoundError as e:
            span.set_attributes({"success": False, "error": "model_not_found"})
            raise MimoeUnavailableError(f"Model '{model_name}' was not found on mimOE.") from e
        except APIStatusError as e:
            span.set_attributes({"success": False, "error": f"http_{e.status_code}"})
            raise AgentError(f"mimOE returned HTTP {e.status_code}: {e.message}") from e
        except MaxTurnsExceeded as e:
            span.set_attributes({"success": False, "error": "max_turns"})
            raise AgentError(f"No final answer after {settings.max_steps} model calls.") from e
        except ModelBehaviorError as e:
            span.set_attributes({"success": False, "error": "model_behavior"})
            raise AgentError(f"The model produced an invalid response: {e}") from e
        except AgentError as e:
            span.set_attributes({"success": False, "error": str(e)})
            raise

        records = _tool_records(items)
        latency_ms = int((time.perf_counter() - started) * 1000)
        span.set_attributes(
            {
                "success": True,
                "n_tool_calls": len(records),
                "tool_errors": sum(not r.ok for r in records),
                "tools_used": [r.name for r in records],
                "latency_ms": latency_ms,
                "answer": answer,
            }
        )
    return AgentResult(answer=answer, tool_calls=records, latency_ms=latency_ms, model=model_name)


class CGMAssistant:
    """Everything wired together: data, mimOE client, model, agent. Used by the CLI, API and evals."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        data = cgm.load_cgm(self.settings.cgm_path)
        self.context = CGMContext(readings=data.readings)
        self.skipped_rows = data.skipped_rows
        self.client = create_client(self.settings)
        self._agent: Agent[CGMContext] | None = None

    async def agent(self) -> Agent[CGMContext]:
        """Resolve the model lazily, so the API can start (and report health) while mimOE is down."""
        if self._agent is None:
            model = await resolve_model(self.client, self.settings)
            self._agent = build_agent(self.client, model, self.settings)
        return self._agent

    async def ask(self, message: str, session: Session | None = None) -> AgentResult:
        return await run_agent(await self.agent(), message, self.context, self.settings, session)
