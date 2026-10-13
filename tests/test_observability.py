"""Traces must show execution (spans, tool names, timings) but never health data or conversation text.

The agent runs end to end through the real OpenAI client and OpenAIChatCompletionsModel; only the HTTP
layer is replaced by a stand-in mimOE that asks for one tool call and then answers.
"""

import json

import httpx
import logfire
import pytest
from agents import set_trace_processors
from openai import AsyncOpenAI

from mimoe_agent.agent import build_agent, run_agent
from mimoe_agent.config import Settings
from mimoe_agent.tools import CGMContext
from tests.test_cgm import series

QUESTION = "What is my average glucose this week? QUESTION-MARKER"
ANSWER = "Your average glucose was 7.3 mmol/L. ANSWER-MARKER"
READING = 7.3  # every reading, so the tool output (mean, min, max) contains 7.3


def fake_mimoe(request: httpx.Request) -> httpx.Response:
    messages = json.loads(request.content)["messages"]
    if not any(message["role"] == "tool" for message in messages):
        reply = {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "get_average_glucose", "arguments": '{"days": 7}'},
        }]}
    else:
        reply = {"role": "assistant", "content": ANSWER}
    return httpx.Response(200, json={
        "id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": "fake",
        "choices": [{"index": 0, "message": reply, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    })


@pytest.mark.anyio
async def test_traces_record_execution_but_not_values(capfire):
    set_trace_processors([])
    logfire.instrument_openai_agents()
    settings = Settings(api_key="test", _env_file=None)
    client = AsyncOpenAI(
        base_url="http://mimoe.test/v1", api_key="test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(fake_mimoe)),
    )
    agent = build_agent(client, "fake", settings)

    result = await run_agent(agent, QUESTION, CGMContext(readings=series([READING] * 288)), settings)
    assert result.answer == ANSWER  # the run really happened, including the tool call

    spans = capfire.exporter.exported_spans_as_dict()
    recorded = json.dumps(spans)
    names = " ".join(span["name"] for span in spans)

    # Execution is visible: the agent run, the model calls and the tool call.
    assert "agent.run" in names
    assert "get_average_glucose" in recorded
    # Health data and conversation text are not.
    for secret in ("QUESTION-MARKER", "ANSWER-MARKER", str(READING), "mean_mmol_l"):
        assert secret not in recorded, f"{secret!r} leaked into a trace"
