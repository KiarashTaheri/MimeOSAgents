import json

import pytest
from agents.tool_context import ToolContext

from mimoe_agent.agent import strip_think
from mimoe_agent.tools import CGM_TOOLS, CGMContext, get_low_events
from tests.test_cgm import series


async def invoke(tool, args: str):
    ctx = ToolContext(
        context=CGMContext(readings=series([5.0, 3.5, 3.4, 3.3, 3.6, 5.0])),
        tool_name=tool.name,
        tool_call_id="call_1",
        tool_arguments=args,
    )
    return await tool.on_invoke_tool(ctx, args)


def test_schemas_expose_days_constraints_only():
    for tool in CGM_TOOLS:
        props = tool.params_json_schema["properties"]
        assert list(props) == ["days"]  # the run context is not leaked into the schema
        assert props["days"]["minimum"] == 1 and props["days"]["maximum"] == 14


@pytest.mark.anyio
async def test_valid_call_returns_metrics():
    out = await invoke(get_low_events, '{"days": 1}')
    assert json.loads(out.model_dump_json() if hasattr(out, "model_dump_json") else out)["count"] == 1


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("args", "reason"),
    [
        ('{"days": 40}', "less than or equal to 14"),
        ('{"days": "lots"}', "valid integer"),
        ("not json", "not valid JSON"),
    ],
)
async def test_invalid_args_are_explained_to_the_model(args, reason):
    out = await invoke(get_low_events, args)
    assert isinstance(out, str) and out.startswith("Tool error")
    assert reason in out  # the model needs the actual reason to fix its call


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("<think>\n\n</think>\n\nHi there!", "Hi there!"),
        ("<think>reasoning...</think>Answer", "Answer"),
        ("reasoning only closed</think> Answer", "Answer"),
        ("Plain answer", "Plain answer"),
        (None, ""),
    ],
)
def test_strip_think(raw, clean):
    assert strip_think(raw) == clean
