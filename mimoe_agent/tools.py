"""CGM tools for the agent.

Each tool is a plain typed function decorated with the Agents SDK's @function_tool:
- the signature (with pydantic `Annotated` constraints) becomes the JSON schema the LLM sees,
- the docstring becomes the tool description,
- the same signature validates the LLM's arguments at call time.

The loaded readings arrive through the run context, so no globals and nothing data-related in the schema.
Numbers are computed deterministically in `cgm.py`; the LLM only chooses tools and explains results.
"""

import json
from dataclasses import dataclass
from typing import Annotated

import logfire
from agents import RunContextWrapper, function_tool
from pydantic import Field, TypeAdapter, ValidationError

from . import cgm

# Tool output goes back into a ~12k token context; keep event lists bounded.
MAX_EVENTS_RETURNED = 10

Days = Annotated[
    int,
    Field(
        ge=1,
        le=14,
        description="How many days back to look, counted from the latest reading. "
        "1 = last 24 hours / 'today' / 'yesterday', 7 = 'this week', 14 = 'two weeks' (max).",
    ),
]

_DAYS = TypeAdapter(Days)


@dataclass
class CGMContext:
    readings: list[cgm.GlucoseReading]


def tool_error(ctx: RunContextWrapper[CGMContext], error: Exception) -> str:
    """Send the real reason back to the model (e.g. 'days must be <= 14') so it can fix its call."""
    # For privacy the SDK strips argument details from validation errors ("Invalid JSON input for tool X").
    # All tools share the same `days` argument, so re-check it here to tell the model what to fix.
    raw_args = getattr(ctx, "tool_arguments", None)
    reason = (_explain_bad_args(raw_args) if raw_args is not None else None) or str(error)
    logfire.warn("tool call failed: {reason}", reason=reason)
    return f"Tool error: {reason}. Fix the arguments and try again, or tell the user the data isn't available."


def _explain_bad_args(raw_args: str) -> str | None:
    try:
        args = json.loads(raw_args)
    except json.JSONDecodeError:
        return 'arguments are not valid JSON; send an object like {"days": 7}'
    if not isinstance(args, dict):
        return 'arguments must be a JSON object like {"days": 7}'
    try:
        _DAYS.validate_python(args.get("days"))
    except ValidationError as e:
        return "days: " + "; ".join(err["msg"] for err in e.errors())
    return None


@function_tool(failure_error_function=tool_error)
def get_average_glucose(ctx: RunContextWrapper[CGMContext], days: Days) -> cgm.AverageStats:
    """Average glucose (mmol/L), variability (SD, CV%), GMI (estimated HbA1c), and the lowest and
    highest single readings with their timestamps."""
    return cgm.average(ctx.context.readings, days)


@function_tool(failure_error_function=tool_error)
def get_time_in_range(ctx: RunContextWrapper[CGMContext], days: Days) -> cgm.RangeStats:
    """Percent of time in target range (3.9-10.0 mmol/L), below and above range, hours in range,
    in-range % by time of day (overnight/morning/afternoon/evening) with the best and worst time of day,
    and in-range % per day. Use for any 'how much' or 'when' am I in range question."""
    return cgm.time_in_range(ctx.context.readings, days)


@function_tool(failure_error_function=tool_error)
def get_low_events(ctx: RunContextWrapper[CGMContext], days: Days) -> cgm.EventReport:
    """Low glucose episodes (below 3.9 mmol/L for at least 15 minutes): count, total minutes,
    how many dropped below 3.0, the most common time of day, and the most recent episodes with
    start/end time, duration and lowest value."""
    return _most_recent(cgm.find_events(ctx.context.readings, days, "low"))


@function_tool(failure_error_function=tool_error)
def get_high_events(ctx: RunContextWrapper[CGMContext], days: Days) -> cgm.EventReport:
    """High glucose episodes (above 10.0 mmol/L for at least 15 minutes): count, total minutes,
    how many went above 13.9, the most common time of day, and the most recent episodes with
    start/end time, duration and peak value."""
    return _most_recent(cgm.find_events(ctx.context.readings, days, "high"))


def _most_recent(report: cgm.EventReport) -> cgm.EventReport:
    return report.model_copy(update={"events": report.events[-MAX_EVENTS_RETURNED:]})


CGM_TOOLS = [get_average_glucose, get_time_in_range, get_low_events, get_high_events]
