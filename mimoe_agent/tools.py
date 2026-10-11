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

# The mock data covers 14 days, which is also the standard length of a CGM report.
MIN_DAYS, MAX_DAYS = 1, 14

Days = Annotated[
    int,
    Field(
        ge=MIN_DAYS,
        le=MAX_DAYS,
        description="How many days back to look, counted from the latest reading. "
        "1 = last 24 hours / 'today' / 'yesterday', 7 = 'this week', 14 = 'two weeks' (max).",
    ),
]

_DAYS_VALIDATOR = TypeAdapter(Days)


@dataclass
class CGMContext:
    readings: list[cgm.GlucoseReading]


def tool_error_message(ctx: RunContextWrapper[CGMContext], error: Exception) -> str:
    """Send the real reason back to the model (e.g. 'days must be <= 14') so it can fix its call."""
    # For privacy the SDK strips argument details from validation errors ("Invalid JSON input for tool X").
    # All tools share the same `days` argument, so re-check it here to tell the model what to fix.
    raw_arguments = getattr(ctx, "tool_arguments", None)
    reason = (_explain_invalid_arguments(raw_arguments) if raw_arguments is not None else None) or str(error)
    logfire.warn("tool call failed: {reason}", reason=reason)
    return f"Tool error: {reason}. Fix the arguments and try again, or tell the user the data isn't available."


def _explain_invalid_arguments(raw_arguments: str) -> str | None:
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError:
        return 'arguments are not valid JSON; send an object like {"days": 7}'
    if not isinstance(arguments, dict):
        return 'arguments must be a JSON object like {"days": 7}'
    try:
        _DAYS_VALIDATOR.validate_python(arguments.get("days"))
    except ValidationError as validation_error:
        return "days: " + "; ".join(detail["msg"] for detail in validation_error.errors())
    return None


@function_tool(failure_error_function=tool_error_message)
def get_average_glucose(ctx: RunContextWrapper[CGMContext], days: Days) -> cgm.AverageStats:
    """Average glucose (mmol/L), variability (SD, CV%), GMI (estimated HbA1c), and the lowest and
    highest single readings with their timestamps."""
    return cgm.average(ctx.context.readings, days)


@function_tool(failure_error_function=tool_error_message)
def get_time_in_range(ctx: RunContextWrapper[CGMContext], days: Days) -> cgm.RangeStats:
    """Percent of time in target range (3.9-10.0 mmol/L), below and above range, hours in range,
    in-range % by time of day (overnight/morning/afternoon/evening) with the best and worst time of day,
    and in-range % per day. Use for any 'how much' or 'when' am I in range question."""
    return cgm.time_in_range(ctx.context.readings, days)


@function_tool(failure_error_function=tool_error_message)
def get_low_events(ctx: RunContextWrapper[CGMContext], days: Days) -> cgm.EventReport:
    """Low glucose episodes (below 3.9 mmol/L for at least 15 minutes): count, total minutes,
    how many dropped below 3.0, the most common time of day, and the most recent episodes with
    start/end time, duration and lowest value."""
    return _keep_most_recent_events(cgm.find_events(ctx.context.readings, days, "low"))


@function_tool(failure_error_function=tool_error_message)
def get_high_events(ctx: RunContextWrapper[CGMContext], days: Days) -> cgm.EventReport:
    """High glucose episodes (above 10.0 mmol/L for at least 15 minutes): count, total minutes,
    how many went above 13.9, the most common time of day, and the most recent episodes with
    start/end time, duration and peak value."""
    return _keep_most_recent_events(cgm.find_events(ctx.context.readings, days, "high"))


def _keep_most_recent_events(report: cgm.EventReport) -> cgm.EventReport:
    return report.model_copy(update={"events": report.events[-MAX_EVENTS_RETURNED:]})


CGM_TOOLS = [get_average_glucose, get_time_in_range, get_low_events, get_high_events]
