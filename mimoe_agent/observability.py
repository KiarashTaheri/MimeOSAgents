"""Pydantic Logfire setup.

Traces are only exported when a Logfire token is present (`logfire auth` + `logfire projects use <name>`,
or LOGFIRE_TOKEN), so the app runs fine without one. The Agents SDK's default exporter (which sends traces
to OpenAI's platform) is removed: with mimOE everything stays local, and Logfire is our only trace sink.

Traces show execution, not health data: which steps ran, which tools were called, timings and errors.
Questions, answers, prompts, tool arguments and outputs (glucose values) are never recorded:
- agent spans: the runner uses `RunConfig(trace_include_sensitive_data=False)` (see agent.py),
- FastAPI spans: `request_attributes_mapper` drops the request body,
- our own spans only carry counts, names, timings and error types.
"""

import logfire
from agents import set_trace_processors

from .config import get_settings

_configured = False


def setup_observability(service_name: str = "mimoe-cgm-agent") -> None:
    global _configured
    if _configured:
        return
    token = get_settings().logfire_token
    logfire.configure(
        token=token.get_secret_value() if token else None,  # None -> falls back to LOGFIRE_TOKEN / `logfire auth`
        service_name=service_name,
        send_to_logfire="if-token-present",
        console=False,  # keep the CLI clean; view traces in the Logfire UI
        # Default scrubbing stays on as a second layer (redacts anything that looks like a secret).
    )
    set_trace_processors([])  # drop the OpenAI-platform exporter
    logfire.instrument_openai_agents()  # agent / generation / tool spans -> Logfire
    _configured = True
