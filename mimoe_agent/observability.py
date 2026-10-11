"""Pydantic Logfire setup.

Traces are only exported when a Logfire token is present (`logfire auth` + `logfire projects use <name>`,
or LOGFIRE_TOKEN), so the app runs fine without one. The Agents SDK's default exporter (which sends traces
to OpenAI's platform) is removed: with mimOE everything stays local, and Logfire is our only trace sink.
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
        scrubbing=False,  # glucose values are the point of the traces; the data is synthetic
    )
    set_trace_processors([])  # drop the OpenAI-platform exporter
    logfire.instrument_openai_agents()  # agent / generation / tool spans -> Logfire
    _configured = True
