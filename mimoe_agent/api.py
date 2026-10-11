"""HTTP API for the CGM agent.

    uv run uvicorn mimoe_agent.api:app --reload      # Swagger UI at http://127.0.0.1:8000/docs
"""

import uuid
from contextlib import asynccontextmanager

import logfire
from agents import SQLiteSession
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from .agent import AgentError, CGMAssistant, MimoeUnavailableError, ToolCallRecord
from .observability import setup_observability


MAX_MESSAGE_CHARS = 2000  # keeps one question well inside the model's ~12k-token context


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS, examples=["When am I usually in range?"])
    session_id: str | None = Field(default=None, description="Reuse to continue a conversation.")


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    tool_calls: list[ToolCallRecord]
    latency_ms: int
    model: str


class HealthResponse(BaseModel):
    status: str
    mimoe_reachable: bool
    model: str | None
    cgm_readings: int
    detail: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.assistant = CGMAssistant()
    app.state.sessions: dict[str, SQLiteSession] = {}  # in-memory; fine for a single local process
    yield


setup_observability()  # before instrument_fastapi, which needs a configured Logfire
app = FastAPI(title="mimOE CGM Agent", version="0.1.0", lifespan=lifespan)
logfire.instrument_fastapi(app)


def _get_assistant(request: Request) -> CGMAssistant:
    return request.app.state.assistant


@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request) -> ChatResponse:
    sessions: dict[str, SQLiteSession] = request.app.state.sessions
    session_id = body.session_id or uuid.uuid4().hex
    session = sessions.setdefault(session_id, SQLiteSession(session_id))
    try:
        result = await _get_assistant(request).ask(body.message, session=session)
    except MimoeUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except AgentError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    return ChatResponse(session_id=session_id, **result.model_dump())


@app.delete("/chat/{session_id}", status_code=204)
async def reset_session(session_id: str, request: Request) -> None:
    session = request.app.state.sessions.pop(session_id, None)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session_id")


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    assistant = _get_assistant(request)
    reading_count = len(assistant.context.readings)
    try:
        agent = await assistant.agent()
    except MimoeUnavailableError as e:
        return HealthResponse(status="degraded", mimoe_reachable=False, model=None, cgm_readings=reading_count, detail=str(e))
    return HealthResponse(status="ok", mimoe_reachable=True, model=str(agent.model.model), cgm_readings=reading_count)
