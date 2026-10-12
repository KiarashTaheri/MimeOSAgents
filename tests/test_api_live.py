"""Live API tests: FastAPI -> real agent -> real mimOE. No fakes.

The happy-path tests skip automatically when mimOE isn't reachable (MIMOE_BASE_URL in .env).
The 503 test needs no mimOE at all: it points the app at a closed port, which is real failure behaviour.

    uv run pytest -m live        # only these
    uv run pytest -m "not live"  # fast offline tests only
"""

import json

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from mimoe_agent import api
from mimoe_agent.config import get_settings

pytestmark = pytest.mark.live

# Port 9 (discard) has nothing listening, so connecting fails immediately: a real "mimOE is down".
CLOSED_PORT_URL = "http://127.0.0.1:9/mimik-ai/openai/v1"


def mimoe_reachable() -> bool:
    try:
        settings = get_settings()
    except ValidationError:  # no .env (e.g. a fresh clone): skip live tests instead of breaking collection
        return False
    headers = {"Authorization": f"Bearer {settings.api_key.get_secret_value()}"}
    try:
        httpx.get(f"{settings.base_url}/models", headers=headers, timeout=3)
    except httpx.HTTPError:
        return False
    return True


needs_mimoe = pytest.mark.skipif(not mimoe_reachable(), reason="mimOE is not reachable at MIMOE_BASE_URL")


@pytest.fixture
def settings_env(monkeypatch):
    """Apply env overrides to a fresh Settings (get_settings is cached)."""

    def apply(**env: str) -> None:
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        get_settings.cache_clear()

    yield apply
    get_settings.cache_clear()


@needs_mimoe
def test_health_reports_loaded_model():
    with TestClient(api.app) as client:
        body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["mimoe_reachable"] is True
    assert body["model"]
    assert body["cgm_readings"] > 0


@needs_mimoe
def test_chat_answers_with_tools_and_keeps_session():
    with TestClient(api.app) as client:
        first = client.post("/chat", json={"message": "What percentage of time was I in range this week?"})
        assert first.status_code == 200, first.text
        first = first.json()
        assert "get_time_in_range" in [c["name"] for c in first["tool_calls"]]

        # The follow-up only makes sense with the previous turn in context.
        second = client.post(
            "/chat",
            json={"message": "And what about over two weeks?", "session_id": first["session_id"]},
        )
        assert second.status_code == 200, second.text
        second = second.json()

    assert second["session_id"] == first["session_id"]
    calls = second["tool_calls"]
    assert any(c["name"] == "get_time_in_range" and json.loads(c["args"]).get("days") == 14 for c in calls), calls


def test_mimoe_down_returns_503(settings_env):
    settings_env(MIMOE_BASE_URL=CLOSED_PORT_URL, MIMOE_MAX_RETRIES="0", MIMOE_TIMEOUT_S="3")
    with TestClient(api.app) as client:
        chat = client.post("/chat", json={"message": "How are my lows?"})
        health = client.get("/health").json()
    assert chat.status_code == 503
    assert "not reachable" in chat.json()["detail"]
    assert health["status"] == "degraded" and health["mimoe_reachable"] is False


def test_empty_message_is_rejected(settings_env):
    settings_env(MIMOE_BASE_URL=CLOSED_PORT_URL)
    with TestClient(api.app) as client:
        assert client.post("/chat", json={"message": ""}).status_code == 422
