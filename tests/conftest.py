from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import unikin


class FakeItem(SimpleNamespace):
    """Ausgabe-Element einer Responses-API-Antwort."""

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return dict(vars(self))


class FakeModel:
    """Steht für den OpenAI-Client: liefert eine feste JSON-Antwort und merkt sich die Prompts.

    Solange `tool_requests` Einträge hat und Werkzeuge angeboten werden, antwortet er
    stattdessen mit einem Werkzeugaufruf (name, arguments als dict oder Rohtext). Danach
    liefert er der Reihe nach die Texte aus `raw_outputs` statt JSON, jeden einmal.
    """

    def __init__(self) -> None:
        self.payload: dict[str, Any] = {
            "thought": "Testgedanke",
            "state_update": "Testzustand",
            "next_focus": "Testfokus",
            "urgency": "medium",
        }
        self.error: Exception | None = None
        self.tool_requests: list[tuple[str, dict[str, Any] | str]] = []
        self.raw_outputs: list[str] = []
        self.requests: list[dict[str, Any]] = []
        self.responses = self

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.requests.append(kwargs)
        if self.error is not None:
            raise self.error
        if self.tool_requests and kwargs.get("tools"):
            name, arguments = self.tool_requests.pop(0)
            call = FakeItem(
                type="function_call",
                name=name,
                arguments=arguments if isinstance(arguments, str) else json.dumps(arguments),
                call_id=f"call_{len(self.requests)}",
            )
            return SimpleNamespace(output=[call], output_text="")
        if self.raw_outputs:
            return SimpleNamespace(output=[FakeItem(type="message")], output_text=self.raw_outputs.pop(0))
        return SimpleNamespace(
            output=[FakeItem(type="message")],
            output_text=json.dumps(self.payload, ensure_ascii=False),
        )

    def user_payload(self, index: int = -1) -> dict[str, Any]:
        return json.loads(self.requests[index]["input"][1]["content"])


@pytest.fixture
def config(tmp_path: Path) -> unikin.AgentConfig:
    return unikin.AgentConfig(
        openai_base_url="http://127.0.0.1:9/v1",
        openai_api_key="test",
        openai_model="test-model",
        openai_timeout=1.0,
        identity="Unikin",
        mission="Testmission",
        min_interval=0.1,
        max_interval=1.0,
        default_interval=0.5,
        memory_file=tmp_path / "memory" / "unikin_state.json",
        log_file=tmp_path / "logs" / "unikin.log",
    )


@pytest.fixture
def fake_model() -> FakeModel:
    return FakeModel()


@pytest.fixture(autouse=True)
def no_retry_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(unikin, "RETRY_BACKOFF_SECONDS", 0.0)
