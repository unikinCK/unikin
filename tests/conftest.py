from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import unikin


class FakeModel:
    """Steht für den OpenAI-Client: liefert eine feste JSON-Antwort und merkt sich die Prompts."""

    def __init__(self) -> None:
        self.payload: dict[str, Any] = {
            "thought": "Testgedanke",
            "state_update": "Testzustand",
            "next_focus": "Testfokus",
            "urgency": "medium",
        }
        self.error: Exception | None = None
        self.requests: list[dict[str, Any]] = []
        self.responses = self

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.requests.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(output_text=json.dumps(self.payload, ensure_ascii=False))

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
