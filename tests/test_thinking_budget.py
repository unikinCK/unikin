from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import httpx2
import pytest
from openai import APITimeoutError

import unikin


class TimeoutAfterBudgetHit:
    """Erste Antwort am Budget abgeschnitten, die Nachfrage ohne Denken läuft ins Timeout."""

    def __init__(self, fake_model, url: str) -> None:
        self.fake_model = fake_model
        self.url = url
        self.requests = fake_model.requests
        self.responses = self

    def create(self, **kwargs):
        if self.requests:
            self.requests.append(kwargs)
            raise APITimeoutError(request=httpx2.Request("POST", self.url))
        self.fake_model.budget_hits = 1
        return self.fake_model.create(**kwargs)


def test_every_request_carries_the_token_budget(config, fake_model):
    state = unikin.load_memory(config)

    unikin.run_cycle(config, fake_model, state, [])

    (request,) = fake_model.requests
    assert request["max_output_tokens"] == config.max_output_tokens
    assert "reasoning" not in request


def test_budget_zero_sends_no_limit(config, fake_model):
    config = dataclasses.replace(config, max_output_tokens=0)
    state = unikin.load_memory(config)

    unikin.run_cycle(config, fake_model, state, [])

    assert "max_output_tokens" not in fake_model.requests[0]


def test_budget_hit_is_asked_once_more_without_thinking(config, fake_model):
    fake_model.budget_hits = 1
    message = unikin.post_message(config, "Bist du da?", "Test")
    state = unikin.load_memory(config)

    unikin.run_cycle(config, fake_model, state, unikin.read_inbox(config))

    first, second = fake_model.requests
    assert "reasoning" not in first
    assert second["reasoning"] == {"effort": "none"}
    assert second["input"] == first["input"]
    entry = unikin.load_memory(config)["journal"][-1]
    assert entry["parse_failed"] is False
    assert entry["thought"] == "Testgedanke"
    assert entry["inbox_ids"] == [message["id"]]
    assert unikin.read_inbox(config) == []


def test_budget_hit_twice_ends_cycle_with_honest_error(config, fake_model):
    fake_model.budget_hits = 2
    message = unikin.post_message(config, "Bist du da?", "Test")
    state = unikin.load_memory(config)

    unikin.run_cycle(config, fake_model, state, unikin.read_inbox(config))

    assert len(fake_model.requests) == 2
    entry = unikin.load_memory(config)["journal"][-1]
    assert entry["parse_failed"] is True
    assert entry["thought"].startswith("Modell hat das Denkbudget überschritten")
    assert "API-Aufruf" not in entry["thought"]
    assert [m["id"] for _, m in unikin.read_inbox(config)] == [message["id"]]


def test_timeout_of_follow_up_still_reports_thinking_budget(config, fake_model):
    client = TimeoutAfterBudgetHit(fake_model, config.openai_base_url)
    state = unikin.load_memory(config)

    unikin.run_cycle(config, client, state, [])

    assert len(fake_model.requests) == 2
    thought = unikin.load_memory(config)["journal"][-1]["thought"]
    assert "Denkbudget überschritten" in thought
    assert "Zeitlimit" in thought
    assert "timed out" not in thought


def test_timeout_is_not_reported_as_api_failure(config, fake_model):
    fake_model.error = APITimeoutError(request=httpx2.Request("POST", config.openai_base_url))
    state = unikin.load_memory(config)

    unikin.run_cycle(config, fake_model, state, [])

    thought = unikin.load_memory(config)["journal"][-1]["thought"]
    assert thought.startswith("Modell hat innerhalb des Zeitlimits nicht geantwortet")
    assert "timed out" not in thought


def test_connection_error_is_still_reported_as_api_failure(config, fake_model):
    fake_model.error = ConnectionError("LM Studio nicht erreichbar")
    state = unikin.load_memory(config)

    unikin.run_cycle(config, fake_model, state, [])

    thought = unikin.load_memory(config)["journal"][-1]["thought"]
    assert thought == "Fehler beim API-Aufruf: LM Studio nicht erreichbar"


def test_budget_hit_during_tool_phase_keeps_tools_on_follow_up(config, fake_model, runner):
    fake_model.budget_hits = 1
    fake_model.tool_requests = [("run_python", {"code": "print(6 * 7)"})]
    state = unikin.load_memory(config)

    unikin.run_cycle(config, fake_model, state, [], runner_target=runner)

    hit, follow_up, *rest = fake_model.requests
    assert [tool["name"] for tool in follow_up["tools"]] == ["run_python"]
    assert follow_up["reasoning"] == {"effort": "none"}
    # Nach dem Werkzeugaufruf denkt das Modell wieder wie gewohnt.
    assert rest and all("reasoning" not in request for request in rest)
    entry = unikin.load_memory(config)["journal"][-1]
    assert entry["tool_calls"][0]["name"] == "run_python"
    assert entry["parse_failed"] is False


@pytest.mark.parametrize(
    ("response", "exhausted"),
    [
        # LM Studio: hört bei max_output_tokens - 1 auf und meldet trotzdem "completed".
        (SimpleNamespace(status="completed", usage=SimpleNamespace(output_tokens=11999)), True),
        (SimpleNamespace(status="completed", usage=SimpleNamespace(output_tokens=11998)), False),
        # OpenAI-API: status "incomplete" mit Grund.
        (
            SimpleNamespace(
                status="incomplete",
                incomplete_details=SimpleNamespace(reason="max_output_tokens"),
                usage=SimpleNamespace(output_tokens=10),
            ),
            True,
        ),
        (SimpleNamespace(output_text="{}"), False),
    ],
)
def test_budget_exhausted_detection(response, exhausted):
    assert unikin.budget_exhausted(response, 12000) is exhausted


def test_budget_exhausted_without_limit():
    response = SimpleNamespace(status="completed", usage=SimpleNamespace(output_tokens=50000))

    assert unikin.budget_exhausted(response, 0) is False


def test_config_default_and_override(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_MAX_OUTPUT_TOKENS", raising=False)
    assert unikin.load_config().max_output_tokens == 12000

    monkeypatch.setenv("OPENAI_MAX_OUTPUT_TOKENS", "4000")
    assert unikin.load_config().max_output_tokens == 4000

    monkeypatch.setenv("OPENAI_MAX_OUTPUT_TOKENS", "0")
    assert unikin.load_config().max_output_tokens == 0
