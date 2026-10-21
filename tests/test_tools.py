from __future__ import annotations

import dataclasses

import unikin


def test_tool_result_reaches_model_and_journal(config, fake_model, runner):
    fake_model.tool_requests = [("run_python", {"code": "print(6 * 7)"})]
    state = unikin.load_memory(config)

    unikin.run_cycle(config, fake_model, state, [], runner_target=runner)

    first, second = fake_model.requests
    assert [tool["name"] for tool in first["tools"]] == ["run_python"]
    assert "run_python" in first["input"][0]["content"]
    assert runner.executed == ["print(6 * 7)"]
    assert second["input"][-1]["type"] == "function_call_output"
    assert "12 Zeichen ausgeführt" in second["input"][-1]["output"]

    entry = unikin.load_memory(config)["journal"][-1]
    assert entry["tool_calls"][0]["name"] == "run_python"
    assert "12 Zeichen ausgeführt" in entry["tool_calls"][0]["output"]
    assert entry["thought"] == "Testgedanke"


def test_exhausted_budget_ends_with_fresh_request_without_tools(config, fake_model, runner):
    config = dataclasses.replace(config, max_tool_calls=2)
    fake_model.tool_requests = [("run_python", {"code": f"print({index})"}) for index in range(5)]
    state = unikin.load_memory(config)

    unikin.run_cycle(config, fake_model, state, [], runner_target=runner)

    assert len(runner.executed) == 2
    final = fake_model.requests[-1]
    assert "tools" not in final
    assert len(final["input"]) == 2
    assert [call["arguments"] for call in fake_model.user_payload()["tool_results"]] == [
        '{"code": "print(0)"}',
        '{"code": "print(1)"}',
    ]
    assert len(unikin.load_memory(config)["journal"][-1]["tool_calls"]) == 2


# So antwortet Qwen manchmal, wenn es nach Werkzeugaufrufen von selbst aufhört.
NON_JSON_ANSWER = 'thought: "Die Rechnung ergibt 42."\nstate_update: "Rechnung geprüft."'


def test_non_json_answer_after_tools_gets_fresh_request(config, fake_model, runner):
    # Wie im Betrieb: zwei Aufrufe bei Budget 3, dann hört das Modell ohne JSON auf.
    fake_model.tool_requests = [("run_python", {"code": f"print({index})"}) for index in range(2)]
    fake_model.raw_outputs = [NON_JSON_ANSWER]
    state = unikin.load_memory(config)

    unikin.run_cycle(config, fake_model, state, [], runner_target=runner)

    assert len(fake_model.requests) == 4
    assert "tools" in fake_model.requests[2]
    final = fake_model.requests[-1]
    assert "tools" not in final
    assert len(final["input"]) == 2
    assert [call["arguments"] for call in fake_model.user_payload()["tool_results"]] == [
        '{"code": "print(0)"}',
        '{"code": "print(1)"}',
    ]

    saved = unikin.load_memory(config)
    entry = saved["journal"][-1]
    assert entry["parse_failed"] is False
    assert (entry["thought"], entry["state_update"], entry["next_focus"]) == ("Testgedanke", "Testzustand", "Testfokus")
    assert len(entry["tool_calls"]) == 2
    assert saved["metrics"]["consecutive_errors"] == 0


def test_non_json_answer_without_tool_calls_gets_no_extra_request(config, fake_model, runner):
    fake_model.raw_outputs = [NON_JSON_ANSWER]
    state = unikin.load_memory(config)

    unikin.run_cycle(config, fake_model, state, [], runner_target=runner)

    (request,) = fake_model.requests
    assert "tools" in request
    entry = unikin.load_memory(config)["journal"][-1]
    assert entry["parse_failed"] is True
    assert "tool_calls" not in entry


def test_non_json_final_answer_is_not_requested_twice(config, fake_model, runner):
    config = dataclasses.replace(config, max_tool_calls=1)
    fake_model.tool_requests = [("run_python", {"code": "print(1)"})]
    fake_model.raw_outputs = [NON_JSON_ANSWER]
    state = unikin.load_memory(config)

    unikin.run_cycle(config, fake_model, state, [], runner_target=runner)

    assert len(fake_model.requests) == 2
    assert "tools" not in fake_model.requests[-1]
    assert unikin.load_memory(config)["journal"][-1]["parse_failed"] is True


def test_invalid_tool_arguments_are_reported_to_model(config, fake_model, runner):
    fake_model.tool_requests = [("run_python", "{kein json")]
    state = unikin.load_memory(config)

    unikin.run_cycle(config, fake_model, state, [], runner_target=runner)

    assert runner.executed == []
    call = unikin.load_memory(config)["journal"][-1]["tool_calls"][0]
    assert call["is_error"] is True
    assert "kein gültiges JSON" in call["output"]


def test_unreachable_runner_runs_cycle_without_tools(config, fake_model, caplog):
    state = unikin.load_memory(config)

    unikin.run_cycle(config, fake_model, state, [], runner_target="http://127.0.0.1:9/mcp")

    (request,) = fake_model.requests
    assert "tools" not in request
    assert "Werkzeuge" not in request["input"][0]["content"]
    entry = unikin.load_memory(config)["journal"][-1]
    assert entry["thought"] == "Testgedanke"
    assert "tool_calls" not in entry
    (warning,) = [record.getMessage() for record in caplog.records if "Runner nicht erreichbar" in record.getMessage()]
    assert "TaskGroup" not in warning


def test_no_tools_when_budget_is_zero(config, fake_model, runner):
    config = dataclasses.replace(config, max_tool_calls=0)
    fake_model.tool_requests = [("run_python", {"code": "print(1)"})]
    state = unikin.load_memory(config)

    unikin.run_cycle(config, fake_model, state, [], runner_target=runner)

    assert "tools" not in fake_model.requests[0]
    assert runner.executed == []
