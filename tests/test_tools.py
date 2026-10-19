from __future__ import annotations

import dataclasses

import pytest
from mcp.server.mcpserver import MCPServer

import unikin


@pytest.fixture
def runner():
    """Steht für den Python-Runner: ein echter MCP-Server im selben Prozess, der nichts ausführt."""
    server = MCPServer(name="Python Code Runner")
    server.executed = []

    @server.tool()
    def run_python(code: str) -> str:
        """Führt Python-Code aus."""
        server.executed.append(code)
        return f"Exit-Code: 0\nSTDOUT:\n{len(code)} Zeichen ausgeführt\nSTDERR:\n"

    return server


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
