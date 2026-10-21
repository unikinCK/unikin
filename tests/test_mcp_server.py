from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import anyio
import pytest
from mcp import Client, StdioServerParameters

import mcp_server
import unikin

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def fast_polling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp_server, "STEP_POLL_SECONDS", 0.05)


def call_tool(config, name, arguments=None):
    async def main():
        async with Client(mcp_server.create_server(config)) as client:
            return await client.call_tool(name, arguments or {})

    return anyio.run(main)


@pytest.mark.parametrize("mode", ["legacy", "auto"])
def test_tools_are_listed_with_annotations(config, mode):
    async def main():
        async with Client(mcp_server.create_server(config), mode=mode) as client:
            return (await client.list_tools()).tools

    tools = {tool.name: tool for tool in anyio.run(main)}

    assert set(tools) == {"get_state", "get_journal", "search_journal", "send_message", "step"}
    assert tools["get_state"].annotations.read_only_hint is True
    assert tools["send_message"].annotations.read_only_hint is False


def test_send_message_queues_into_inbox(config):
    result = call_tool(config, "send_message", {"text": "Hallo Unikin", "sender": "Test"})

    assert not result.is_error
    message_id = result.structured_content["message_id"]
    assert (unikin.inbox_dir(config) / f"{message_id}.json").exists()


def test_send_message_rejects_empty_text(config):
    result = call_tool(config, "send_message", {"text": "   "})

    assert result.is_error
    assert not list(unikin.inbox_dir(config).glob("*.json"))


def test_missing_state_is_a_tool_error(config):
    result = call_tool(config, "get_state")

    assert result.is_error
    assert "Noch kein Zustand" in result.content[0].text


def test_get_journal_returns_newest_entries(config, fake_model):
    state = unikin.load_memory(config)
    for _ in range(3):
        unikin.run_cycle(config, fake_model, state, [])

    result = call_tool(config, "get_journal", {"limit": 2})

    assert len(result.structured_content["result"]) == 2


def test_search_journal_finds_archived_entries(config, fake_model):
    state = unikin.load_memory(config)
    unikin.run_cycle(config, fake_model, state, [])
    fake_model.payload["thought"] = "Kuramoto mit N=20 simuliert"
    unikin.run_cycle(config, fake_model, state, [])

    result = call_tool(config, "search_journal", {"query": "Kuramoto"})

    assert not result.is_error
    (hit,) = result.structured_content["result"]
    assert (hit["loop"], hit["thought"]) == (2, "Kuramoto mit N=20 simuliert")
    assert "Kuramoto" in hit["match"]


def test_search_journal_without_archive_is_a_tool_error(config):
    result = call_tool(config, "search_journal", {"query": "Kuramoto"})

    assert result.is_error
    assert "Noch kein Langzeitgedächtnis" in result.content[0].text
    # Der Server liest nur: Anlegen ist Sache des Loops.
    assert not unikin.archive_path(config).exists()


def test_step_returns_the_cycle_that_read_the_note(config, fake_model):
    fake_model.payload["reply"] = "Verstanden"
    unikin.load_memory(config)

    async def fake_loop():
        while not list(unikin.inbox_dir(config).glob("*.json")):
            await anyio.sleep(0.05)
        state = unikin.load_memory(config)
        await anyio.to_thread.run_sync(unikin.run_cycle, config, fake_model, state, unikin.read_inbox(config))

    async def main():
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(fake_loop)
            async with Client(mcp_server.create_server(config)) as client:
                return await client.call_tool("step", {"note": "Bitte bestätigen", "timeout_seconds": 10})

    result = anyio.run(main)

    assert result.structured_content["status"] == "done"
    assert result.structured_content["entry"]["reply"] == "Verstanden"
    assert [m["text"] for m in fake_model.user_payload()["inbox"]] == ["Bitte bestätigen"]


def test_step_without_running_loop_reports_pending(config):
    result = call_tool(config, "step", {"note": "Hallo?", "timeout_seconds": 1})

    assert result.structured_content["status"] == "pending"
    assert len(unikin.read_inbox(config)) == 1


def server_env(config):
    return {"MEMORY_FILE": str(config.memory_file), "LOG_FILE": str(config.log_file)}


@pytest.mark.parametrize("mode", ["legacy", "auto"])
def test_stdio_transport_end_to_end(config, tmp_path, mode):
    unikin.load_memory(config)
    server = StdioServerParameters(
        command=sys.executable,
        args=[str(ROOT / "mcp_server.py")],
        env=server_env(config),
        cwd=str(tmp_path),
    )

    async def main():
        async with Client(server, mode=mode) as client:
            return await client.call_tool("get_state", {})

    result = anyio.run(main)

    assert not result.is_error
    assert result.structured_content["identity"] == "Unikin"
    assert config.log_file.with_name("unikin-mcp.log").exists()


def test_sigterm_stops_server_while_it_waits_on_stdin(config, tmp_path):
    # Im Container ist der Server PID 1: ohne eigenen Handler ignoriert er SIGTERM, und
    # `docker stop` müsste die Karenzzeit abwarten.
    process = subprocess.Popen(
        [sys.executable, str(ROOT / "mcp_server.py")],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, **server_env(config)},
        cwd=tmp_path,
    )
    try:
        mcp_log = config.log_file.with_name("unikin-mcp.log")
        deadline = time.monotonic() + 15
        while not (mcp_log.exists() and "MCP-Server gestartet" in mcp_log.read_text(encoding="utf-8")):
            assert time.monotonic() < deadline, "Server ist nicht gestartet"
            time.sleep(0.1)

        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=5) == 0
    finally:
        if process.poll() is None:
            process.kill()
        for stream in (process.stdin, process.stdout, process.stderr):
            stream.close()
