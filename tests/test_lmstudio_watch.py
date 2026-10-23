from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

import unikin
from lmstudio_watch import (
    LOG_TRUNCATION_CHARS,
    OWN_REQUEST_MARKER,
    ForeignRequestGuard,
    LmStudioBusyError,
    LmStudioLog,
    YieldingClient,
)

# So schreibt LM Studio Anfragen ins Serverlog (gekürzt nach Mustern vom Mac Studio, 22.10.2026).
UNIKIN_REQUEST = """\
[{ts}][DEBUG] Received request: POST to /v1/responses with body {{
  "model": "qwen/qwen3.5-122b-a10b",
  "input": [
    {{
      "role": "system",
      "content": "Du bist Unikin, eine kontinuierlich arbeitende nic... <Truncated in logs> ...eigenen früheren Ergebnisse, keine neuen Eingaben."
    }},
    {{
      "role": "user",
      "content": "{{\\"identity\\": \\"Unikin\\", \\"mission\\": \\"Konti... <Truncated in logs> ...d konkret.\\"}}"
    }}
  ],
  "temperature": 0.3
}}
[{ts}][INFO][qwen/qwen3.5-122b-a10b] Generating synchronous v1/responses response...
"""

FOREIGN_REQUEST = """\
[{ts}][DEBUG] Received request: POST to /v1/chat/completions with body {{
  "model": "qwen/qwen3.5-122b-a10b",
  "messages": [
    {{
      "role": "system",
      "content": "Du bist ein striktes visuelles Prüfsystem für Werk... <Truncated in logs> ...lich mit dem zugehörigen Gutbild derselben Kamera."
    }}
  ],
  "temperature": 0
}}
[{ts}][INFO][qwen/qwen3.5-122b-a10b] Running chat completion on conversation with 2 messages.
"""

# LibreChat schickt die Werkzeuge vor den Nachrichten, und deren Beschreibungen nennen Unikin.
LIBRECHAT_REQUEST = """\
[{ts}][DEBUG] Received request: POST to /v1/chat/completions with body {{
  "model": "qwen/qwen3.6-35b-a3b",
  "user": "6ad396230ae13b4462a79c57",
  "stream": true,
  "tools": [
    {{
      "type": "function",
      "function": {{
        "name": "get_state_mcp_unikin",
        "description": "Liefert Unikins gespeicherten Zustand: Identität, Mission"
      }}
    }}
  ],
  "messages": [
    {{
      "role": "user",
      "content": "Was macht Unikin gerade?"
    }}
  ]
}}
[{ts}][INFO][qwen/qwen3.6-35b-a3b] Running chat completion on conversation with 1 messages.
"""

EMBEDDING_REQUEST = """\
[{ts}][DEBUG] Received request: POST to /v1/embeddings with body {{
  "model": "text-embedding-embeddinggemma-300m-qat",
  "input": "Ein Absatz aus einem Dokument"
}}
[{ts}][INFO] Embeddings computed.
"""

OTHER_LINES = """\
[{ts}][INFO][qwen/qwen3.5-122b-a10b] Prompt processing progress: 100.0%
[{ts}][DEBUG] Received request: GET to /api/v0/models
[{ts}][INFO][LMSAuthenticator][Client=lms-cli][Endpoint=listLoaded] Listing loaded models
"""


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def log_file(directory: Path, name: str = "2026-10-22.1.log") -> Path:
    path = directory / name[:7] / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def append(path: Path, *blocks: str, ts: str = "2026-10-22 21:45:56") -> None:
    with path.open("a", encoding="utf-8") as handle:
        for block in blocks:
            handle.write(block.format(ts=ts))


def test_classifies_own_foreign_and_ignored_requests(tmp_path):
    path = log_file(tmp_path)
    append(path, UNIKIN_REQUEST, FOREIGN_REQUEST, LIBRECHAT_REQUEST, EMBEDDING_REQUEST, OTHER_LINES)

    requests = LmStudioLog(tmp_path).read_new()

    assert [(request.endpoint, request.own, request.foreign) for request in requests] == [
        ("/v1/responses", True, False),
        ("/v1/chat/completions", False, True),
        ("/v1/chat/completions", False, True),
        ("/v1/embeddings", False, False),
    ]
    assert requests[1].describe() == (
        "POST /v1/chat/completions (qwen/qwen3.5-122b-a10b, „Du bist ein striktes visuelles Prüfsyste…“)"
    )
    assert requests[2].prompt == "Was macht Unikin gerade?"
    assert requests[1].describe(with_prompt=False) == "POST /v1/chat/completions (qwen/qwen3.5-122b-a10b)"


def test_marker_starts_unikins_system_prompt_within_the_logged_part(config):
    system_prompt, _ = unikin.build_prompt(config, unikin.initial_state(config), tool_names=["run_python"])

    assert system_prompt.startswith(OWN_REQUEST_MARKER)
    assert len(OWN_REQUEST_MARKER) <= LOG_TRUNCATION_CHARS


def test_request_counts_once_its_block_is_complete(tmp_path):
    path = log_file(tmp_path)
    log = LmStudioLog(tmp_path)
    block = FOREIGN_REQUEST.format(ts="2026-10-22 21:45:56").encode()
    # Mitten im Body abgeschnitten, die Zeile danach erst halb geschrieben.
    split = block.index(b"[2026", 1) + 20

    with path.open("ab") as handle:
        handle.write(block[:split])
    assert log.read_new() == []
    with path.open("ab") as handle:
        handle.write(block[split:])
    assert [request.endpoint for request in log.read_new()] == ["/v1/chat/completions"]
    assert log.read_new() == []


def test_follows_log_into_the_next_file(tmp_path):
    first = log_file(tmp_path, "2026-10-22.9.log")
    append(first, OTHER_LINES)
    log = LmStudioLog(tmp_path)
    assert log.read_new() == []

    # Erst die alte Datei zu Ende lesen, dann die neue (10 nach 9, nicht nach Zeichen sortiert).
    append(first, FOREIGN_REQUEST)
    second = log_file(tmp_path, "2026-10-22.10.log")
    append(second, LIBRECHAT_REQUEST)

    assert [request.model for request in log.read_new()] == ["qwen/qwen3.5-122b-a10b", "qwen/qwen3.6-35b-a3b"]
    assert log.path == second


def test_foreign_request_pauses_for_cooldown(tmp_path):
    path = log_file(tmp_path)
    clock = Clock()
    guard = ForeignRequestGuard(LmStudioLog(tmp_path), cooldown=1200, clock=clock)
    guard.start()
    assert not guard.paused()

    append(path, UNIKIN_REQUEST, EMBEDDING_REQUEST)
    assert not guard.paused()

    append(path, FOREIGN_REQUEST)
    assert guard.paused()
    assert guard.remaining() == 1200
    assert guard.info()["reason"] == "Andere Anwendung nutzt LM Studio: POST /v1/chat/completions (qwen/qwen3.5-122b-a10b)"

    # Eine weitere fremde Anfrage verlängert die Pause, eigene nicht.
    clock.now += 600
    append(path, FOREIGN_REQUEST, ts="2026-10-22 21:55:56")
    append(path, UNIKIN_REQUEST, ts="2026-10-22 21:55:57")
    assert guard.paused()
    assert guard.remaining() == 1199

    clock.now += 1199
    assert not guard.paused()
    assert guard.info() is None


def test_start_counts_foreign_requests_from_before_the_restart(tmp_path):
    path = log_file(tmp_path)
    append(path, FOREIGN_REQUEST, FOREIGN_REQUEST, ts="2026-10-22 21:00:00")
    append(path, FOREIGN_REQUEST, ts="2026-10-22 21:40:00")
    append(path, UNIKIN_REQUEST, ts="2026-10-22 21:45:00")
    # Seit einer Minute steht nichts Neues im Log.
    os.utime(path, (time.time() - 60, time.time() - 60))

    guard = ForeignRequestGuard(LmStudioLog(tmp_path), cooldown=1200, clock=Clock())
    guard.start()

    assert guard.paused()
    assert 1200 - 360 - 5 < guard.remaining() <= 1200 - 360
    # Die beiden Anfragen von 21:00 liegen vor der Abklingzeit und zählen nicht zur Pause.
    assert guard.count == 1


def test_start_ignores_foreign_requests_older_than_cooldown(tmp_path):
    path = log_file(tmp_path)
    append(path, FOREIGN_REQUEST, ts="2026-10-22 21:00:00")
    append(path, UNIKIN_REQUEST, ts="2026-10-22 21:45:00")

    guard = ForeignRequestGuard(LmStudioLog(tmp_path), cooldown=1200, clock=Clock())
    guard.start()

    assert not guard.paused()


def test_guard_without_log_or_cooldown_never_pauses(tmp_path):
    append(log_file(tmp_path), FOREIGN_REQUEST)

    disabled = ForeignRequestGuard(LmStudioLog(tmp_path), cooldown=0)
    disabled.start()
    assert not disabled.paused()

    missing = ForeignRequestGuard(LmStudioLog(tmp_path / "fehlt"), cooldown=1200)
    missing.start()
    assert not missing.paused()


class ArrivingForeignRequest:
    """Steht für LM Studio: Während Unikins erster Anfrage fragt eine andere Anwendung an."""

    def __init__(self, model, path: Path) -> None:
        self.model = model
        self.path = path
        self.responses = self

    def create(self, **kwargs):
        response = self.model.create(**kwargs)
        append(self.path, FOREIGN_REQUEST)
        return response


def test_cycle_stops_before_next_request_when_another_app_arrives(config, fake_model, runner, tmp_path):
    path = log_file(tmp_path / "lmstudio")
    guard = ForeignRequestGuard(LmStudioLog(tmp_path / "lmstudio"), cooldown=1200)
    guard.start()
    client = YieldingClient(ArrivingForeignRequest(fake_model, path), guard)
    fake_model.tool_requests = [("run_python", {"code": "print(6 * 7)"})]
    state = unikin.load_memory(config)
    saved = config.memory_file.read_text(encoding="utf-8")

    with pytest.raises(LmStudioBusyError):
        unikin.run_cycle(config, client, state, [], runner_target=runner)

    # Kein zweiter Versuch, keine Schlussanfrage, und der Zyklus hinterlässt nichts im Zustand.
    assert len(fake_model.requests) == 1
    assert config.memory_file.read_text(encoding="utf-8") == saved


def test_note_pause_records_and_clears_pause():
    state: dict = {}
    pause = {"until": "2026-10-22T20:05:56+00:00", "reason": "Andere Anwendung nutzt LM Studio: POST /v1/chat/completions"}

    assert unikin.note_pause(state, pause)
    assert state["pause"] == pause
    assert not unikin.note_pause(state, dict(pause))
    assert unikin.note_pause(state, None)
    assert "pause" not in state
    assert not unikin.note_pause(state, None)
