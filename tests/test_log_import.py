from __future__ import annotations

import json
import logging
import sqlite3
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

import journal_archive
import unikin
from journal_archive import SAME_CYCLE_SECONDS, JournalArchive, import_log_files, parse_log

# Gekürzt aus logs/unikin.log vom Mac Studio: Die Werkzeuge stehen vor der Loop-Zeile ihres
# Zyklus, eine Antwort kann mehrzeilig sein, und vor Loop 190 fehlte „| inbox=… | tools=…“.
LOG = """\
2026-10-20 21:20:01 | INFO | Starte Unikin mit Modell 'qwen/qwen3.6-35b-a3b' via http://host.docker.internal:1234/v1
2026-10-20 21:20:01 | INFO | Werkzeuge: http://runner:8812/mcp (höchstens 3 Aufrufe pro Zyklus)
2026-10-20 21:20:30 | INFO | HTTP Request: POST http://runner:8812/mcp "HTTP/1.1 200 OK"
2026-10-20 21:20:31 | INFO | Werkzeug run_python: Exit-Code: 0 STDOUT: System | K | Min r Kuramoto N=20 | 2.0 | 0.734
2026-10-20 21:20:52 | INFO | Werkzeug run_python (Fehler): Exit-Code: 1 STDERR: NameError: name 'np' is not defined
2026-10-20 21:21:10 | INFO | HTTP Request: POST http://host.docker.internal:1234/v1/responses "HTTP/1.1 200 OK"
2026-10-20 21:21:10 | INFO | Loop 461 | urgency=medium | sim=0.21 | api=40.12s | next=300.00s | inbox=1 | tools=2
2026-10-20 21:21:10 | INFO | Thought: Ab K=2.0 synchronisieren die Oszillatoren spontan.
2026-10-20 21:21:10 | INFO | Next focus: Phasendiagramm über K erweitern
2026-10-20 21:21:10 | INFO | Reply: Ergebnisse:
- K=0.5: r=0.12

- K=2.0: r=0.73
2026-10-20 21:26:15 | INFO | Loop 462 | urgency=low | sim=0.40 | api=5.01s | next=300.00s | inbox=0 | tools=0
2026-10-20 21:26:15 | INFO | Thought: Die Inbox ist leer.
2026-10-20 21:26:15 | INFO | Next focus: Weiter auf Signale warten.
2026-10-20 21:30:00 | INFO | Werkzeug run_python: Exit-Code: 0 STDOUT: abgebrochen
2026-10-20 21:30:02 | INFO | Signal 15 empfangen. Kontrollierter Shutdown. Ende.
2026-10-20 21:30:03 | INFO | Starte Unikin mit Modell 'qwen/qwen3.6-35b-a3b' via http://host.docker.internal:1234/v1
2026-10-20 21:30:20 | INFO | Loop 463 | urgency=low | sim=0.00 | api=17.69s | next=43.22s
2026-10-20 21:30:20 | INFO | Thought: Neustart, die Sandbox prüfe ich erneut.
2026-10-20 21:30:20 | INFO | Next focus: Sandbox prüfen
"""


def journal_entry(ts: str, thought: str, **extra) -> dict:
    return {"ts": ts, "thought": thought, "state_update": "Zustand aus dem Journal", "next_focus": "", **extra}


@pytest.fixture
def log_file(tmp_path):
    path = tmp_path / "logs" / "unikin.log"
    path.parent.mkdir()
    path.write_text(LOG, encoding="utf-8")
    return path


@pytest.fixture
def archive(config) -> JournalArchive:
    return JournalArchive(unikin.archive_path(config))


def rows(archive: JournalArchive) -> list[dict]:
    with sqlite3.connect(archive.path) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute("SELECT * FROM entries ORDER BY ts")]


def test_log_lines_become_journal_like_entries():
    first, second, third = parse_log(LOG.split("\n"))

    assert first == {
        "ts": "2026-10-20T21:21:10+00:00",
        "loop": 461,
        "urgency": "medium",
        "thought": "Ab K=2.0 synchronisieren die Oszillatoren spontan.",
        "next_focus": "Phasendiagramm über K erweitern",
        "reply": "Ergebnisse:\n- K=0.5: r=0.12\n\n- K=2.0: r=0.73",
        "tool_calls": [
            {
                "name": "run_python",
                "arguments": "",
                "output": "Exit-Code: 0 STDOUT: System | K | Min r Kuramoto N=20 | 2.0 | 0.734",
                "is_error": False,
            },
            {
                "name": "run_python",
                "arguments": "",
                "output": "Exit-Code: 1 STDERR: NameError: name 'np' is not defined",
                "is_error": True,
            },
        ],
    }
    assert (second["loop"], second["tool_calls"], "reply" in second) == (462, [], False)
    # Der Werkzeugaufruf vor dem Neustart gehörte zu einem abgebrochenen Zyklus.
    assert (third["loop"], third["urgency"], third["tool_calls"]) == (463, "low", [])
    assert third["thought"] == "Neustart, die Sandbox prüfe ich erneut."


def test_log_times_are_converted_to_utc():
    (entry, *_) = parse_log(LOG.split("\n"), ZoneInfo("Europe/Berlin"))
    assert entry["ts"] == "2026-10-20T19:21:10+00:00"


def test_parser_reads_what_the_loop_logs(config, fake_model, runner, archive):
    """Der Parser muss das Format von run_cycle und execute_tool_call verstehen, nicht nur das Beispiel."""
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    unikin.setup_logging(config.log_file)
    try:
        fake_model.tool_requests = [("run_python", {"code": "print(1)"}), ("search_memory", {"query": "Kuramoto"})]
        fake_model.payload["reply"] = "Erste Zeile\nZweite Zeile"
        state = unikin.load_memory(config)
        unikin.run_cycle(config, fake_model, state, [], runner_target=runner)
        unikin.run_cycle(config, fake_model, state, [], runner_target=runner)
    finally:
        for handler in root.handlers:
            handler.close()
        root.handlers[:] = handlers
        root.setLevel(level)

    journal = unikin.load_memory(config)["journal"]
    # logging schreibt Ortszeit; im Container ist das UTC, auf dem Testrechner vielleicht nicht.
    local = datetime.now().astimezone().tzinfo
    parsed = parse_log(config.log_file.read_text(encoding="utf-8").split("\n"), local)
    assert len(parsed) == len(journal) == 2
    for from_log, from_journal in zip(parsed, journal):
        assert {key: from_log[key] for key in ("loop", "urgency", "thought", "next_focus", "reply")} == {
            key: from_journal[key] for key in ("loop", "urgency", "thought", "next_focus", "reply")
        }
        offset = datetime.fromisoformat(from_journal["ts"]) - datetime.fromisoformat(from_log["ts"])
        assert abs(offset.total_seconds()) < SAME_CYCLE_SECONDS
        assert [(call["name"], call["is_error"]) for call in from_log["tool_calls"]] == [
            (call["name"], call["is_error"]) for call in from_journal.get("tool_calls", [])
        ]
    run_python = journal[0]["tool_calls"][0]
    assert parsed[0]["tool_calls"][0]["output"] == unikin.clean_text(run_python["output"])

    # Beide Zyklen hat der Loop schon archiviert, der Import findet sie über loop wieder.
    assert import_log_files(archive, [config.log_file], local) == (2, 0)
    assert archive.count() == 2


def test_import_skips_cycles_the_journal_already_archived(archive, log_file):
    archive.add(
        [
            # Journal-Einträge von vor dem Archiv haben keine Zyklusnummer: gleicher Gedanke, ts knapp danach.
            journal_entry("2026-10-20T21:26:15.431907+00:00", "Die Inbox ist leer."),
            journal_entry("2026-10-20T21:30:20.120000+00:00", "Anderer Wortlaut", loop=463),
            # Von Hand eingetragen, 14 s nach Loop 461, aber kein Zyklus: Loop 461 fehlt im Archiv.
            journal_entry("2026-10-20T21:21:24.785042+00:00", "Docker-MCP-Test", note="docker-mcp-test"),
        ]
    )

    assert import_log_files(archive, [log_file]) == (3, 1)
    assert import_log_files(archive, [log_file]) == (3, 0)
    assert [(row["loop"], row["source"]) for row in rows(archive)] == [
        (461, "log"),
        (None, "journal"),
        (None, "journal"),
        (463, "journal"),
    ]


def test_journal_entry_replaces_its_cycle_from_the_log(archive, log_file):
    import_log_files(archive, [log_file])
    later = journal_entry("2026-10-20T21:30:20.733000+00:00", "Neustart, die Sandbox prüfe ich erneut.")
    other_loop = journal_entry("2026-10-20T21:26:16.100000+00:00", "Die Inbox ist leer.", loop=999)

    assert archive.add([later, other_loop]) == 2

    by_ts = {row["ts"]: row for row in rows(archive)}
    assert set(by_ts) == {"2026-10-20T21:21:10+00:00", "2026-10-20T21:26:15+00:00", later["ts"], other_loop["ts"]}
    assert by_ts[later["ts"]]["state_update"] == "Zustand aus dem Journal"
    # Eine andere Zyklusnummer ist ein anderer Zyklus, auch bei gleichem Gedanken.
    assert by_ts["2026-10-20T21:26:15+00:00"]["source"] == "log"
    assert import_log_files(archive, [log_file]) == (3, 0)


def test_rotated_logs_are_read_oldest_first(archive, log_file):
    older = log_file.with_name("unikin.log.1")
    lines = LOG.split("\n")
    # Rotation zwischen den Werkzeugaufrufen und der Loop-Zeile von Loop 461.
    older.write_text("\n".join(lines[:5]) + "\n", encoding="utf-8")
    log_file.write_text("\n".join(lines[5:]), encoding="utf-8")

    assert import_log_files(archive, [log_file, older]) == (3, 3)
    (hit,) = archive.search("Kuramoto", 5)
    assert hit["loop"] == 461
    assert "run_python\n→ Exit-Code: 0" in hit["tool_calls"]


def test_imported_cycles_are_found_and_shown_with_their_thought(config, archive, log_file):
    import_log_files(archive, [log_file])

    (hit,) = json.loads(unikin.search_memory(config, {"query": "Kuramoto"}))
    assert hit["loop"] == 461
    assert hit["thought"] == "Ab K=2.0 synchronisieren die Oszillatoren spontan."
    assert "state_update" not in hit
    assert "0.734" in hit["match"]


def test_archive_from_version_1_gets_the_source_column(archive, log_file):
    archive.add([journal_entry("2026-10-19T08:00:00.500000+00:00", "Alt")])
    with sqlite3.connect(archive.path) as conn:
        conn.executescript("ALTER TABLE entries DROP COLUMN source; PRAGMA user_version = 1;")

    assert archive.search("Alt", 5, readonly=True)[0]["thought"] == "Alt"
    assert import_log_files(archive, [log_file]) == (3, 3)

    assert [row["source"] for row in rows(archive)] == ["journal", "log", "log", "log"]
    with sqlite3.connect(archive.path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == journal_archive.SCHEMA_VERSION


def test_command_line_import(monkeypatch, capsys, archive, log_file):
    argv = ["journal_archive.py", "--archive", str(archive.path), "import-log", str(log_file), "--log-timezone"]
    monkeypatch.setattr(sys, "argv", [*argv, "Europe/Berlin"])
    journal_archive.main()
    assert "3 Zyklen im Log, 0 davon standen schon im Archiv.\n3 neue Einträge archiviert" in capsys.readouterr().out
    assert rows(archive)[0]["ts"] == "2026-10-20T19:21:10+00:00"

    journal_archive.main()
    assert "3 davon standen schon im Archiv.\n0 neue Einträge archiviert, insgesamt 3" in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", [*argv, "Mars/Olympus"])
    with pytest.raises(SystemExit):
        journal_archive.main()
    assert "unbekannte Zeitzone" in capsys.readouterr().err
