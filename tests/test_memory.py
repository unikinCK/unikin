from __future__ import annotations

import dataclasses
import json

import pytest

import unikin
from journal_archive import JournalArchive, fts_query, import_state_files


def journal_entry(ts: str, thought: str, state_update: str = "Zustand", **extra) -> dict:
    return {
        "ts": ts,
        "thought": thought,
        "state_update": state_update,
        "next_focus": "Weiter",
        "urgency": "low",
        **extra,
    }


def archive(config) -> JournalArchive:
    return JournalArchive(unikin.archive_path(config))


KURAMOTO = journal_entry(
    "2026-10-20T21:28:00+00:00",
    "Die Skalierung auf N=20 Agenten im Kuramoto-Modell zeigt spontane Synchronisation.",
    "Kuramoto mit N=20 simuliert: Ordnungsparameter r=0.734.",
    tool_calls=[
        {
            "name": "run_python",
            "arguments": json.dumps({"code": "import numpy as np\nprint(kuramoto(N=20))"}),
            "output": "Exit-Code: 0\nSTDOUT:\nr=0.734\n",
            "is_error": False,
        }
    ],
)


def test_every_cycle_is_archived_with_its_loop_number(config, fake_model):
    state = unikin.load_memory(config)
    unikin.run_cycle(config, fake_model, state, [])
    fake_model.payload["thought"] = "Zweiter Gedanke"
    unikin.run_cycle(config, fake_model, state, [])

    journal = unikin.load_memory(config)["journal"]
    assert [entry["loop"] for entry in journal] == [1, 2]
    assert archive(config).count() == 2
    (hit,) = archive(config).search("Zweiter", 5)
    assert (hit["ts"], hit["loop"], hit["thought"]) == (journal[-1]["ts"], 2, "Zweiter Gedanke")


def test_existing_journal_is_backfilled_and_kept_after_the_journal_drops_it(config, fake_model):
    state = unikin.load_memory(config)
    state["journal"] = [KURAMOTO] + [
        journal_entry(f"2026-10-21T08:{minute:02d}:00+00:00", f"Leerlauf {minute}") for minute in range(39)
    ]
    unikin.save_memory(config, state)

    unikin.run_cycle(config, fake_model, state, [])
    unikin.archive_journal(config, state["journal"])

    journal = unikin.load_memory(config)["journal"]
    assert len(journal) == 40
    assert KURAMOTO["ts"] not in {entry["ts"] for entry in journal}
    assert archive(config).count() == 41
    (hit,) = archive(config).search("Kuramoto", 5)
    assert hit["ts"] == KURAMOTO["ts"]
    assert hit["loop"] is None
    assert "r=0.734" in hit["tool_calls"]


def test_broken_archive_never_stops_the_cycle(config, fake_model, caplog):
    # Ein Verzeichnis, wo die Datei liegen soll: SQLite kann sie weder öffnen noch anlegen.
    unikin.archive_path(config).mkdir(parents=True)
    fake_model.tool_requests = [("search_memory", {"query": "Kuramoto"})]
    state = unikin.load_memory(config)
    state["next_focus"] = "Kuramoto weiter untersuchen"

    unikin.run_cycle(config, fake_model, state, [])

    entry = unikin.load_memory(config)["journal"][-1]
    assert entry["thought"] == "Testgedanke"
    assert entry["tool_calls"][0]["is_error"] is True
    assert "related_memory" not in fake_model.user_payload(0)
    warnings = [record.getMessage() for record in caplog.records if record.levelname == "WARNING"]
    assert any("nicht archiviert" in message for message in warnings)
    assert any("related_memory bleibt leer" in message for message in warnings)


def test_search_memory_finds_old_results_without_asking_the_model(config, fake_model):
    unikin.archive_journal(config, [KURAMOTO, journal_entry("2026-10-21T08:00:00+00:00", "Inbox leer")])
    fake_model.tool_requests = [("search_memory", {"query": "Kuramoto Synchronisation", "limit": 2})]
    state = unikin.load_memory(config)

    unikin.run_cycle(config, fake_model, state, [])

    first, second = fake_model.requests
    assert [tool["name"] for tool in first["tools"]] == ["search_memory"]
    assert "search_memory durchsucht" in first["input"][0]["content"]
    (hit,) = json.loads(second["input"][-1]["output"])
    assert hit["ts"] == KURAMOTO["ts"]
    assert "r=0.734" in hit["state_update"]
    assert "Kuramoto" in hit["match"]

    call = unikin.load_memory(config)["journal"][-1]["tool_calls"][0]
    assert (call["name"], call["is_error"]) == ("search_memory", False)
    # Archiviert wird nur die Suche, nicht ihr Ergebnis: Sonst fände jede spätere Suche sich selbst.
    assert [hit["ts"] for hit in archive(config).search("0.734", 5)] == [KURAMOTO["ts"]]
    assert len(archive(config).search("Synchronisation", 5)) == 2


def test_search_memory_counts_toward_the_tool_budget(config, fake_model, runner):
    config = dataclasses.replace(config, max_tool_calls=2)
    fake_model.tool_requests = [
        ("search_memory", {"query": "Kuramoto"}),
        ("run_python", {"code": "print(1)"}),
        ("search_memory", {"query": "noch einmal"}),
    ]
    state = unikin.load_memory(config)

    unikin.run_cycle(config, fake_model, state, [], runner_target=runner)

    assert runner.executed == ["print(1)"]
    assert "tools" not in fake_model.requests[-1]
    names = [call["name"] for call in unikin.load_memory(config)["journal"][-1]["tool_calls"]]
    assert names == ["search_memory", "run_python"]


def test_search_memory_reports_empty_query_and_no_hits(config, fake_model):
    fake_model.tool_requests = [("search_memory", {"query": "  "}), ("search_memory", {"query": "Kuramoto"})]
    state = unikin.load_memory(config)

    unikin.run_cycle(config, fake_model, state, [])

    empty, no_hits = unikin.load_memory(config)["journal"][-1]["tool_calls"]
    assert empty["is_error"] is True and "query fehlt" in empty["output"]
    assert no_hits["is_error"] is False and no_hits["output"].startswith("Keine Treffer")


def test_related_memory_brings_back_older_entries_for_the_focus(config, fake_model):
    recent = [
        journal_entry(f"2026-10-21T09:{minute:02d}:00+00:00", f"Kuramoto erneut geprüft {minute}", f"Stand {minute}")
        for minute in range(unikin.JOURNAL_WINDOW)
    ]
    state = unikin.load_memory(config)
    state["journal"] = [KURAMOTO, *recent]
    state["next_focus"] = "Kuramoto-Phasendiagramm erweitern"
    unikin.save_memory(config, state)
    unikin.archive_journal(config, state["journal"])

    unikin.run_cycle(config, fake_model, state, [])

    (request,) = fake_model.requests
    assert "related_memory" in request["input"][0]["content"]
    # Was recent_journal ohnehin zeigt, kommt nicht noch einmal.
    assert [hit["ts"] for hit in fake_model.user_payload()["related_memory"]] == [KURAMOTO["ts"]]


def test_related_memory_can_be_switched_off(config, fake_model):
    config = dataclasses.replace(config, related_memory_hits=0)
    unikin.archive_journal(config, [KURAMOTO])
    state = unikin.load_memory(config)
    state["next_focus"] = "Kuramoto"

    unikin.run_cycle(config, fake_model, state, [])

    (request,) = fake_model.requests
    assert "related_memory" not in request["input"][0]["content"]
    assert "related_memory" not in fake_model.user_payload()


def test_repeated_thoughts_count_as_one_hit(config):
    same = "Loop {n}. Die Inbox ist leer, die Sandbox mit code.py ist kartiert, ich warte."
    unikin.archive_journal(
        config,
        [journal_entry(f"2026-10-21T10:0{n}:00+00:00", same.format(n=n), "Sandbox kartiert") for n in range(4)]
        + [journal_entry("2026-10-21T11:00:00+00:00", "Neue Frage zur Sandbox: Wie viel Speicher gibt es?")],
    )

    assert len(archive(config).search("Sandbox", 3)) == 3
    assert len(archive(config).search("Sandbox", 3, skip_repeats=True)) == 2


@pytest.mark.parametrize("query", ['os.listdir("code.py") AND NOT (', "listdir*", 'NEAR(os "listdir")'])
def test_search_text_is_never_fts_syntax(config, query):
    unikin.archive_journal(
        config,
        [
            journal_entry(
                "2026-10-21T10:00:00+00:00",
                "Arbeitsverzeichnis geprüft",
                # Im Journal abgeschnittenes JSON: Die Escapes werden trotzdem aufgelöst.
                tool_calls=[
                    {"name": "run_python", "arguments": '{"code": "import os\\nprint(os.listdir(', "output": ""}
                ],
            )
        ],
    )

    (hit,) = archive(config).search(query, 5)
    assert "print(os.listdir(" in hit["tool_calls"]


def test_search_finds_word_beginnings_and_ignores_diacritics(config):
    entry = journal_entry("2026-10-21T10:00:00+00:00", "Die Ergebnisse der Prüfung liegen vor.")
    unikin.archive_journal(config, [entry])

    assert len(archive(config).search("Ergebnis", 5)) == 1
    assert len(archive(config).search("Prufung", 5)) == 1
    assert archive(config).search("***", 5) == []
    assert fts_query("Die os.listdir() Ergebnisse, die") == '"die" OR "os" OR "listdir"* OR "ergebnisse"*'


def test_import_of_saved_states_is_idempotent(config, tmp_path):
    backup = tmp_path / "unikin_state.backup.json"
    backup.write_text(json.dumps({"journal": [KURAMOTO, {"thought": "ohne ts"}, "kein Objekt"]}), encoding="utf-8")

    assert import_state_files(archive(config), [backup]) == 1
    assert import_state_files(archive(config), [backup]) == 0
    assert archive(config).count() == 1


def test_config_related_memory_hits(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RELATED_MEMORY_HITS", raising=False)
    assert unikin.load_config().related_memory_hits == 3

    monkeypatch.setenv("RELATED_MEMORY_HITS", "0")
    assert unikin.load_config().related_memory_hits == 0

    monkeypatch.setenv("RELATED_MEMORY_HITS", "50")
    assert unikin.load_config().related_memory_hits == unikin.MAX_MEMORY_HITS
