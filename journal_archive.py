#!/usr/bin/env python3
"""Langzeitgedächtnis für Unikin: jeder Journal-Eintrag in SQLite, durchsuchbar per Volltext (FTS5).

Das Journal in unikin_state.json behält nur die letzten 40 Einträge, und der Prompt zeigt davon
sechs. Das Archiv behält alle. Es braucht weder einen eigenen Dienst noch ein Embedding-Modell,
also auch keine zweite Anfrage an LM Studio.

Nur der Loop schreibt ins Archiv; der MCP-Server öffnet es schreibgeschützt. Einträge werden nie
geändert, ein zweites Archivieren desselben Eintrags (gleicher ts) bleibt wirkungslos.

    python journal_archive.py import memory/unikin_state.backup-*.json   # alte Journale nachtragen
    python journal_archive.py import-log logs/unikin.log*                # Zyklen aus dem Log nachtragen
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone, tzinfo
from difflib import SequenceMatcher
from itertools import chain
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

ARCHIVE_FILENAME = "unikin_archive.sqlite3"
SCHEMA_VERSION = 2
BUSY_TIMEOUT_SECONDS = 10.0
JOURNAL = "journal"
# Aus logs/unikin.log nachgetragen: ohne state_update, Inbox und Argumente der Werkzeuge,
# deren Ausgabe auf 200 Zeichen gekürzt.
LOG = "log"
# Den ts eines Journal-Eintrags setzt run_cycle kurz vor der Logzeile „Loop N“, die nur ganze
# Sekunden zeigt. Dazwischen liegen Sekundenbruchteile, bei blockiertem Archiv bis zu
# BUSY_TIMEOUT_SECONDS. Zwei Zyklen in diesem Fenster unterscheiden sich an loop oder thought.
SAME_CYCLE_SECONDS = 15
# Unikins eigenes Suchwerkzeug. Seine Ausgabe kommt nicht ins Archiv: Sie besteht aus alten
# Einträgen, die dort schon stehen, und tauchte sonst bei jeder späteren Suche erneut auf.
SEARCH_TOOL = "search_memory"
MAX_QUERY_TERMS = 16
# Präfixsuche erst ab dieser Wortlänge, sonst findet "die" auch "Dienst" und "dies".
MIN_PREFIX_CHARS = 4
SNIPPET_TOKENS = 32
# Wie similarity_score in unikin.py: Ab hier gilt ein Treffer als Wiederholung eines besseren.
# Unikin schreibt oft viele Zyklen lang fast denselben Gedanken.
REPEAT_SIMILARITY = 0.85
REPEAT_CANDIDATES_FACTOR = 5

COLUMNS = ("ts", "loop", "urgency", "thought", "state_update", "next_focus", "inbox", "reply", "tool_calls")

SCHEMA = f"""
BEGIN;
CREATE TABLE IF NOT EXISTS entries (
    id INTEGER PRIMARY KEY,
    ts TEXT NOT NULL UNIQUE,
    loop INTEGER,
    urgency TEXT NOT NULL DEFAULT '',
    thought TEXT NOT NULL DEFAULT '',
    state_update TEXT NOT NULL DEFAULT '',
    next_focus TEXT NOT NULL DEFAULT '',
    inbox TEXT NOT NULL DEFAULT '',
    reply TEXT NOT NULL DEFAULT '',
    tool_calls TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT '{JOURNAL}'
);
CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts USING fts5(
    thought, state_update, next_focus, inbox, reply, tool_calls,
    content='entries', content_rowid='id', tokenize='unicode61 remove_diacritics 2'
);
-- Hält den Index synchron, auch wenn jemand von Hand Einträge löscht oder ändert.
CREATE TRIGGER IF NOT EXISTS entries_ai AFTER INSERT ON entries BEGIN
    INSERT INTO entries_fts(rowid, thought, state_update, next_focus, inbox, reply, tool_calls)
    VALUES (new.id, new.thought, new.state_update, new.next_focus, new.inbox, new.reply, new.tool_calls);
END;
CREATE TRIGGER IF NOT EXISTS entries_ad AFTER DELETE ON entries BEGIN
    INSERT INTO entries_fts(entries_fts, rowid, thought, state_update, next_focus, inbox, reply, tool_calls)
    VALUES ('delete', old.id, old.thought, old.state_update, old.next_focus, old.inbox, old.reply, old.tool_calls);
END;
CREATE TRIGGER IF NOT EXISTS entries_au AFTER UPDATE ON entries BEGIN
    INSERT INTO entries_fts(entries_fts, rowid, thought, state_update, next_focus, inbox, reply, tool_calls)
    VALUES ('delete', old.id, old.thought, old.state_update, old.next_focus, old.inbox, old.reply, old.tool_calls);
    INSERT INTO entries_fts(rowid, thought, state_update, next_focus, inbox, reply, tool_calls)
    VALUES (new.id, new.thought, new.state_update, new.next_focus, new.inbox, new.reply, new.tool_calls);
END;
PRAGMA user_version = {SCHEMA_VERSION};
COMMIT;
"""

# Archive der Version 1 kennen nur Einträge aus dem Journal.
MIGRATION_FROM_V1 = f"""
BEGIN;
ALTER TABLE entries ADD COLUMN source TEXT NOT NULL DEFAULT '{JOURNAL}';
PRAGMA user_version = {SCHEMA_VERSION};
COMMIT;
"""

INSERT = (
    f"INSERT OR IGNORE INTO entries ({', '.join(COLUMNS)}, source) "
    f"VALUES ({', '.join(':' + column for column in COLUMNS)}, :source)"
)

# Derselbe Zyklus: höchstens SAME_CYCLE_SECONDS auseinander und dieselbe Zyklusnummer, oder, wo
# eine fehlt (Journal-Einträge von vor dem Archiv), derselbe Gedanke. Alle ts sind ISO-Zeiten in
# UTC, ihr Textvergleich ist also ein Zeitvergleich und nutzt den Index auf ts.
SAME_CYCLE = (
    "ts BETWEEN :earliest AND :latest "
    "AND (loop = :loop OR ((loop IS NULL OR :loop IS NULL) AND thought = :thought))"
)
FIND_CYCLE = f"SELECT 1 FROM entries WHERE {SAME_CYCLE} LIMIT 1"
DELETE_LOG_COPY = f"DELETE FROM entries WHERE source = '{LOG}' AND {SAME_CYCLE}"

WORD = re.compile(r"\w+")
JSON_ESCAPE = re.compile(r"\\(.)")
JSON_ESCAPES = {"n": "\n", "t": "\t", "r": ""}

# Format von setup_logging in unikin.py. Zeilen ohne dieses Präfix setzen die Nachricht davor
# fort, etwa eine mehrzeilige Antwort in „Reply:“.
LOG_RECORD = re.compile(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) \| [A-Z]+ \| (.*)")
LOG_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
# Aus run_cycle; „| inbox=… | tools=…“ steht erst seit Loop 190 dahinter.
LOG_LOOP = re.compile(r"Loop (\d+) \| urgency=(\S*)")
# Aus execute_tool_call; davor, weil die Werkzeuge vor der Loop-Zeile ihres Zyklus laufen.
LOG_TOOL = re.compile(r"Werkzeug (\S+?)( \(Fehler\))?: ?(.*)", re.DOTALL)
LOG_FIELDS = {"Thought: ": "thought", "Next focus: ": "next_focus", "Reply: ": "reply"}
LOG_START = "Starte "


class JournalArchive:
    """Das Archiv in einer SQLite-Datei; jede Operation öffnet eine eigene kurze Verbindung."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @contextmanager
    def _connect(self, readonly: bool = False) -> Iterator[sqlite3.Connection]:
        if readonly:
            # mode=ro legt keine Datei an und schreibt garantiert nichts.
            conn = sqlite3.connect(f"{self.path.resolve().as_uri()}?mode=ro", uri=True, timeout=BUSY_TIMEOUT_SECONDS)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.path, timeout=BUSY_TIMEOUT_SECONDS)
        try:
            conn.row_factory = sqlite3.Row
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if not readonly and version < SCHEMA_VERSION:
                conn.executescript(MIGRATION_FROM_V1 if version == 1 else SCHEMA)
            yield conn
        finally:
            conn.close()

    def add(self, entries: Iterable[Any]) -> int:
        """Archiviert Journal-Einträge in einer Transaktion; liefert die Zahl der neuen.

        Ein Zyklus, der schon aus dem Log nachgetragen ist, weicht seinem Journal-Eintrag: Der hat
        zusätzlich state_update, die Inbox und die Argumente der Werkzeuge.
        """
        rows = [archive_row(entry) for entry in entries if isinstance(entry, dict) and entry.get("ts")]
        if not rows:
            return 0
        with self._connect() as conn, conn:
            conn.executemany(DELETE_LOG_COPY, [cycle for row in rows if (cycle := same_cycle(row))])
            return conn.executemany(INSERT, rows).rowcount

    def add_from_log(self, entries: Iterable[dict[str, Any]]) -> int:
        """Archiviert Zyklen aus dem Log, die noch nicht im Archiv stehen; liefert die Zahl der neuen."""
        rows = [archive_row(entry, source=LOG) for entry in entries]
        with self._connect() as conn, conn:
            # Sperrt Schreiber bis zum Commit: Sonst könnte der Loop zwischen Prüfen und Einfügen
            # denselben Zyklus aus dem Journal archivieren.
            conn.execute("BEGIN IMMEDIATE")
            rows = [row for row in rows if not conn.execute(FIND_CYCLE, same_cycle(row)).fetchone()]
            return conn.executemany(INSERT, rows).rowcount if rows else 0

    def count(self, readonly: bool = False) -> int:
        with self._connect(readonly) as conn:
            return conn.execute("SELECT count(*) FROM entries").fetchone()[0]

    def search(
        self,
        query: str,
        limit: int,
        exclude_ts: Iterable[str] = (),
        skip_repeats: bool = False,
        readonly: bool = False,
    ) -> list[dict[str, Any]]:
        """Die besten Treffer (bm25) mit allen Spalten und der passenden Textstelle in match.

        `exclude_ts` blendet Einträge aus, etwa die, die der Prompt ohnehin zeigt. Mit
        `skip_repeats` fallen Treffer weg, die einem besseren fast gleichen.
        """
        match = fts_query(query)
        if not match or limit < 1:
            return []
        exclude = [str(ts) for ts in exclude_ts]
        exclusion = f"AND entries.ts NOT IN ({', '.join('?' * len(exclude))})" if exclude else ""
        sql = (
            f"SELECT {', '.join('entries.' + column for column in COLUMNS)}, "
            f"snippet(entries_fts, -1, '', '', '…', {SNIPPET_TOKENS}) AS match "
            "FROM entries_fts JOIN entries ON entries.id = entries_fts.rowid "
            f"WHERE entries_fts MATCH ? {exclusion} ORDER BY entries_fts.rank LIMIT ?"
        )
        fetch = limit * REPEAT_CANDIDATES_FACTOR if skip_repeats else limit
        with self._connect(readonly) as conn:
            rows = [dict(row) for row in conn.execute(sql, [match, *exclude, fetch])]
        return without_repeats(rows, limit) if skip_repeats else rows


def fts_query(text: str) -> str:
    """Macht aus freiem Suchtext eine FTS5-Abfrage, die nie an der Syntax scheitert.

    Jedes Wort steht einzeln in Anführungszeichen, so wirken AND, NOT, * oder Klammern nicht
    als Operatoren. Ein * dahinter sucht nach Wortanfängen, damit "Ergebnis" auch "Ergebnisse"
    findet. Die Wörter sind mit OR verknüpft, auch die Teile von "Kuramoto-Phasendiagramm" oder
    "os.listdir"; bm25 sortiert Einträge mit mehr und selteneren Treffern nach vorn.
    """
    words = list(dict.fromkeys(WORD.findall(str(text or "").lower())))[:MAX_QUERY_TERMS]
    return " OR ".join(f'"{word}"' + ("*" if len(word) >= MIN_PREFIX_CHARS else "") for word in words)


def without_repeats(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for row in rows:
        text = f"{row['thought']} {row['state_update']}"
        if all(
            SequenceMatcher(None, text, f"{other['thought']} {other['state_update']}").ratio() < REPEAT_SIMILARITY
            for other in kept
        ):
            kept.append(row)
            if len(kept) >= limit:
                break
    return kept


def same_cycle(row: dict[str, Any]) -> dict[str, Any] | None:
    """Parameter für SAME_CYCLE; None, wenn ts keine Zeit mit Zeitzone ist."""
    try:
        ts = datetime.fromisoformat(row["ts"])
    except ValueError:
        return None
    if ts.tzinfo is None:
        return None
    window = timedelta(seconds=SAME_CYCLE_SECONDS)
    return {
        "earliest": (ts - window).astimezone(timezone.utc).isoformat(),
        "latest": (ts + window).astimezone(timezone.utc).isoformat(),
        "loop": row["loop"],
        "thought": row["thought"],
    }


def archive_row(entry: dict[str, Any], source: str = JOURNAL) -> dict[str, Any]:
    loop = entry.get("loop")
    return {
        "source": source,
        "ts": str(entry["ts"]),
        "loop": loop if isinstance(loop, int) and not isinstance(loop, bool) else None,
        "urgency": text_of(entry.get("urgency")),
        "thought": text_of(entry.get("thought")),
        "state_update": text_of(entry.get("state_update")),
        "next_focus": text_of(entry.get("next_focus")),
        "inbox": "\n".join(
            f"{text_of(message.get('sender'))}: {text_of(message.get('text'))}"
            for message in as_list(entry.get("inbox"))
            if isinstance(message, dict)
        ),
        "reply": text_of(entry.get("reply")),
        "tool_calls": "\n\n".join(
            tool_call_text(call) for call in as_list(entry.get("tool_calls")) if isinstance(call, dict)
        ),
    }


def tool_call_text(call: dict[str, Any]) -> str:
    name = text_of(call.get("name")) or "?"
    arguments = readable_arguments(text_of(call.get("arguments")))
    # Das Log nennt keine Argumente.
    call_text = f"{name}: {arguments}" if arguments else name
    if name == SEARCH_TOOL:
        return call_text
    return f"{call_text}\n→ {text_of(call.get('output'))}"


def readable_arguments(raw: str) -> str:
    """Macht aus den JSON-Argumenten lesbaren Text, z. B. den Code von run_python.

    Im Journal sind die Argumente oft abgeschnitten und damit kein gültiges JSON mehr; dann
    werden nur die Escapes aufgelöst. Sonst fände die Suche "print" nicht in "\\nprint".
    """
    try:
        parsed = json.loads(raw)
    except ValueError:
        return JSON_ESCAPE.sub(lambda match: JSON_ESCAPES.get(match.group(1), match.group(1)), raw)
    if isinstance(parsed, dict):
        return "\n".join(str(value) for value in parsed.values())
    return str(parsed)


def text_of(value: Any) -> str:
    return "" if value is None else str(value).strip()


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def import_state_files(archive: JournalArchive, paths: Iterable[Path]) -> int:
    """Trägt die Journale gespeicherter Zustände (auch Backups) ins Archiv ein; liefert die Zahl der neuen."""
    added = 0
    for path in paths:
        state = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(state, dict):
            raise ValueError(f"{path}: kein JSON-Objekt")
        added += archive.add(as_list(state.get("journal")))
    return added


def log_records(lines: Iterable[str]) -> Iterator[tuple[str, str]]:
    """(Zeitstempel, Nachricht) je Logeintrag; Folgezeilen gehören zur Nachricht davor."""
    timestamp = ""
    message: list[str] = []
    for line in lines:
        line = line.rstrip("\r")
        if match := LOG_RECORD.match(line):
            if timestamp:
                yield timestamp, "\n".join(message).rstrip("\n")
            timestamp, message = match[1], [match[2]]
        elif timestamp:
            message.append(line)
    if timestamp:
        yield timestamp, "\n".join(message).rstrip("\n")


def parse_log(lines: Iterable[str], log_timezone: tzinfo = timezone.utc) -> list[dict[str, Any]]:
    """Baut aus dem Log des Loops Einträge wie im Journal, einen pro Zeile „Loop N | …“.

    Die Werkzeugaufrufe stehen im Log vor der Loop-Zeile ihres Zyklus, Thought, Next focus und
    Reply dahinter. `log_timezone` ist die Zeitzone, in der der Loop lief (im Container UTC);
    ts wird wie im Journal eine ISO-Zeit in UTC.
    """
    entries: list[dict[str, Any]] = []
    entry: dict[str, Any] | None = None
    tool_calls: list[dict[str, Any]] = []
    for timestamp, message in log_records(lines):
        if loop := LOG_LOOP.match(message):
            ts = datetime.strptime(timestamp, LOG_TIME_FORMAT).replace(tzinfo=log_timezone)
            entry = {
                "ts": ts.astimezone(timezone.utc).isoformat(),
                "loop": int(loop[1]),
                "urgency": loop[2],
                "tool_calls": tool_calls,
            }
            entries.append(entry)
            tool_calls = []
        elif tool := LOG_TOOL.match(message):
            tool_calls.append({"name": tool[1], "arguments": "", "output": tool[3], "is_error": bool(tool[2])})
        elif message.startswith(LOG_START):
            # Werkzeugaufrufe ohne Loop-Zeile danach gehörten zu einem abgebrochenen Zyklus.
            entry, tool_calls = None, []
        elif entry is not None:
            for prefix, field in LOG_FIELDS.items():
                if message.startswith(prefix) and field not in entry:
                    entry[field] = message[len(prefix) :]
                    break
    return entries


def rotation_index(path: Path) -> int:
    """unikin.log.3 → 3; die laufende Datei unikin.log → 0."""
    suffix = path.suffix[1:]
    return int(suffix) if suffix.isdigit() else 0


def import_log_files(
    archive: JournalArchive, paths: Iterable[Path], log_timezone: tzinfo = timezone.utc
) -> tuple[int, int]:
    """Trägt die Zyklen aus dem Log ins Archiv ein; liefert (Zyklen im Log, davon neu archiviert).

    Die Dateien werden von der ältesten Rotation an als ein Log gelesen, denn ein Zyklus kann
    über die Grenze zweier Dateien reichen.
    """
    files = sorted(paths, key=rotation_index, reverse=True)
    lines = chain.from_iterable(path.read_text(encoding="utf-8", errors="replace").split("\n") for path in files)
    entries = parse_log(lines, log_timezone)
    return len(entries), archive.add_from_log(entries)


def timezone_name(name: str) -> tzinfo:
    if name.upper() == "UTC":
        return timezone.utc
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"unbekannte Zeitzone: {name}") from exc


def main() -> None:
    default_archive = Path(os.getenv("MEMORY_FILE", "memory/unikin_state.json")).parent / ARCHIVE_FILENAME
    parser = argparse.ArgumentParser(description="Unikins Langzeitgedächtnis (Journal-Archiv)")
    parser.add_argument(
        "--archive", type=Path, default=default_archive, help=f"Archivdatei (Standard: {default_archive})"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    import_parser = subparsers.add_parser("import", help="Journale aus gespeicherten Zuständen nachtragen")
    import_parser.add_argument("state_files", nargs="+", type=Path, help="z. B. memory/unikin_state.backup-*.json")
    log_parser = subparsers.add_parser("import-log", help="Zyklen aus dem Log des Loops nachtragen")
    log_parser.add_argument("log_files", nargs="+", type=Path, help="z. B. logs/unikin.log*")
    log_parser.add_argument(
        "--log-timezone",
        type=timezone_name,
        default=timezone.utc,
        help="Zeitzone der Zeitstempel im Log (Standard: UTC wie im Container; nativ etwa Europe/Berlin)",
    )
    args = parser.parse_args()

    archive = JournalArchive(args.archive)
    try:
        if args.command == "import-log":
            cycles, added = import_log_files(archive, args.log_files, args.log_timezone)
            print(f"{cycles} Zyklen im Log, {cycles - added} davon standen schon im Archiv.")
        else:
            added = import_state_files(archive, args.state_files)
    except (OSError, ValueError, sqlite3.Error) as exc:
        sys.exit(f"Import fehlgeschlagen: {exc}")
    print(f"{added} neue Einträge archiviert, insgesamt {archive.count()} in {args.archive}.")


if __name__ == "__main__":
    main()
