#!/usr/bin/env python3
"""MCP-Server für Unikin über stdio, gebaut auf dem offiziellen MCP-Python-SDK.

Der Server ist eine Schnittstelle zum laufenden Loop: Er liest dessen Gedächtnis und
legt Nachrichten in die Inbox, schreibt aber nie selbst in unikin_state.json. So bleibt
der Loop der einzige Schreiber, und parallele Aufrufe überschreiben keinen Zustand.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
from logging.handlers import RotatingFileHandler
from typing import Annotated, Any

import anyio
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from unikin import (
    LOG_BACKUP_COUNT,
    LOG_MAX_BYTES,
    MAX_MESSAGE_CHARS,
    MAX_SENDER_CHARS,
    AgentConfig,
    load_config,
    post_message,
)

SERVER_NAME = "unikin"
SERVER_VERSION = "0.2.0"
DEFAULT_SENDER = "Mensch"
JOURNAL_LIMIT = 40
STEP_POLL_SECONDS = 1.0
STEP_TIMEOUT_SECONDS = 240
STEP_MAX_TIMEOUT_SECONDS = 600

INSTRUCTIONS = (
    "Schnittstelle zum laufenden Unikin-Agenten, einer autonomen Schleife auf einem lokalen Modell. "
    "get_state und get_journal lesen sein Gedächtnis. send_message legt eine Nachricht in seine Inbox "
    "und kehrt sofort zurück; step stößt sofort einen Zyklus an und wartet auf dessen Ergebnis. "
    "Unikins Antwort auf Nachrichten steht im Journal-Eintrag im Feld reply."
)

READ_ONLY = ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False)
WRITES_INBOX = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)

Sender = Annotated[str, Field(max_length=MAX_SENDER_CHARS, description="Name oder Rolle des Absenders.")]


def setup_logging(config: AgentConfig) -> None:
    config.log_file.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # stdout gehört dem MCP-Protokoll. Auf stderr nur Warnungen, damit Clients wie
    # mcp_client.py die Ausgabe nicht mit Routinemeldungen füllen.
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(logging.WARNING)
    stream_handler.setFormatter(formatter)

    # Eigene Datei statt der rotierenden Logdatei des Loops: zwei Schreiber mit
    # eigener Rotation auf derselben Datei würden sich gegenseitig Zeilen verlieren.
    mcp_log = config.log_file.with_name(f"{config.log_file.stem}-mcp{config.log_file.suffix}")
    file_handler = RotatingFileHandler(mcp_log, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root.addHandler(stream_handler)
    root.addHandler(file_handler)


def read_state(config: AgentConfig) -> dict[str, Any]:
    """Liest den Zustand, ohne ihn anzulegen oder zu reparieren; das bleibt Sache des Loops."""
    try:
        state = json.loads(config.memory_file.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ToolError("Noch kein Zustand gespeichert: Der Loop hat noch keinen Zyklus abgeschlossen.") from exc
    except (OSError, ValueError) as exc:
        raise ToolError(f"Zustand nicht lesbar: {exc}") from exc
    if not isinstance(state, dict):
        raise ToolError("Zustand nicht lesbar: kein JSON-Objekt.")
    return state


def find_cycle(state: dict[str, Any], message_id: str) -> dict[str, Any] | None:
    for entry in reversed(state.get("journal", [])):
        if message_id in entry.get("inbox_ids", ()):
            return entry
    return None


def create_server(config: AgentConfig) -> MCPServer:
    server = MCPServer(name=SERVER_NAME, title="Unikin", version=SERVER_VERSION, instructions=INSTRUCTIONS)

    @server.tool(title="Zustand lesen", annotations=READ_ONLY)
    def get_state() -> dict[str, Any]:
        """Liefert Unikins gespeicherten Zustand: Identität, Mission, letzter Gedanke, nächster Fokus, Metriken und Journal."""
        return read_state(config)

    @server.tool(title="Journal lesen", annotations=READ_ONLY)
    def get_journal(
        limit: Annotated[int, Field(ge=1, le=JOURNAL_LIMIT, description="Anzahl der neuesten Einträge.")] = 10,
    ) -> list[dict[str, Any]]:
        """Liefert die neuesten Journal-Einträge, ältester zuerst.

        Einträge von Zyklen mit Nachrichten enthalten inbox (die Nachrichten) und reply (Unikins Antwort).
        """
        return read_state(config).get("journal", [])[-limit:]

    @server.tool(title="Nachricht senden", annotations=WRITES_INBOX)
    def send_message(
        text: Annotated[str, Field(min_length=1, max_length=MAX_MESSAGE_CHARS, description="Nachricht an Unikin.")],
        sender: Sender = DEFAULT_SENDER,
    ) -> dict[str, Any]:
        """Legt eine Nachricht in Unikins Inbox und kehrt sofort zurück.

        Der laufende Loop wacht davon auf und liest sie im nächsten Zyklus, meist binnen Sekunden;
        steckt er gerade in einem Modellaufruf, danach. Die Antwort erscheint im Journal (Feld reply).
        Wer direkt auf die Antwort warten will, nutzt step mit note.
        """
        if not text.strip():
            raise ToolError("text darf nicht leer sein.")
        message = post_message(config, text, sender)
        return {"message_id": message["id"], "status": "queued"}

    @server.tool(title="Zyklus anstoßen", annotations=WRITES_INBOX)
    async def step(
        note: Annotated[
            str,
            Field(max_length=MAX_MESSAGE_CHARS, description="Optionale Nachricht, die Unikin in diesem Zyklus liest."),
        ] = "",
        sender: Sender = DEFAULT_SENDER,
        timeout_seconds: Annotated[
            int, Field(ge=1, le=STEP_MAX_TIMEOUT_SECONDS, description="Wie lange auf das Ergebnis gewartet wird.")
        ] = STEP_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        """Stößt sofort einen Zyklus des laufenden Loops an und wartet auf dessen Journal-Eintrag.

        Mit note liest Unikin die Nachricht in genau diesem Zyklus und antwortet im Feld reply.
        Läuft der Loop nicht oder ist das Modell nicht erreichbar, kommt nach timeout_seconds
        status "pending" zurück; die Nachricht bleibt dann in der Inbox liegen.
        """
        message = post_message(config, note, sender)
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            await anyio.sleep(STEP_POLL_SECONDS)
            try:
                entry = find_cycle(read_state(config), message["id"])
            except ToolError:
                continue
            if entry is not None:
                return {"status": "done", "message_id": message["id"], "entry": entry}
        return {
            "status": "pending",
            "message_id": message["id"],
            "hint": "Der Loop hat den Zyklus noch nicht abgeschlossen (läuft er, ist LM Studio erreichbar?). "
            "Das Ergebnis erscheint später im Journal.",
        }

    return server


def _handle_sigterm(_signum: int, _frame: Any) -> None:
    # Als PID 1 im Container würde SIGTERM sonst ignoriert und `docker stop`
    # erst nach der Karenzzeit hart beenden. SystemExit genügt nicht: Der SDK-Thread,
    # der stdin liest, blockiert dann das Herunterfahren. Zu sichern gibt es nichts,
    # Inbox-Dateien entstehen atomar, und die Logdatei wird pro Zeile geschrieben.
    os._exit(0)


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_sigterm)
    config = load_config()
    server = create_server(config)
    # Erst nach MCPServer(): dessen eigene Logging-Konfiguration wird so ersetzt, nicht verdoppelt.
    setup_logging(config)
    logging.info("MCP-Server gestartet: %s %s", SERVER_NAME, SERVER_VERSION)
    server.run("stdio")


if __name__ == "__main__":
    main()
