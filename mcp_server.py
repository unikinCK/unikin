#!/usr/bin/env python3
"""MCP-Server für Unikin, gebaut auf dem offiziellen MCP-Python-SDK.

Standard ist stdio; mit --http läuft er dauerhaft über Streamable HTTP (z. B. für
LibreChat), geschützt per OAuth 2.1 mit eigenem Login (siehe oauth.py).

Der Server ist eine Schnittstelle zum laufenden Loop: Er liest dessen Gedächtnis und
legt Nachrichten in die Inbox, schreibt aber nie selbst in unikin_state.json. So bleibt
der Loop der einzige Schreiber, und parallele Aufrufe überschreiben keinen Zustand.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sqlite3
import sys
import time
from logging.handlers import RotatingFileHandler
from typing import Annotated, Any
from urllib.parse import urlparse

import anyio
import uvicorn
from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions, RevocationOptions
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from pydantic import Field
from starlette.applications import Starlette

from journal_archive import JournalArchive
from oauth import SCOPE, LoginPage, OAuthStore, UnikinOAuthProvider
from passwords import parse_users
from unikin import (
    LOG_BACKUP_COUNT,
    LOG_MAX_BYTES,
    MAX_MESSAGE_CHARS,
    MAX_SENDER_CHARS,
    AgentConfig,
    archive_path,
    load_config,
    post_message,
)

SERVER_NAME = "unikin"
SERVER_VERSION = "0.2.0"
DEFAULT_SENDER = "Mensch"
JOURNAL_LIMIT = 40
SEARCH_LIMIT = 20
STEP_POLL_SECONDS = 1.0
STEP_TIMEOUT_SECONDS = 240
STEP_MAX_TIMEOUT_SECONDS = 600
HTTP_PORT = 8020
OAUTH_DB_NAME = "oauth.sqlite3"

INSTRUCTIONS = (
    "Schnittstelle zum laufenden Unikin-Agenten, einer autonomen Schleife auf einem lokalen Modell. "
    "get_state und get_journal lesen sein Gedächtnis, search_journal durchsucht alle archivierten "
    "Journal-Einträge, auch die, die nicht mehr im Journal stehen. send_message legt eine Nachricht in seine Inbox "
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


def search_archive(config: AgentConfig, query: str, limit: int) -> list[dict[str, Any]]:
    """Sucht schreibgeschützt im Archiv des Loops; anlegen und füllen ist allein seine Sache."""
    path = archive_path(config)
    if not path.exists():
        raise ToolError("Noch kein Langzeitgedächtnis: Der Loop hat seit dem Update noch keinen Zyklus abgeschlossen.")
    try:
        return JournalArchive(path).search(query, limit, readonly=True)
    except sqlite3.Error as exc:
        raise ToolError(f"Langzeitgedächtnis nicht lesbar: {exc}") from exc


def find_cycle(state: dict[str, Any], message_id: str) -> dict[str, Any] | None:
    for entry in reversed(state.get("journal", [])):
        if message_id in entry.get("inbox_ids", ()):
            return entry
    return None


def create_server(
    config: AgentConfig,
    oauth_provider: UnikinOAuthProvider | None = None,
    users: dict[str, str] | None = None,
) -> MCPServer:
    auth = None
    if oauth_provider is not None:
        auth = AuthSettings(
            issuer_url=oauth_provider.public_url,
            resource_server_url=oauth_provider.resource_url,
            validate_token_resource=True,
            required_scopes=[SCOPE],
            client_registration_options=ClientRegistrationOptions(
                enabled=True, valid_scopes=[SCOPE], default_scopes=[SCOPE]
            ),
            revocation_options=RevocationOptions(enabled=True),
        )
    server = MCPServer(
        name=SERVER_NAME,
        title="Unikin",
        version=SERVER_VERSION,
        instructions=INSTRUCTIONS,
        auth_server_provider=oauth_provider,
        auth=auth,
    )
    if oauth_provider is not None:
        server.custom_route("/login", methods=["GET", "POST"])(LoginPage(oauth_provider, users or {}).handle)

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
        Das Journal behält nur die letzten 40 Einträge; ältere findet search_journal.
        """
        return read_state(config).get("journal", [])[-limit:]

    @server.tool(title="Langzeitgedächtnis durchsuchen", annotations=READ_ONLY)
    def search_journal(
        query: Annotated[str, Field(min_length=1, max_length=500, description="Suchbegriffe, beliebiger Text.")],
        limit: Annotated[int, Field(ge=1, le=SEARCH_LIMIT, description="Höchstzahl der Treffer.")] = 10,
    ) -> list[dict[str, Any]]:
        """Volltextsuche in allen archivierten Journal-Einträgen, bester Treffer zuerst.

        Jeder Treffer enthält ts, loop (bei Einträgen aus der Zeit vor dem Archiv leer), thought,
        state_update, next_focus, urgency, inbox, reply, tool_calls (gekürzt) und match, die
        passende Textstelle. Jedes Wort sucht auch nach Wortanfängen, ein Treffer braucht nicht alle.
        """
        if not query.strip():
            raise ToolError("query darf nicht leer sein.")
        return search_archive(config, query, limit)

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


def split_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def create_http_app(
    config: AgentConfig,
    public_url: str,
    users: dict[str, str],
    allowed_redirect_hosts: set[str],
    allowed_hosts: list[str],
) -> Starlette:
    """MCP über Streamable HTTP mit OAuth: /mcp, Discovery, /register, /authorize, /token, /revoke, /login."""
    store = OAuthStore(config.memory_file.parent / OAUTH_DB_NAME)
    provider = UnikinOAuthProvider(store, public_url, allowed_redirect_hosts)
    public = urlparse(public_url)
    security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=allowed_hosts,
        allowed_origins=[f"{public.scheme}://{public.netloc}"],
    )
    return create_server(config, provider, users).streamable_http_app(transport_security=security)


def _handle_sigterm(_signum: int, _frame: Any) -> None:
    # Als PID 1 im Container würde SIGTERM sonst ignoriert und `docker stop`
    # erst nach der Karenzzeit hart beenden. SystemExit genügt nicht: Der SDK-Thread,
    # der stdin liest, blockiert dann das Herunterfahren. Zu sichern gibt es nichts,
    # Inbox-Dateien entstehen atomar, und die Logdatei wird pro Zeile geschrieben.
    os._exit(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="MCP-Server für Unikin")
    parser.add_argument("--http", action="store_true", help="Streamable HTTP mit OAuth statt stdio")
    parser.add_argument("--host", default="0.0.0.0", help="Adresse für --http")
    parser.add_argument("--port", type=int, default=HTTP_PORT, help="Port für --http")
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _handle_sigterm)
    config = load_config()

    if not args.http:
        server = create_server(config)
        # Erst nach MCPServer(): dessen eigene Logging-Konfiguration wird so ersetzt, nicht verdoppelt.
        setup_logging(config)
        logging.info("MCP-Server gestartet: %s %s (stdio)", SERVER_NAME, SERVER_VERSION)
        server.run("stdio")
        return

    public_url = os.getenv("UNIKIN_MCP_PUBLIC_URL", "").strip().rstrip("/")
    if not public_url:
        sys.exit("UNIKIN_MCP_PUBLIC_URL fehlt (z. B. https://macstudio.tail717550.ts.net:8020); ohne sie startet --http nicht.")
    try:
        users = parse_users(os.getenv("UNIKIN_OAUTH_USERS", ""))
    except ValueError as exc:
        sys.exit(str(exc))
    if not users:
        sys.exit("UNIKIN_OAUTH_USERS fehlt: Ohne Konto kann sich niemand anmelden. Hash mit `python3 passwords.py` erzeugen.")

    public_host = urlparse(public_url).hostname or ""
    allowed_redirect_hosts = set(split_list(os.getenv("UNIKIN_OAUTH_ALLOWED_REDIRECT_HOSTS", public_host)))
    # Host-Header, die /mcp annimmt (Schutz gegen DNS-Rebinding): die öffentliche Adresse und lokal.
    allowed_hosts = split_list(
        os.getenv("UNIKIN_MCP_ALLOWED_HOSTS", f"{public_host}:*,{public_host},localhost:*,127.0.0.1:*")
    )
    try:
        app = create_http_app(config, public_url, users, allowed_redirect_hosts, allowed_hosts)
    except ValueError as exc:
        sys.exit(f"Ungültige UNIKIN_MCP_PUBLIC_URL: {exc}")
    setup_logging(config)
    logging.info(
        "MCP-Server gestartet: %s %s (HTTP auf %s:%s, öffentlich %s, Konten: %s)",
        SERVER_NAME,
        SERVER_VERSION,
        args.host,
        args.port,
        public_url,
        ", ".join(users),
    )
    # log_config=None: uvicorn schreibt über unsere Logging-Handler statt eigene anzulegen.
    # Offene Streams (wartendes step, SSE von LibreChat) sollen `docker stop` nicht aufhalten.
    uvicorn.run(app, host=args.host, port=args.port, log_config=None, timeout_graceful_shutdown=5)


if __name__ == "__main__":
    main()
