#!/usr/bin/env python3
"""Kommandozeilen-Client für den Unikin-MCP-Server (offizielles MCP-Python-SDK, stdio)."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any

import anyio
from mcp import Client, StdioServerParameters

DEFAULT_SERVER_CMD = f"{shlex.quote(sys.executable)} {shlex.quote(str(Path(__file__).with_name('mcp_server.py')))}"
# Puffer über timeout_seconds von step hinaus, damit der Client nicht vor dem Server aufgibt.
STEP_READ_MARGIN_SECONDS = 30


def print_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def tool_call(args: argparse.Namespace) -> tuple[str, dict[str, Any], float | None]:
    if args.command == "state":
        return "get_state", {}, None
    if args.command == "journal":
        return "get_journal", {"limit": args.limit}, None
    if args.command == "search":
        return "search_journal", {"query": args.query, "limit": args.limit}, None
    if args.command == "send":
        return "send_message", {"text": args.text, "sender": args.sender}, None
    return (
        "step",
        {"note": args.note, "sender": args.sender, "timeout_seconds": args.timeout},
        args.timeout + STEP_READ_MARGIN_SECONDS,
    )


async def run(args: argparse.Namespace) -> int:
    command, *command_args = shlex.split(args.server_cmd)
    # Volle Umgebung durchreichen: `docker compose` braucht u. a. den Docker-Kontext bzw. DOCKER_HOST.
    server = StdioServerParameters(command=command, args=command_args, env=dict(os.environ))

    async with Client(server) as client:
        if args.command == "tools":
            listing = await client.list_tools()
            print_json([tool.model_dump(mode="json", by_alias=True, exclude_none=True) for tool in listing.tools])
            return 0

        name, arguments, read_timeout = tool_call(args)
        result = await client.call_tool(name, arguments, read_timeout_seconds=read_timeout)
        if result.is_error:
            print("\n".join(block.text for block in result.content if block.type == "text"), file=sys.stderr)
            return 1
        if result.structured_content is not None:
            print_json(result.structured_content)
        else:
            print_json([block.model_dump(mode="json", by_alias=True, exclude_none=True) for block in result.content])
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="MCP-Client für Unikin")
    parser.add_argument(
        "--server-cmd",
        default=DEFAULT_SERVER_CMD,
        help="Befehl zum Starten des MCP-Servers (stdio)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("tools", help="Verfügbare Tools anzeigen")
    subparsers.add_parser("state", help="Aktuellen Unikin-Zustand abrufen")

    journal_parser = subparsers.add_parser("journal", help="Neueste Journal-Einträge abrufen")
    journal_parser.add_argument("--limit", type=int, default=5, help="Anzahl der Einträge")

    search_parser = subparsers.add_parser("search", help="Alle archivierten Journal-Einträge durchsuchen")
    search_parser.add_argument("query", help="Suchbegriffe")
    search_parser.add_argument("--limit", type=int, default=5, help="Höchstzahl der Treffer")

    send_parser = subparsers.add_parser("send", help="Nachricht in die Inbox legen (kehrt sofort zurück)")
    send_parser.add_argument("text", help="Nachricht an Unikin")
    send_parser.add_argument("--sender", default="Mensch", help="Absender")

    step_parser = subparsers.add_parser("step", help="Sofort einen Zyklus anstoßen und auf das Ergebnis warten")
    step_parser.add_argument("--note", default="", help="Nachricht, die Unikin in diesem Zyklus liest")
    step_parser.add_argument("--sender", default="Mensch", help="Absender")
    step_parser.add_argument("--timeout", type=int, default=240, help="Wartezeit in Sekunden")

    args = parser.parse_args()
    sys.exit(anyio.run(run, args))


if __name__ == "__main__":
    main()
