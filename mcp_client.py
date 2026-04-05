#!/usr/bin/env python3
"""Simple MCP client for the local Unikin MCP server."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Any


def rpc(proc: subprocess.Popen[str], request_id: int, method: str, params: dict[str, Any]) -> dict[str, Any]:
    req = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
    assert proc.stdin is not None
    assert proc.stdout is not None
    proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
    proc.stdin.flush()

    line = proc.stdout.readline()
    if not line:
        raise RuntimeError("Keine Antwort vom MCP-Server erhalten.")

    response = json.loads(line)
    if "error" in response:
        err = response["error"]
        raise RuntimeError(f"MCP-Fehler {err.get('code')}: {err.get('message')}")

    return response.get("result", {})


def print_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="MCP-Client für Unikin")
    parser.add_argument(
        "--server-cmd",
        default=f"{sys.executable} mcp_server.py",
        help="Befehl zum Starten des MCP-Servers",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("tools", help="Verfügbare Tools anzeigen")
    subparsers.add_parser("state", help="Aktuellen Unikin-State abrufen")

    step_parser = subparsers.add_parser("step", help="Einen Unikin-Schritt ausführen")
    step_parser.add_argument("--note", default="", help="Optionaler Journal-Hinweis")

    args = parser.parse_args()

    proc = subprocess.Popen(
        args.server_cmd,
        shell=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        init_result = rpc(proc, 1, "initialize", {"protocolVersion": "2024-11-05", "capabilities": {}})
        if not init_result:
            raise RuntimeError("Server-Initialisierung fehlgeschlagen")

        if args.command == "tools":
            print_json(rpc(proc, 2, "tools/list", {}))
        elif args.command == "state":
            payload = rpc(
                proc,
                3,
                "tools/call",
                {"name": "unikin.get_state", "arguments": {}},
            )
            print_json(payload)
        elif args.command == "step":
            payload = rpc(
                proc,
                4,
                "tools/call",
                {"name": "unikin.step", "arguments": {"note": args.note}},
            )
            print_json(payload)

    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
