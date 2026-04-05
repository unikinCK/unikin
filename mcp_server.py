#!/usr/bin/env python3
"""Minimal MCP server for Unikin over stdio (JSON-RPC lines)."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from openai import OpenAI

from unikin import (
    build_prompt,
    call_model,
    clean_text,
    compute_interval,
    load_config,
    load_memory,
    save_memory,
    similarity_score,
)

SERVER_INFO = {"name": "unikin-mcp", "version": "0.1.0"}
PROTOCOL_VERSION = "2024-11-05"


class McpJsonLineServer:
    def __init__(self) -> None:
        self.config = load_config()
        self._setup_logging()
        self.client = OpenAI(
            base_url=self.config.openai_base_url,
            api_key=self.config.openai_api_key,
        )
        logging.info("MCP-Server gestartet: %s", SERVER_INFO["name"])


    def _setup_logging(self) -> None:
        self.config.log_file.parent.mkdir(parents=True, exist_ok=True)
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        root.handlers.clear()

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(formatter)

        file_handler = logging.FileHandler(self.config.log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)

        root.addHandler(stream_handler)
        root.addHandler(file_handler)

    def _tool_list(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "unikin.get_state",
                "description": "Lädt den aktuellen persistierten Unikin-Zustand.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
            {
                "name": "unikin.step",
                "description": "Führt einen autonomen Unikin-Reflexionsschritt aus und speichert das Ergebnis.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "note": {
                            "type": "string",
                            "description": "Optionaler Hinweis für den Journal-Eintrag.",
                        }
                    },
                    "additionalProperties": False,
                },
            },
        ]

    def _run_step(self, note: str | None = None) -> dict[str, Any]:
        state = load_memory(self.config)
        metrics = state.setdefault("metrics", {})
        metrics.setdefault("loop_count", 0)
        metrics.setdefault("error_count", 0)
        metrics.setdefault("consecutive_errors", 0)
        metrics.setdefault("consecutive_similar", 0)
        metrics.setdefault("last_response_seconds", None)
        metrics.setdefault("current_interval", self.config.default_interval)

        metrics["loop_count"] += 1
        system_prompt, user_prompt = build_prompt(self.config, state)
        result, response_seconds, parse_failed = call_model(
            client=self.client,
            model=self.config.openai_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        prev_thought = clean_text(state.get("last_thought", ""))
        current_thought = clean_text(result["thought"])
        sim = similarity_score(prev_thought, current_thought)

        if sim >= 0.85:
            metrics["consecutive_similar"] += 1
        else:
            metrics["consecutive_similar"] = 0

        if parse_failed:
            metrics["error_count"] += 1
            metrics["consecutive_errors"] += 1
        else:
            metrics["consecutive_errors"] = 0

        new_interval = compute_interval(
            config=self.config,
            current_interval=float(metrics.get("current_interval", self.config.default_interval)),
            urgency=result["urgency"],
            consecutive_errors=int(metrics.get("consecutive_errors", 0)),
            consecutive_similar=int(metrics.get("consecutive_similar", 0)),
            response_seconds=response_seconds,
        )

        state["last_thought"] = result["thought"]
        state["last_state"] = result["state_update"]
        state["next_focus"] = result["next_focus"]
        metrics["last_response_seconds"] = round(response_seconds, 3)
        metrics["current_interval"] = round(new_interval, 3)

        journal = state.setdefault("journal", [])
        journal.append(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "thought": result["thought"],
                "state_update": result["state_update"],
                "next_focus": result["next_focus"],
                "urgency": result["urgency"],
                "similarity": round(sim, 3),
                "response_seconds": round(response_seconds, 3),
                "parse_failed": parse_failed,
                "note": note or "",
            }
        )
        if len(journal) > 40:
            del journal[:-40]

        save_memory(self.config, state)

        return {
            "result": result,
            "similarity": round(sim, 3),
            "response_seconds": round(response_seconds, 3),
            "next_interval_seconds": round(new_interval, 3),
            "parse_failed": parse_failed,
            "loop_count": metrics["loop_count"],
        }

    def _result(self, request_id: Any, payload: Any) -> None:
        message = {"jsonrpc": "2.0", "id": request_id, "result": payload}
        sys.stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    def _error(self, request_id: Any, code: int, message: str) -> None:
        payload = {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}
        sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    def serve_forever(self) -> None:
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue

            try:
                req = json.loads(line)
            except json.JSONDecodeError:
                continue

            request_id = req.get("id")
            method = req.get("method")
            params = req.get("params", {})

            try:
                if method == "initialize":
                    self._result(
                        request_id,
                        {
                            "protocolVersion": PROTOCOL_VERSION,
                            "serverInfo": SERVER_INFO,
                            "capabilities": {"tools": {}},
                        },
                    )
                elif method == "ping":
                    self._result(request_id, {})
                elif method == "tools/list":
                    self._result(request_id, {"tools": self._tool_list()})
                elif method == "tools/call":
                    name = params.get("name")
                    arguments = params.get("arguments", {})

                    if name == "unikin.get_state":
                        state = load_memory(self.config)
                        self._result(request_id, {"content": [{"type": "json", "json": state}]})
                    elif name == "unikin.step":
                        payload = self._run_step(note=arguments.get("note"))
                        self._result(request_id, {"content": [{"type": "json", "json": payload}]})
                    else:
                        self._error(request_id, -32601, f"Unknown tool: {name}")
                else:
                    self._error(request_id, -32601, f"Unknown method: {method}")
            except Exception as exc:  # noqa: BLE001
                logging.exception("MCP request failed: %s", exc)
                self._error(request_id, -32000, str(exc))


def main() -> None:
    McpJsonLineServer().serve_forever()


if __name__ == "__main__":
    main()
