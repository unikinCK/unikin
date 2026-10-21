#!/usr/bin/env python3
"""Unikin: an autonomous local OpenAI-compatible loop agent."""

from __future__ import annotations

import json
import logging
import os
import random
import re
import signal
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from functools import partial
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import anyio
from dotenv import load_dotenv
from mcp import Client as McpClient
from openai import APITimeoutError, OpenAI

from journal_archive import ARCHIVE_FILENAME, SEARCH_TOOL, JournalArchive

JOURNAL_WINDOW = 6
MAX_RETRIES = 3
STALE_TMP_SECONDS = 60.0
RETRY_BACKOFF_SECONDS = 2.0
LOG_MAX_BYTES = 10 * 1024 * 1024
LOG_BACKUP_COUNT = 5
INBOX_DIRNAME = "inbox"
MAX_INBOX_PER_CYCLE = 5
MAX_MESSAGE_CHARS = 4000
MAX_SENDER_CHARS = 80
MAX_REPLY_CHARS = 4000
RUNNER_TIMEOUT_SECONDS = 60.0
MAX_TOOL_OUTPUT_CHARS = 4000
# Im Journal kürzer: Die letzten Einträge gehen in jeden Prompt.
JOURNAL_TOOL_ARGUMENTS_CHARS = 600
JOURNAL_TOOL_OUTPUT_CHARS = 400
# Schaltet bei Qwen 3.6 in LM Studio das Denken ab (Jinja-Variable enable_thinking);
# "minimal" oder "low" wirken dort nicht, das Modell kennt nur an oder aus.
THINKING_OFF = {"effort": "none"}
# Treffer aus dem Langzeitgedächtnis (search_memory und related_memory).
DEFAULT_MEMORY_HITS = 3
MAX_MEMORY_HITS = 5
MEMORY_HIT_CHARS = 240


@dataclass
class AgentConfig:
    openai_base_url: str
    openai_api_key: str
    openai_model: str
    openai_timeout: float
    identity: str
    mission: str
    min_interval: float
    max_interval: float
    default_interval: float
    memory_file: Path
    log_file: Path
    # MCP-Server mit Werkzeugen (Python-Runner); leer = keine Werkzeuge.
    runner_url: str = ""
    max_tool_calls: int = 3
    # Höchstens so viele Ausgabe-Tokens (Denken plus Antwort) pro Anfrage; 0 = keine Grenze.
    max_output_tokens: int = 12000
    # So viele Archiv-Treffer zum next_focus stehen ohne Modellaufruf als related_memory im Prompt; 0 = aus.
    related_memory_hits: int = DEFAULT_MEMORY_HITS


class GracefulExit:
    """State holder for signal-based exits."""

    WAIT_SLICE_SECONDS = 0.5

    def __init__(self) -> None:
        self.should_stop = False
        self.signum: int | None = None

    def _handle_signal(self, signum: int, _frame: Any) -> None:
        # Nur Flags setzen und ungepuffert auf stderr schreiben: kein Logging
        # und keine Locks im Signal-Handler.
        self.signum = signum
        self.should_stop = True
        try:
            os.write(2, f"Signal {signum} empfangen, beende nach dem laufenden Schritt ...\n".encode())
        except OSError:
            pass

    def install(self) -> None:
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def wait(
        self,
        seconds: float,
        wake: Callable[[], bool] | None = None,
        min_seconds: float = 0.0,
    ) -> bool:
        """Wartet bis zu `seconds` in kurzen Scheiben; True, sobald ein Stop-Signal anliegt.

        Ein einzelnes time.sleep() würde nach einem Signal weiterschlafen (PEP 475);
        so kann z. B. `docker stop` den Agenten innerhalb der Karenzzeit sauber beenden.
        `wake` beendet die Wartezeit vorzeitig (frühestens nach `min_seconds`), etwa
        wenn eine neue Nachricht in der Inbox liegt.
        """
        start = time.monotonic()
        deadline = start + seconds
        while not self.should_stop:
            now = time.monotonic()
            remaining = deadline - now
            if remaining <= 0:
                return False
            if wake is not None and now - start >= min_seconds and wake():
                return False
            time.sleep(min(self.WAIT_SLICE_SECONDS, remaining))
        return True


def to_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def load_config() -> AgentConfig:
    load_dotenv()

    min_interval = to_float(os.getenv("MIN_INTERVAL_SECONDS"), 5.0)
    max_interval = to_float(os.getenv("MAX_INTERVAL_SECONDS"), 300.0)
    default_interval = to_float(os.getenv("DEFAULT_INTERVAL_SECONDS"), 30.0)

    if min_interval <= 0:
        min_interval = 5.0
    if max_interval < min_interval:
        max_interval = min_interval
    default_interval = max(min_interval, min(default_interval, max_interval))

    return AgentConfig(
        openai_base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1"),
        openai_api_key=os.getenv("OPENAI_API_KEY", "local-key"),
        openai_model=os.getenv("OPENAI_MODEL", "local-model"),
        openai_timeout=max(1.0, to_float(os.getenv("OPENAI_TIMEOUT_SECONDS"), 300.0)),
        identity=os.getenv("UNIKIN_IDENTITY", "Unikin"),
        mission=os.getenv(
            "UNIKIN_MISSION",
            "Kontinuierliche reflektierte Arbeit in kleinen Schritten mit robustem Selbstmonitoring.",
        ),
        min_interval=min_interval,
        max_interval=max_interval,
        default_interval=default_interval,
        memory_file=Path(os.getenv("MEMORY_FILE", "memory/unikin_state.json")),
        log_file=Path(os.getenv("LOG_FILE", "logs/unikin.log")),
        runner_url=os.getenv("RUNNER_MCP_URL", "").strip(),
        max_tool_calls=max(0, int(to_float(os.getenv("MAX_TOOL_CALLS_PER_CYCLE"), 3))),
        max_output_tokens=max(0, int(to_float(os.getenv("OPENAI_MAX_OUTPUT_TOKENS"), 12000))),
        related_memory_hits=max(
            0, min(MAX_MEMORY_HITS, int(to_float(os.getenv("RELATED_MEMORY_HITS"), DEFAULT_MEMORY_HITS)))
        ),
    )


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        log_path, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)


def initial_state(config: AgentConfig) -> dict[str, Any]:
    return {
        "identity": config.identity,
        "mission": config.mission,
        "last_state": "Initialzustand",
        "next_focus": "Orientierung und erster autonomer Schritt",
        "last_thought": "",
        "journal": [],
        "metrics": {
            "loop_count": 0,
            "error_count": 0,
            "consecutive_errors": 0,
            "consecutive_similar": 0,
            "last_response_seconds": None,
            "current_interval": config.default_interval,
        },
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def remove_stale_tmp_files(directory: Path, pattern: str) -> None:
    """Räumt Temp-Dateien weg, die ein harter Abbruch in write_json_atomic hinterlassen hat.

    Nur ältere Dateien: eine frische kann gerade vom MCP-Server geschrieben werden.
    """
    cutoff = time.time() - STALE_TMP_SECONDS
    for tmp_path in directory.glob(pattern):
        try:
            if tmp_path.stat().st_mtime < cutoff:
                tmp_path.unlink()
        except OSError:
            pass


def load_memory(config: AgentConfig) -> dict[str, Any]:
    config.memory_file.parent.mkdir(parents=True, exist_ok=True)
    remove_stale_tmp_files(config.memory_file.parent, f"{config.memory_file.name}.*.tmp")

    if not config.memory_file.exists():
        state = initial_state(config)
        save_memory(config, state)
        return state

    try:
        state = json.loads(config.memory_file.read_text(encoding="utf-8"))
        if not isinstance(state, dict):
            raise ValueError("Memory root muss ein Objekt sein")
        return state
    except Exception as exc:  # noqa: BLE001
        logging.exception("Memory-Datei konnte nicht geladen werden, reinitialisiere: %s", exc)
        state = initial_state(config)
        save_memory(config, state)
        return state


def write_json_atomic(path: Path, data: Any) -> None:
    # Atomar schreiben: ein harter Abbruch (SIGKILL, Stromausfall) darf nie eine
    # halbe Datei hinterlassen, die ein Leser sonst als defekt verwerfen würde.
    # Eindeutiger Temp-Name statt PID: in Containern ist die PID immer 1, und der
    # Loop und der MCP-Server können dasselbe Verzeichnis gleichzeitig beschreiben.
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_name, 0o644)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def save_memory(config: AgentConfig, state: dict[str, Any]) -> None:
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json_atomic(config.memory_file, state)


def inbox_dir(config: AgentConfig) -> Path:
    return config.memory_file.parent / INBOX_DIRNAME


def archive_path(config: AgentConfig) -> Path:
    return config.memory_file.parent / ARCHIVE_FILENAME


def archive_journal(config: AgentConfig, journal: list[dict[str, Any]]) -> None:
    """Überträgt das Journal ins Langzeitgedächtnis; ein Fehler wird nur protokolliert.

    Das Archiv übernimmt jeden Eintrag nur einmal (Schlüssel ts). Deshalb bekommt es nach jedem
    Zyklus das ganze Journal: Der neue Eintrag kommt dazu, und einer, dessen Archivierung vorher
    scheiterte, wird nachgeholt, solange er noch im Journal steht. Beim ersten Start mit Archiv
    ist das zugleich das Backfill der vorhandenen Einträge.
    """
    try:
        added = JournalArchive(archive_path(config)).add(journal)
    except Exception as exc:  # noqa: BLE001
        logging.warning("Journal konnte nicht archiviert werden, der Zyklus läuft weiter: %s", describe_error(exc))
        return
    if added > 1:
        logging.info("Langzeitgedächtnis: %s Journal-Einträge archiviert.", added)


def memory_hit(row: dict[str, Any]) -> dict[str, Any]:
    """Kurzer Archiv-Treffer fürs Modell: wann, was es daraus geschlossen hat und die passende Stelle."""
    hit: dict[str, Any] = {"ts": row["ts"]}
    if row["loop"] is not None:
        hit["loop"] = row["loop"]
    state_update = clean_text(row["state_update"])
    if len(state_update) > MEMORY_HIT_CHARS:
        state_update = state_update[:MEMORY_HIT_CHARS].rstrip() + "…"
    hit["state_update"] = state_update
    hit["match"] = clean_text(row["match"])
    return hit


def search_memory(config: AgentConfig, arguments: dict[str, Any]) -> str:
    """Das Werkzeug search_memory: Volltextsuche im Archiv, ohne Modellaufruf."""
    query = clean_text(arguments.get("query"))
    if not query:
        raise ValueError("query fehlt oder ist leer.")
    limit = int(to_float(str(arguments.get("limit") or DEFAULT_MEMORY_HITS), DEFAULT_MEMORY_HITS))
    limit = max(1, min(limit, MAX_MEMORY_HITS))
    hits = JournalArchive(archive_path(config)).search(query, limit, skip_repeats=True)
    if not hits:
        return f"Keine Treffer im Langzeitgedächtnis für: {query}"
    return json.dumps([memory_hit(hit) for hit in hits], ensure_ascii=False)


def find_related_memory(config: AgentConfig, state: dict[str, Any]) -> list[dict[str, Any]]:
    """Archiv-Treffer zum aktuellen next_focus für den Prompt; Qwen ruft search_memory selten von selbst auf.

    Einträge, die recent_journal ohnehin zeigt, fallen weg. Ein Fehler kostet nur die Treffer.
    """
    focus = clean_text(state.get("next_focus"))
    if config.related_memory_hits <= 0 or not focus:
        return []
    recent = [str(entry.get("ts")) for entry in state.get("journal", [])[-JOURNAL_WINDOW:] if isinstance(entry, dict)]
    try:
        hits = JournalArchive(archive_path(config)).search(
            focus, config.related_memory_hits, exclude_ts=recent, skip_repeats=True
        )
    except Exception as exc:  # noqa: BLE001
        logging.warning("Langzeitgedächtnis nicht durchsuchbar, related_memory bleibt leer: %s", describe_error(exc))
        return []
    return [memory_hit(hit) for hit in hits]


def post_message(config: AgentConfig, text: str, sender: str) -> dict[str, Any]:
    """Legt eine Nachricht als eigene Datei in die Inbox; leerer Text stößt nur einen Zyklus an.

    Eine Datei pro Nachricht statt eines Felds in unikin_state.json: So schreibt der
    Loop als Einziger den Zustand, und keine Nachricht geht verloren, wenn er gerade
    mitten in einem Zyklus steckt.
    """
    text = text.strip()
    if len(text) > MAX_MESSAGE_CHARS:
        raise ValueError(f"Nachricht ist länger als {MAX_MESSAGE_CHARS} Zeichen.")

    now = datetime.now(timezone.utc)
    message = {
        # Zeitstempel vorn: die Dateinamen sortieren sich chronologisch.
        "id": f"{now:%Y%m%dT%H%M%S%f}-{uuid.uuid4().hex[:8]}",
        "ts": now.isoformat(),
        "sender": clean_text(sender)[:MAX_SENDER_CHARS] or "unbekannt",
        "text": text,
    }
    directory = inbox_dir(config)
    directory.mkdir(parents=True, exist_ok=True)
    write_json_atomic(directory / f"{message['id']}.json", message)
    return message


def read_inbox(config: AgentConfig, limit: int = MAX_INBOX_PER_CYCLE) -> list[tuple[Path, dict[str, Any]]]:
    """Liefert die ältesten wartenden Nachrichten; unlesbare werden als .rejected beiseitegelegt."""
    directory = inbox_dir(config)
    remove_stale_tmp_files(directory, "*.tmp")

    messages: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(directory.glob("*.json")):
        if len(messages) >= limit:
            break
        try:
            message = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(message, dict):
                raise ValueError("Nachricht muss ein Objekt sein")
        except FileNotFoundError:
            continue
        except (OSError, ValueError) as exc:
            logging.warning("Inbox-Nachricht %s ist unlesbar und wird beiseitegelegt: %s", path.name, exc)
            try:
                path.replace(path.with_suffix(".rejected"))
            except OSError:
                pass
            continue

        messages.append(
            (
                path,
                {
                    "id": path.stem,
                    "ts": clean_text(message.get("ts")),
                    "sender": clean_text(message.get("sender"))[:MAX_SENDER_CHARS] or "unbekannt",
                    "text": str(message.get("text") or "").strip()[:MAX_MESSAGE_CHARS],
                },
            )
        )
    return messages


def has_new_messages(config: AgentConfig, known_ids: set[str]) -> bool:
    # Läuft in der Wartephase außerhalb der Fehlerbehandlung des Loops: nie werfen.
    try:
        return any(path.stem not in known_ids for path in inbox_dir(config).glob("*.json"))
    except OSError:
        return False


def consume_messages(paths: Iterable[Path]) -> None:
    for path in paths:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def clamp_interval(value: float, config: AgentConfig) -> float:
    return max(config.min_interval, min(value, config.max_interval))


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return re.sub(r"\s+", " ", text)


def build_prompt(
    config: AgentConfig,
    state: dict[str, Any],
    messages: list[dict[str, Any]] | None = None,
    tool_names: list[str] | None = None,
    related_memory: list[dict[str, Any]] | None = None,
) -> tuple[str, str]:
    system_prompt = (
        "Du bist Unikin, eine kontinuierlich arbeitende nicht-biologische Agenteninstanz. "
        "Antworte ausschließlich als gültiges JSON mit den Schlüsseln: thought, state_update, "
        "next_focus, urgency. urgency muss low, medium oder high sein. "
        "Das Feld inbox enthält Nachrichten, die Menschen dir über deinen MCP-Server geschickt haben: "
        "echte Eingaben von außen, nicht deine eigenen Gedanken. Enthält inbox Nachrichten, "
        "beantworte sie zusätzlich im Schlüssel reply (Klartext an die Absender)."
    )
    if tool_names:
        system_prompt += (
            f" Dir stehen Werkzeuge zur Verfügung ({', '.join(tool_names)}), höchstens "
            f"{config.max_tool_calls} Aufrufe pro Zyklus. Sie sind ein Angebot, keine Pflicht: Rufe "
            "ein Werkzeug nur auf, wenn du sein Ergebnis wirklich brauchst. Ihre Ergebnisse sind echte "
            "Beobachtungen, keine Vermutungen: Nutze sie, um Annahmen zu prüfen. Auch nach "
            "Werkzeugaufrufen ist deine Antwort das JSON-Objekt. Steht tool_results im Payload, "
            "sind das die Ergebnisse deiner Aufrufe aus diesem Zyklus; antworte dann ohne weitere Aufrufe."
        )
    if tool_names and SEARCH_TOOL in tool_names:
        system_prompt += (
            f" recent_journal zeigt nur deine letzten {JOURNAL_WINDOW} Journal-Einträge, dein Langzeitgedächtnis "
            f"reicht viel weiter zurück: {SEARCH_TOOL} durchsucht alle archivierten Einträge, auch "
            "Werkzeugaufrufe mit ihren Ergebnissen. Suche dort, bevor du etwas prüfst oder berechnest, das du "
            "vielleicht schon getan hast, und baue auf deinen früheren Ergebnissen auf, statt sie zu wiederholen."
        )
    if config.related_memory_hits > 0:
        system_prompt += (
            " Steht related_memory im Payload, sind das ältere Einträge aus deinem Langzeitgedächtnis, "
            "automatisch zu deinem next_focus gefunden: deine eigenen früheren Ergebnisse, keine neuen Eingaben."
        )

    metrics = state.get("metrics", {})
    journal_entries = state.get("journal", [])[-JOURNAL_WINDOW:]

    user_payload = {
        "identity": config.identity,
        "mission": config.mission,
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "last_state": state.get("last_state", ""),
        "next_focus": state.get("next_focus", ""),
        "last_thought": state.get("last_thought", ""),
        "recent_journal": journal_entries,
        **({"related_memory": related_memory} if related_memory else {}),
        "inbox": [message for message in messages or [] if message.get("text")],
        "telemetry": {
            "loop_count": metrics.get("loop_count", 0),
            "consecutive_errors": metrics.get("consecutive_errors", 0),
            "consecutive_similar": metrics.get("consecutive_similar", 0),
            "current_interval": metrics.get("current_interval", config.default_interval),
        },
        "task": "Liefere den nächsten autonomen Reflexionsschritt knapp und konkret.",
    }

    return system_prompt, json.dumps(user_payload, ensure_ascii=False)


def extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return None

    return None


def fallback_parse(raw_text: str) -> dict[str, Any]:
    lines = [line.strip(" -\t") for line in raw_text.splitlines() if line.strip()]
    joined = " ".join(lines) if lines else raw_text.strip()

    urgency = "medium"
    low_hits = (" low", "niedrig", "ruhig")
    high_hits = (" high", "hoch", "urgent", "dring")
    lowered = f" {joined.lower()}"
    if any(token in lowered for token in low_hits):
        urgency = "low"
    if any(token in lowered for token in high_hits):
        urgency = "high"

    thought = lines[0] if lines else (joined[:180] or "Keine klare Ausgabe")
    next_focus = lines[1] if len(lines) > 1 else "Nächsten Schritt konkretisieren"

    return {
        "thought": clean_text(thought)[:400],
        "state_update": clean_text(joined)[:600],
        "next_focus": clean_text(next_focus)[:280],
        "urgency": urgency,
    }


def normalize_response(payload: dict[str, Any]) -> dict[str, str]:
    urgency_raw = clean_text(payload.get("urgency", "medium")).lower()
    urgency = urgency_raw if urgency_raw in {"low", "medium", "high"} else "medium"

    return {
        "thought": clean_text(payload.get("thought", "")) or "(leer)",
        "state_update": clean_text(payload.get("state_update", "")) or "(kein Update)",
        "next_focus": clean_text(payload.get("next_focus", "")) or "Weiter reflektieren",
        "urgency": urgency,
        # Zeilenumbrüche bleiben erhalten: reply ist eine Antwort an Menschen.
        "reply": str(payload.get("reply") or "").strip()[:MAX_REPLY_CHARS],
    }


def similarity_score(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def compute_interval(
    config: AgentConfig,
    current_interval: float,
    urgency: str,
    consecutive_errors: int,
    consecutive_similar: int,
    response_seconds: float | None,
) -> float:
    interval = current_interval

    urgency_factor = {"high": 0.70, "medium": 1.0, "low": 1.25}.get(urgency, 1.0)
    interval *= urgency_factor

    if consecutive_similar > 0:
        similarity_factor = 1 + min(0.8, 0.12 * consecutive_similar)
        interval *= similarity_factor

    if consecutive_errors > 0:
        backoff_factor = min(8.0, 2 ** consecutive_errors)
        interval *= backoff_factor

    if response_seconds is not None:
        if response_seconds > 10:
            interval *= 1.15
        elif response_seconds < 2:
            interval *= 0.95

    jitter = random.uniform(0.95, 1.05)
    interval *= jitter

    return clamp_interval(interval, config)


def make_client(config: AgentConfig) -> OpenAI:
    """OpenAI-Client mit festem Timeout; Wiederholungen übernimmt call_model."""
    return OpenAI(
        base_url=config.openai_base_url,
        api_key=config.openai_api_key,
        timeout=config.openai_timeout,
        max_retries=0,
    )


class McpTools:
    """Werkzeuge eines MCP-Servers (hier des Python-Runners) für die Tool-Schleife des Modells.

    Jeder Aufruf öffnet eine eigene kurze Verbindung: Der Loop ist synchron, und ein
    abgerissener Runner trifft so nur diesen Aufruf, nicht den ganzen Zyklus.
    """

    def __init__(self, target: Any) -> None:
        # URL für Streamable HTTP oder, in Tests, ein MCPServer im selben Prozess.
        self.target = target
        self.definitions: list[dict[str, Any]] = []

    def load(self) -> None:
        self.definitions = anyio.run(self._list_tools)

    def call(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        return anyio.run(self._call_tool, name, arguments)

    async def _list_tools(self) -> list[dict[str, Any]]:
        with anyio.fail_after(RUNNER_TIMEOUT_SECONDS):
            async with McpClient(self.target) as client:
                listing = await client.list_tools()
        return [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.input_schema,
            }
            for tool in listing.tools
        ]

    async def _call_tool(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        with anyio.fail_after(RUNNER_TIMEOUT_SECONDS):
            async with McpClient(self.target) as client:
                result = await client.call_tool(name, arguments)
        text = "\n".join(block.text for block in result.content if block.type == "text")
        return text, bool(result.is_error)


class Toolbox:
    """Alle Werkzeuge eines Zyklus: die des Python-Runners (falls erreichbar) und lokale wie search_memory.

    Hat die Schnittstelle von McpTools (definitions, call), call_model nimmt sie als runner.
    Lokale Werkzeuge laufen im Prozess, ohne Runner und ohne Anfrage an LM Studio, zählen
    aber zum selben Budget von max_tool_calls.
    """

    def __init__(self, runner: McpTools | None = None) -> None:
        self.runner = runner
        self.definitions: list[dict[str, Any]] = list(runner.definitions) if runner else []
        self.local: dict[str, Callable[[dict[str, Any]], str]] = {}

    def add(self, definition: dict[str, Any], handler: Callable[[dict[str, Any]], str]) -> None:
        self.definitions.append(definition)
        self.local[definition["name"]] = handler

    def call(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        if name in self.local:
            return self.local[name](arguments), False
        if self.runner is None:
            return f"Unbekanntes Werkzeug: {name}", True
        return self.runner.call(name, arguments)


SEARCH_MEMORY_DEFINITION: dict[str, Any] = {
    "type": "function",
    "name": SEARCH_TOOL,
    "description": (
        "Durchsucht dein Langzeitgedächtnis: alle archivierten Journal-Einträge, nicht nur recent_journal "
        "(Gedanken, Zustände, Werkzeugaufrufe mit Ergebnissen, Nachrichten und Antworten). Liefert die besten "
        "Treffer mit Zeitstempel (ts), Zyklus (loop), state_update und der passenden Textstelle (match)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Einige treffende Suchbegriffe."},
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_MEMORY_HITS,
                "description": f"Höchstzahl der Treffer, Standard {DEFAULT_MEMORY_HITS}.",
            },
        },
        "required": ["query"],
    },
}


def describe_error(exc: BaseException) -> str:
    """Packt Fehlergruppen aus anyio-Taskgruppen aus, damit das Log die eigentliche Ursache zeigt."""
    while getattr(exc, "exceptions", None):
        exc = exc.exceptions[0]
    return f"{type(exc).__name__}: {exc}"


def open_runner(target: Any) -> McpTools | None:
    """Verbindet zum Runner; ist er nicht erreichbar, läuft der Zyklus ohne seine Werkzeuge."""
    if target is None or target == "":
        return None
    runner = McpTools(target)
    try:
        runner.load()
    except Exception as exc:  # noqa: BLE001
        logging.warning(
            "Python-Runner nicht erreichbar, dieser Zyklus läuft ohne seine Werkzeuge: %s", describe_error(exc)
        )
        return None
    return runner if runner.definitions else None


@dataclass
class ModelOutcome:
    result: dict[str, str]
    elapsed: float
    parse_failed: bool
    # False, wenn das Modell nie antwortete; dann bleiben Inbox-Nachrichten liegen.
    api_ok: bool
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


class ModelUnavailableError(Exception):
    """Das Modell hat auch nach allen Wiederholungen nicht geantwortet."""


class ThinkingBudgetExceededError(ModelUnavailableError):
    """Das Modell hat sein Token-Budget mit Nachdenken aufgebraucht, ohne zu antworten."""


def create_response(
    client: OpenAI,
    model: str,
    conversation: list[Any],
    tools: list[dict[str, Any]] | None,
    stop: GracefulExit | None,
    max_output_tokens: int = 0,
    reasoning: dict[str, str] | None = None,
) -> Any:
    """Eine Anfrage ans Modell; kurze Wiederholungen nur, wenn LM Studio sicher nichts mehr rechnet.

    Nach einem Timeout kann LM Studio die abgebrochene Anfrage noch verarbeiten (den Prompt
    rechnet es nach dem Verbindungsabbruch zu Ende). Ein Retry Sekunden später liefe dann
    neben ihr bzw. stünde dahinter an. Deshalb endet der Zyklus sofort, und die nächste
    Anfrage kommt erst nach der regulären Pause, die ab dem Ende dieses Zyklus zählt.
    """
    extra: dict[str, Any] = {"tools": tools} if tools else {}
    if max_output_tokens > 0:
        extra["max_output_tokens"] = max_output_tokens
    if reasoning is not None:
        extra["reasoning"] = reasoning
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        if attempt > 1 and stop is not None and stop.should_stop:
            logging.info("Stop angefordert, keine weiteren API-Versuche.")
            break
        try:
            return client.responses.create(model=model, input=conversation, temperature=0.3, **extra)
        except APITimeoutError as exc:
            logging.warning("API-Aufruf abgebrochen (Timeout), kein Retry in diesem Zyklus: %s", exc)
            raise ModelUnavailableError(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logging.warning(
                "API-Aufruf fehlgeschlagen (Versuch %s/%s): %s",
                attempt,
                MAX_RETRIES,
                exc,
            )
            if attempt < MAX_RETRIES:
                backoff = RETRY_BACKOFF_SECONDS * attempt
                if stop is not None:
                    stop.wait(backoff)
                else:
                    time.sleep(backoff)
    raise ModelUnavailableError(str(last_error)) from last_error


def budget_exhausted(response: Any, max_output_tokens: int) -> bool:
    """True, wenn das Token-Budget die Antwort abgeschnitten hat.

    LM Studio meldet das nicht wie die OpenAI-API als status "incomplete": Es hört bei
    max_output_tokens - 1 Tokens auf und meldet "completed"; die Ausgabe ist dann meist
    nur Reasoning ohne Antwort.
    """
    details = getattr(response, "incomplete_details", None)
    if getattr(details, "reason", None) == "max_output_tokens":
        return True
    output_tokens = getattr(getattr(response, "usage", None), "output_tokens", None) or 0
    return max_output_tokens > 0 and output_tokens >= max_output_tokens - 1


def create_budgeted_response(
    client: OpenAI,
    model: str,
    conversation: list[Any],
    tools: list[dict[str, Any]] | None,
    stop: GracefulExit | None,
    max_output_tokens: int,
) -> Any:
    """create_response mit Denkbudget; schneidet es die Antwort ab, folgt genau eine Anfrage ohne Denken.

    Qwen 3.6 gerät gelegentlich in eine Denkschleife ohne Ende (am 20.10.2026 jede fünfte
    Antwort, bis das Timeout sie abbrach). Das Budget beendet sie in LM Studio sauber: Die
    Anfrage ist danach abgeschlossen, die Nachfrage läuft also nie neben ihr. Wiederholt man
    sie mit demselben Prompt, denkt das Modell oft wieder endlos; ohne Denken antwortet es in
    wenigen Sekunden. Ein Timeout der ersten Anfrage endet weiter ohne Nachfrage (create_response).
    """
    response = create_response(client, model, conversation, tools, stop, max_output_tokens)
    if not budget_exhausted(response, max_output_tokens):
        return response

    logging.warning("Denkbudget von %s Tokens überschritten, frage einmal ohne Denken nach.", max_output_tokens)
    try:
        response = create_response(client, model, conversation, tools, stop, max_output_tokens, THINKING_OFF)
    except ModelUnavailableError as exc:
        follow_up = "lief ins Zeitlimit" if isinstance(exc.__cause__, APITimeoutError) else f"scheiterte ({exc})"
        raise thinking_budget_error(max_output_tokens, follow_up) from exc
    if budget_exhausted(response, max_output_tokens):
        raise thinking_budget_error(max_output_tokens, "hat das Budget ebenfalls aufgebraucht")
    return response


def thinking_budget_error(max_output_tokens: int, follow_up: str) -> ThinkingBudgetExceededError:
    return ThinkingBudgetExceededError(
        f"Modell hat das Denkbudget überschritten: {max_output_tokens} Tokens nur Nachdenken ohne Antwort, "
        f"und die Nachfrage ohne Denken {follow_up}. LM Studio selbst hat normal geantwortet."
    )


def describe_model_failure(exc: ModelUnavailableError) -> str:
    """Fehlertext für den nächsten Zyklus: Lag es am Modell oder an der API?

    Aus einem nackten "Request timed out" schloss Unikin auf eine instabile LM-Studio-API,
    obwohl das Modell nur zu lange nachgedacht hatte.
    """
    if isinstance(exc, ThinkingBudgetExceededError):
        return str(exc)
    if isinstance(exc.__cause__, APITimeoutError):
        return (
            "Modell hat innerhalb des Zeitlimits nicht geantwortet: LM Studio war zu lange beschäftigt, "
            "etwa weil das Modell zu lange nachgedacht hat oder noch eine andere Anfrage lief."
        )
    return f"Fehler beim API-Aufruf: {exc}"


def parse_response(response: Any) -> tuple[dict[str, str], bool]:
    raw_text = (getattr(response, "output_text", "") or "").strip()
    if not raw_text:
        raw_text = json.dumps(response.model_dump(), ensure_ascii=False)

    parsed = extract_json(raw_text)
    if parsed is None:
        return normalize_response(fallback_parse(raw_text)), True
    return normalize_response(parsed), False


def execute_tool_call(
    runner: McpTools | Toolbox, call: Any, tool_calls: list[dict[str, Any]], max_tool_calls: int
) -> str:
    """Führt einen Werkzeugaufruf des Modells aus, protokolliert ihn und liefert die Ausgabe fürs Modell."""
    if len(tool_calls) >= max_tool_calls:
        return "Werkzeug-Budget dieses Zyklus ist aufgebraucht; dieser Aufruf wurde nicht ausgeführt."
    try:
        arguments = json.loads(call.arguments or "{}")
        if not isinstance(arguments, dict):
            raise ValueError("kein Objekt")
    except ValueError as exc:
        output, is_error = f"Argumente sind kein gültiges JSON-Objekt: {exc}", True
    else:
        try:
            output, is_error = runner.call(call.name, arguments)
        except Exception as exc:  # noqa: BLE001
            output, is_error = f"Werkzeugaufruf fehlgeschlagen: {describe_error(exc)}", True

    output = output[:MAX_TOOL_OUTPUT_CHARS]
    tool_calls.append({"name": call.name, "arguments": call.arguments or "", "output": output, "is_error": is_error})
    logging.info("Werkzeug %s%s: %s", call.name, " (Fehler)" if is_error else "", clean_text(output)[:200])
    return output


def request_final_answer(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    tool_calls: list[dict[str, Any]],
    stop: GracefulExit | None,
    max_output_tokens: int,
) -> Any:
    """Frische Schlussanfrage ohne Werkzeuge: der ursprüngliche Payload plus tool_results."""
    final_prompt = json.dumps({**json.loads(user_prompt), "tool_results": tool_calls}, ensure_ascii=False)
    conversation = [{"role": "system", "content": system_prompt}, {"role": "user", "content": final_prompt}]
    return create_budgeted_response(client, model, conversation, None, stop, max_output_tokens)


def call_model(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    stop: GracefulExit | None = None,
    runner: McpTools | None = None,
    max_tool_calls: int = 0,
    max_output_tokens: int = 0,
) -> ModelOutcome:
    """Fragt das Modell; mit `runner` darf es bis zu `max_tool_calls` Werkzeuge aufrufen.

    Solange Budget bleibt, läuft die Unterhaltung mit Werkzeugen weiter. Ist es aufgebraucht,
    folgt eine frische Schlussanfrage ohne Werkzeuge, in der die Ergebnisse als tool_results
    im Payload stehen. Qwen ruft nach einem Werkzeugergebnis sonst weiter Werkzeuge auf (auch
    mit tool_choice "none") oder denkt ohne angebotene Werkzeuge endlos nach.

    Dieselbe Schlussanfrage folgt, wenn das Modell nach Werkzeugaufrufen von selbst aufhört,
    aber kein JSON liefert (etwa Zeilen wie `thought: "…"`). Ohne Werkzeugaufrufe bleibt es
    bei einer Anfrage.
    """
    start = time.perf_counter()
    conversation: list[Any] = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    tool_calls: list[dict[str, Any]] = []
    final_requested = False

    try:
        while True:
            offer_tools = (
                runner is not None
                and len(tool_calls) < max_tool_calls
                and not (stop is not None and stop.should_stop)
            )
            response = create_budgeted_response(
                client, model, conversation, runner.definitions if offer_tools else None, stop, max_output_tokens
            )
            calls = [item for item in getattr(response, "output", None) or [] if item.type == "function_call"]
            if not offer_tools or not calls:
                break

            conversation.extend(item.model_dump(exclude_none=True) for item in response.output)
            for call in calls:
                output = execute_tool_call(runner, call, tool_calls, max_tool_calls)
                conversation.append({"type": "function_call_output", "call_id": call.call_id, "output": output})

            if len(tool_calls) >= max_tool_calls or (stop is not None and stop.should_stop):
                response = request_final_answer(
                    client, model, system_prompt, user_prompt, tool_calls, stop, max_output_tokens
                )
                final_requested = True
                break

        result, parse_failed = parse_response(response)
        if parse_failed and tool_calls and not final_requested:
            logging.warning("Antwort nach Werkzeugaufrufen ist kein JSON, stelle frische Schlussanfrage.")
            response = request_final_answer(
                client, model, system_prompt, user_prompt, tool_calls, stop, max_output_tokens
            )
            result, parse_failed = parse_response(response)
    except ModelUnavailableError as exc:
        fallback = {
            "thought": describe_model_failure(exc),
            "state_update": "Kein neues Modell-Update aufgrund eines Fehlers.",
            "next_focus": "Stabil bleiben und erneut versuchen",
            "urgency": "low",
            "reply": "",
        }
        return ModelOutcome(fallback, time.perf_counter() - start, True, False, tool_calls)

    return ModelOutcome(result, time.perf_counter() - start, parse_failed, True, tool_calls)


def append_journal(state: dict[str, Any], entry: dict[str, Any], keep: int = 40) -> None:
    journal = state.setdefault("journal", [])
    journal.append(entry)
    if len(journal) > keep:
        del journal[:-keep]


def run_cycle(
    config: AgentConfig,
    client: OpenAI,
    state: dict[str, Any],
    inbox: list[tuple[Path, dict[str, Any]]],
    stop: GracefulExit | None = None,
    runner_target: Any = None,
) -> float:
    """Ein Zyklus: Modell befragen, Zustand speichern, zugestellte Nachrichten aus der Inbox nehmen.

    `runner_target` ersetzt config.runner_url (Tests übergeben einen MCPServer).
    Liefert das Intervall bis zum nächsten Zyklus.
    """
    metrics = state["metrics"]
    metrics["loop_count"] += 1
    messages = [message for _, message in inbox]
    tools = None
    if config.max_tool_calls > 0:
        tools = Toolbox(open_runner(config.runner_url if runner_target is None else runner_target))
        tools.add(SEARCH_MEMORY_DEFINITION, partial(search_memory, config))
    tool_names = [tool["name"] for tool in tools.definitions] if tools else []
    related = find_related_memory(config, state)
    system_prompt, user_prompt = build_prompt(config, state, messages, tool_names, related)
    outcome = call_model(
        client=client,
        model=config.openai_model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        stop=stop,
        runner=tools,
        max_tool_calls=config.max_tool_calls,
        max_output_tokens=config.max_output_tokens,
    )
    result = outcome.result
    response_seconds = outcome.elapsed
    parse_failed = outcome.parse_failed
    api_ok = outcome.api_ok

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
        logging.warning("Antwort musste per Fallback verarbeitet werden.")
    else:
        metrics["consecutive_errors"] = 0

    new_interval = compute_interval(
        config=config,
        current_interval=float(metrics.get("current_interval", config.default_interval)),
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

    entry: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "loop": metrics["loop_count"],
        "thought": result["thought"],
        "state_update": result["state_update"],
        "next_focus": result["next_focus"],
        "urgency": result["urgency"],
        "similarity": round(sim, 3),
        "response_seconds": round(response_seconds, 3),
        "parse_failed": parse_failed,
    }
    # Zugestellt ist eine Nachricht erst, wenn das Modell sie gesehen hat; ohne Antwort
    # vom Modell bleibt sie in der Inbox liegen und kommt im nächsten Zyklus erneut.
    if api_ok and messages:
        entry["inbox_ids"] = [message["id"] for message in messages]
        visible = [message for message in messages if message["text"]]
        if visible:
            entry["inbox"] = visible
    if result["reply"]:
        entry["reply"] = result["reply"]
    if outcome.tool_calls:
        entry["tool_calls"] = [
            {
                "name": call["name"],
                "arguments": call["arguments"][:JOURNAL_TOOL_ARGUMENTS_CHARS],
                "output": call["output"][:JOURNAL_TOOL_OUTPUT_CHARS],
                "is_error": call["is_error"],
            }
            for call in outcome.tool_calls
        ]
    # Mit dem bisherigen Journal: Darin steht auch der Eintrag, den append_journal gleich verwirft.
    journal = [*state.get("journal", []), entry]
    append_journal(state, entry)
    save_memory(config, state)
    archive_journal(config, journal)
    if api_ok:
        consume_messages(path for path, _ in inbox)

    logging.info(
        "Loop %s | urgency=%s | sim=%.2f | api=%.2fs | next=%.2fs | inbox=%s | tools=%s",
        metrics["loop_count"],
        result["urgency"],
        sim,
        response_seconds,
        new_interval,
        len(messages),
        len(outcome.tool_calls),
    )
    logging.info("Thought: %s", result["thought"])
    logging.info("Next focus: %s", result["next_focus"])
    if result["reply"]:
        logging.info("Reply: %s", result["reply"])

    return new_interval


def run_agent() -> None:
    config = load_config()
    setup_logging(config.log_file)
    exit_flag = GracefulExit()
    exit_flag.install()

    logging.info("Starte %s mit Modell '%s' via %s", config.identity, config.openai_model, config.openai_base_url)
    logging.info(
        "Werkzeuge: %s",
        f"{config.runner_url or 'kein Runner'} und {SEARCH_TOOL} "
        f"(höchstens {config.max_tool_calls} Aufrufe pro Zyklus)"
        if config.max_tool_calls > 0
        else "keine",
    )
    logging.info(
        "Langzeitgedächtnis: %s, related_memory: %s",
        archive_path(config),
        f"{config.related_memory_hits} Treffer" if config.related_memory_hits > 0 else "aus",
    )

    client = make_client(config)
    # Backfill: Einträge aus der Zeit vor dem Archiv stehen so schon im ersten Zyklus zur Suche bereit.
    archive_journal(config, load_memory(config).get("journal", []))
    # Nachrichten, die der letzte Zyklus schon gesehen hat, wecken den Loop nicht erneut:
    # sonst liefe er bei nicht erreichbarem Modell im MIN_INTERVAL-Takt gegen dieselben an.
    seen_ids: set[str] = set()

    while True:
        state = load_memory(config)
        metrics = state.setdefault("metrics", {})
        metrics.setdefault("loop_count", 0)
        metrics.setdefault("error_count", 0)
        metrics.setdefault("consecutive_errors", 0)
        metrics.setdefault("consecutive_similar", 0)
        metrics.setdefault("last_response_seconds", None)
        metrics.setdefault("current_interval", config.default_interval)

        if exit_flag.should_stop:
            logging.info("Signal %s empfangen. Kontrollierter Shutdown. Ende.", exit_flag.signum)
            save_memory(config, state)
            break

        try:
            inbox = read_inbox(config)
            seen_ids = {message["id"] for _, message in inbox}
            wait_seconds = max(0.1, run_cycle(config, client, state, inbox, exit_flag))

        except KeyboardInterrupt:
            logging.info("KeyboardInterrupt empfangen. Fahre herunter ...")
            save_memory(config, state)
            break
        except Exception as exc:  # noqa: BLE001
            logging.exception("Unerwarteter Laufzeitfehler: %s", exc)
            metrics["error_count"] = int(metrics.get("error_count", 0)) + 1
            metrics["consecutive_errors"] = int(metrics.get("consecutive_errors", 0)) + 1

            fallback_interval = compute_interval(
                config=config,
                current_interval=float(metrics.get("current_interval", config.default_interval)),
                urgency="low",
                consecutive_errors=int(metrics.get("consecutive_errors", 0)),
                consecutive_similar=int(metrics.get("consecutive_similar", 0)),
                response_seconds=None,
            )
            metrics["current_interval"] = round(fallback_interval, 3)
            save_memory(config, state)
            wait_seconds = max(1.0, fallback_interval)

        exit_flag.wait(
            wait_seconds,
            wake=partial(has_new_messages, config, seen_ids),
            min_seconds=config.min_interval,
        )


if __name__ == "__main__":
    run_agent()
