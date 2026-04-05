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
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

JOURNAL_WINDOW = 6
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0


@dataclass
class AgentConfig:
    openai_base_url: str
    openai_api_key: str
    openai_model: str
    identity: str
    mission: str
    min_interval: float
    max_interval: float
    default_interval: float
    memory_file: Path
    log_file: Path


class GracefulExit:
    """State holder for signal-based exits."""

    def __init__(self) -> None:
        self.should_stop = False

    def _handle_signal(self, signum: int, _frame: Any) -> None:
        logging.info("Signal %s empfangen. Fahre kontrolliert herunter ...", signum)
        self.should_stop = True

    def install(self) -> None:
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)


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

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
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


def load_memory(config: AgentConfig) -> dict[str, Any]:
    config.memory_file.parent.mkdir(parents=True, exist_ok=True)

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


def save_memory(config: AgentConfig, state: dict[str, Any]) -> None:
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    config.memory_file.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def clamp_interval(value: float, config: AgentConfig) -> float:
    return max(config.min_interval, min(value, config.max_interval))


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return re.sub(r"\s+", " ", text)


def build_prompt(config: AgentConfig, state: dict[str, Any]) -> tuple[str, str]:
    system_prompt = (
        "Du bist Unikin, eine kontinuierlich arbeitende nicht-biologische Agenteninstanz. "
        "Antworte ausschließlich als gültiges JSON mit den Schlüsseln: thought, state_update, "
        "next_focus, urgency. urgency muss low, medium oder high sein."
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


def call_model(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> tuple[dict[str, str], float, bool]:
    start = time.perf_counter()
    raw_text = ""
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
            )

            raw_text = (getattr(response, "output_text", "") or "").strip()
            if not raw_text:
                raw_text = json.dumps(response.model_dump(), ensure_ascii=False)

            parsed = extract_json(raw_text)
            parse_failed = False
            if parsed is None:
                parse_failed = True
                parsed = fallback_parse(raw_text)

            elapsed = time.perf_counter() - start
            normalized = normalize_response(parsed)
            return normalized, elapsed, parse_failed

        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logging.warning(
                "API-Aufruf fehlgeschlagen (Versuch %s/%s): %s",
                attempt,
                MAX_RETRIES,
                exc,
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    elapsed = time.perf_counter() - start
    fallback = {
        "thought": f"Fehler beim API-Aufruf: {last_error}",
        "state_update": "Kein neues Modell-Update aufgrund eines Fehlers.",
        "next_focus": "Stabil bleiben und erneut versuchen",
        "urgency": "low",
    }
    return fallback, elapsed, True


def append_journal(state: dict[str, Any], entry: dict[str, Any], keep: int = 40) -> None:
    journal = state.setdefault("journal", [])
    journal.append(entry)
    if len(journal) > keep:
        del journal[:-keep]


def run_agent() -> None:
    config = load_config()
    setup_logging(config.log_file)
    exit_flag = GracefulExit()
    exit_flag.install()

    logging.info("Starte %s mit Modell '%s' via %s", config.identity, config.openai_model, config.openai_base_url)

    client = OpenAI(base_url=config.openai_base_url, api_key=config.openai_api_key)

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
            logging.info("Kontrollierter Shutdown angefordert. Ende.")
            save_memory(config, state)
            break

        try:
            metrics["loop_count"] += 1
            system_prompt, user_prompt = build_prompt(config, state)
            result, response_seconds, parse_failed = call_model(
                client=client,
                model=config.openai_model,
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

            append_journal(
                state,
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "thought": result["thought"],
                    "state_update": result["state_update"],
                    "next_focus": result["next_focus"],
                    "urgency": result["urgency"],
                    "similarity": round(sim, 3),
                    "response_seconds": round(response_seconds, 3),
                    "parse_failed": parse_failed,
                },
            )
            save_memory(config, state)

            logging.info(
                "Loop %s | urgency=%s | sim=%.2f | api=%.2fs | next=%.2fs",
                metrics["loop_count"],
                result["urgency"],
                sim,
                response_seconds,
                new_interval,
            )
            logging.info("Thought: %s", result["thought"])
            logging.info("Next focus: %s", result["next_focus"])

            sleep_seconds = max(0.1, new_interval)
            time.sleep(sleep_seconds)

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
            time.sleep(max(1.0, fallback_interval))


if __name__ == "__main__":
    run_agent()
