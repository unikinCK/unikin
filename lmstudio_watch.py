"""Erkennt im Serverlog von LM Studio Anfragen anderer Anwendungen, damit Unikin ihnen den Vorrang lässt.

LM Studio bedient jedes Modell mit PARALLEL 1: Eine Anfrage wartet, bis die vor ihr fertig ist.
Läuft gerade ein Zyklus von Unikin, stehen LibreChat, die Buchhaltung oder das Prüfsystem dahinter
an. Deshalb pausiert Unikin, sobald eine andere Anwendung LM Studio nutzt, und arbeitet erst weiter,
wenn eine Abklingzeit lang (Standard 20 Minuten) keine fremde Anfrage mehr kam.

Die HTTP-API von LM Studio verrät nicht, ob und für wen ein Modell gerade rechnet. Das Serverlog
(`~/.lmstudio/server-logs/JJJJ-MM/JJJJ-MM-TT.N.log`) tut es, wenn in LM Studio "Verbose Logging"
an ist. Jede Anfrage steht dort als Block:

    [2026-10-22 21:45:56][DEBUG] Received request: POST to /v1/chat/completions with body {
      "model": "qwen/qwen3.5-122b-a10b",
      "messages": [
        {
          "role": "system",
          "content": "Du bist ein striktes visuelles Prüfsystem für Werk... <Truncated in logs> ...derselben Kamera."

Lange Texte kürzt LM Studio auf ihre ersten und letzten 50 Zeichen. Unikins eigene Anfragen erkennt
der Wächter am Anfang seines Systemprompts (OWN_REQUEST_MARKER), jede andere Anfrage gilt als fremd.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# So beginnt Unikins Systemprompt (build_prompt). LM Studio behält von langen Texten nur die ersten
# LOG_TRUNCATION_CHARS Zeichen, die Kennung muss darin stehen.
OWN_REQUEST_MARKER = "Du bist Unikin, eine kontinuierlich"
LOG_TRUNCATION_CHARS = 50
# Embeddings rechnet ein eigenes kleines Modell; sie warten nicht hinter Unikins Anfragen.
IGNORED_ENDPOINT_SUFFIXES = ("/embeddings",)
PROMPT_PREVIEW_CHARS = 40
READ_CHUNK_BYTES = 1024 * 1024
# Beim Start liest der Wächter so viele der jüngsten Dateien ganz; LM Studio beginnt bei rund 10 MB
# und an jedem Tag eine neue, die Abklingzeit liegt damit sicher darin.
STARTUP_FILES = 2

LOG_FILE_NAME = re.compile(r"^(\d{4}-\d{2}-\d{2})\.(\d+)\.log$")
LOG_LINE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")
REQUEST_LINE = re.compile(r"Received request: POST to (\S+)")
MODEL_FIELD = re.compile(r'^\s*"model": "([^"]*)"')
PROMPT_FIELD = re.compile(r'^\s*"(?:content|instructions|input|prompt|text)": "(.*)$')
TRUNCATION = "... <Truncated in logs>"
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


@dataclass
class LoggedRequest:
    """Eine Anfrage aus dem Serverlog; `ts` ist die Ortszeit des Hosts, wie LM Studio sie schreibt."""

    ts: datetime
    endpoint: str
    model: str = ""
    # Anfang des ersten Prompt-Texts, nur fürs Log.
    prompt: str = ""
    own: bool = False

    @property
    def foreign(self) -> bool:
        return not self.own and not self.endpoint.endswith(IGNORED_ENDPOINT_SUFFIXES)

    def describe(self, with_prompt: bool = True) -> str:
        details = [self.model] if self.model else []
        if with_prompt and self.prompt:
            details.append(f"„{self.prompt}…“")
        return f"POST {self.endpoint}" + (f" ({', '.join(details)})" if details else "")


class LmStudioLog:
    """Liest das Serverlog von LM Studio fortlaufend und liefert die Anfragen darin.

    Eine Anfrage ist vollständig, sobald die nächste Logzeile beginnt; LM Studio schreibt direkt
    danach "Running chat completion …". Zwischen zwei Aufrufen merkt sich der Leser Datei und
    Position, auch über den Wechsel auf eine neue Datei hinweg.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.path: Path | None = None
        self.offset = 0
        self.partial = b""
        self.block: LoggedRequest | None = None
        # Jüngster Zeitstempel im Log: die Uhr, an der read_new das Alter einer Anfrage misst.
        self.last_ts: datetime | None = None

    def files(self) -> list[Path]:
        """Alle Logdateien, älteste zuerst; N in JJJJ-MM-TT.N.log zählt innerhalb eines Tages."""
        found = []
        for path in self.directory.glob("*/*.log"):
            match = LOG_FILE_NAME.match(path.name)
            if match:
                found.append(((match.group(1), int(match.group(2))), path))
        return [path for _, path in sorted(found)]

    def read_recent(self) -> tuple[list[LoggedRequest], float]:
        """Liest beim Start die jüngsten Dateien ganz; danach setzt read_new an ihrem Ende fort.

        Liefert auch, seit wie vielen Sekunden das Log still ist (Änderungszeit der jüngsten Datei):
        Die Zeitstempel im Log sind Ortszeit ohne Zone, das Alter einer Anfrage ergibt sich so aus
        dem Abstand zur letzten Zeile plus dieser Stille, ohne die Zeitzone des Hosts zu kennen.
        """
        paths = self.files()[-STARTUP_FILES:]
        requests: list[LoggedRequest] = []
        for path in paths:
            self.path, self.offset, self.partial = path, 0, b""
            requests.extend(self._read_path())
        quiet = max(0.0, time.time() - paths[-1].stat().st_mtime) if paths else 0.0
        return requests, quiet

    def read_new(self) -> list[LoggedRequest]:
        """Anfragen, die seit dem letzten Aufruf dazugekommen sind; liest bei einem Dateiwechsel die alte zu Ende."""
        files = self.files()
        if not files:
            return []
        newest = files[-1]
        requests: list[LoggedRequest] = []
        if self.path is not None and self.path != newest and self.path.exists():
            requests.extend(self._read_path())
        if self.path != newest:
            self.path, self.offset, self.partial = newest, 0, b""
        requests.extend(self._read_path())
        return requests

    def _read_path(self) -> list[LoggedRequest]:
        assert self.path is not None
        requests: list[LoggedRequest] = []
        with self.path.open("rb") as handle:
            if handle.seek(0, 2) < self.offset:
                # Datei ersetzt oder gekürzt: von vorn.
                self.offset, self.partial = 0, b""
            handle.seek(self.offset)
            while chunk := handle.read(READ_CHUNK_BYTES):
                self.offset += len(chunk)
                lines = (self.partial + chunk).split(b"\n")
                # Eine angefangene Zeile wartet auf ihren Rest.
                self.partial = lines.pop()
                requests.extend(self._parse(line.decode("utf-8", "replace") for line in lines))
        return requests

    def _parse(self, lines: Iterable[str]) -> Iterator[LoggedRequest]:
        for line in lines:
            stamp = LOG_LINE.match(line)
            if stamp is None:
                if self.block is not None:
                    self._inspect(self.block, line)
                continue
            if self.block is not None:
                yield self.block
                self.block = None
            try:
                self.last_ts = datetime.strptime(stamp.group(1), TIMESTAMP_FORMAT)
            except ValueError:
                continue
            request = REQUEST_LINE.search(line, stamp.end())
            if request:
                self.block = LoggedRequest(self.last_ts, request.group(1))

    @staticmethod
    def _inspect(block: LoggedRequest, line: str) -> None:
        if not block.model and (match := MODEL_FIELD.match(line)):
            block.model = match.group(1)
        elif match := PROMPT_FIELD.match(line):
            text = match.group(1)
            # Jeder Text zählt, nicht nur der erste: Lieber erkennt der Wächter eine fremde Anfrage
            # nicht, als dass Unikin vor seinen eigenen pausiert.
            if text.startswith(OWN_REQUEST_MARKER):
                block.own = True
            if not block.prompt:
                preview = re.sub(r'",?$', "", text.split(TRUNCATION)[0]).replace("\\n", " ")
                block.prompt = preview[:PROMPT_PREVIEW_CHARS].strip()


class ForeignRequestGuard:
    """Pausiert Unikin, solange andere Anwendungen LM Studio nutzen, und `cooldown` Sekunden darüber hinaus.

    Ohne Log oder mit cooldown 0 pausiert er nie. Keine Methode wirft: Der Wächter läuft auch in der
    Wartephase des Loops, außerhalb von dessen Fehlerbehandlung.
    """

    def __init__(
        self, log: LmStudioLog | None, cooldown: float, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.log = log if cooldown > 0 else None
        self.cooldown = cooldown
        self.clock = clock
        # Ende der Pause auf der monotonen Uhr; until_utc nur zur Anzeige.
        self.until = 0.0
        self.until_utc: datetime | None = None
        self.last_request: LoggedRequest | None = None
        self.active = False
        self.started = 0.0
        self.count = 0
        self.read_failed = False

    def start(self) -> None:
        """Liest das Log beim Start: Fremde Anfragen aus der Abklingzeit vor dem Start zählen mit."""
        if self.log is None:
            return
        try:
            if not self.log.files():
                logging.warning(
                    "Kein Serverlog von LM Studio unter %s: Unikin erkennt fremde Anfragen erst, wenn dort eines "
                    "entsteht. Ist das Verzeichnis eingebunden?",
                    self.log.directory,
                )
            requests, quiet = self.log.read_recent()
        except OSError as exc:
            logging.warning("Serverlog von LM Studio nicht lesbar: %s", exc)
            return
        self._consider(requests, quiet)

    def poll(self) -> bool:
        """Liest neue Logzeilen; True, wenn darin eine fremde Anfrage stand."""
        if self.log is None:
            return False
        try:
            requests = self.log.read_new()
        except OSError as exc:
            if not self.read_failed:
                logging.warning("Serverlog von LM Studio nicht lesbar, fremde Anfragen bleiben unerkannt: %s", exc)
            self.read_failed = True
            return False
        self.read_failed = False
        return self._consider(requests)

    def _consider(self, requests: list[LoggedRequest], quiet: float = 0.0) -> bool:
        last_ts = self.log.last_ts if self.log is not None else None

        def time_left(request: LoggedRequest) -> float:
            age = quiet + (max(0.0, (last_ts - request.ts).total_seconds()) if last_ts else 0.0)
            return self.cooldown - age

        # Beim Start stehen auch ältere fremde Anfragen im Log; zählen nur die aus der Abklingzeit.
        foreign = [request for request in requests if request.foreign and time_left(request) > 0]
        if not foreign:
            return False
        latest = foreign[-1]
        left = time_left(latest)

        now = self.clock()
        if now + left > self.until:
            self.until = now + left
            self.until_utc = (datetime.now(timezone.utc) + timedelta(seconds=left)).replace(microsecond=0)
            self.last_request = latest
        if not self.active:
            self.active, self.started, self.count = True, now, 0
            logging.info(
                "Andere Anwendung nutzt LM Studio: %s. Unikin pausiert bis %s UTC, %s Minuten nach der letzten "
                "fremden Anfrage.",
                latest.describe(),
                self.until_utc.strftime("%H:%M:%S") if self.until_utc else "?",
                f"{self.cooldown / 60:g}",
            )
        self.count += len(foreign)
        return True

    def remaining(self) -> float:
        return max(0.0, self.until - self.clock())

    def paused(self) -> bool:
        """Liest neue Logzeilen und sagt, ob Unikin pausiert; meldet das Ende einer Pause im Log."""
        self.poll()
        if self.remaining() > 0:
            return True
        if self.active:
            self.active = False
            logging.info(
                "Pause beendet nach %.0f Minuten (%s fremde Anfragen an LM Studio), Unikin arbeitet weiter.",
                (self.clock() - self.started) / 60,
                self.count,
            )
        return False

    def info(self) -> dict[str, Any] | None:
        """Die laufende Pause für unikin_state.json, ohne Prompt-Text der fremden Anwendung."""
        if not self.active or self.until_utc is None or self.last_request is None:
            return None
        return {
            "until": self.until_utc.isoformat(),
            "reason": f"Andere Anwendung nutzt LM Studio: {self.last_request.describe(with_prompt=False)}",
        }


class LmStudioBusyError(Exception):
    """Unikin hält eine Anfrage zurück, weil eine andere Anwendung LM Studio nutzt."""


class YieldingClient:
    """Steht für den OpenAI-Client und lässt eine Anfrage nur durch, wenn der Wächter nicht pausiert.

    So prüft Unikin vor jeder einzelnen Anfrage eines Zyklus (Werkzeugschleife, Schlussanfrage,
    Nachfrage ohne Denken), ob inzwischen eine andere Anwendung LM Studio nutzt, und bricht den
    Zyklus dann ab. Eine laufende Anfrage bricht er nicht ab: Den Prompt rechnet LM Studio auch nach
    einem Verbindungsabbruch zu Ende.
    """

    def __init__(self, client: Any, guard: ForeignRequestGuard) -> None:
        self.client = client
        self.guard = guard
        self.responses = self

    def create(self, **kwargs: Any) -> Any:
        if self.guard.paused():
            reason = self.guard.last_request.describe(with_prompt=False) if self.guard.last_request else ""
            raise LmStudioBusyError(f"Andere Anwendung nutzt LM Studio: {reason}")
        return self.client.responses.create(**kwargs)
