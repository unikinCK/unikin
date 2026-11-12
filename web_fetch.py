"""Internetzugriff für Unikin: das Werkzeug fetch_url ruft eine Seite per HTTP-GET ab.

Es läuft im Loop-Prozess, nicht im Runner: Der Runner bleibt ohne Netz, und abgerufen wird nur
eine URL, kein Code des Modells. Erreichbar sind allein öffentliche Adressen. LM Studio auf dem
Host, der Runner, Tailscale und das LAN sind gesperrt, auch als Ziel einer Weiterleitung. Die
Verbindung geht an genau die geprüfte Adresse, damit eine zweite DNS-Antwort (DNS-Rebinding)
nicht doch ins LAN führt.
"""

from __future__ import annotations

import http.client
import ipaddress
import re
import socket
import time
import zlib
from collections.abc import Callable
from functools import partial
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urldefrag, urljoin, urlsplit

FETCH_TOOL = "fetch_url"
DEFAULT_WEB_CALLS = 2
FETCH_TIMEOUT_SECONDS = 20.0
# Gesamtzeit für Weiterleitungen und Lesen: Ein Server, der tröpfchenweise sendet, hält den Zyklus nicht auf.
FETCH_DEADLINE_SECONDS = 30.0
MAX_FETCH_BYTES = 2 * 1024 * 1024
MAX_REDIRECTS = 5
# Kein Browser-Kennzeichen: Wer automatisierte Abrufe nicht will, soll sie erkennen können.
USER_AGENT = "Mozilla/5.0 (compatible; LLM-Agent-Fetch/1.0)"
# Steht im Haupttext (main, article) weniger, ist die Seite anders gebaut und zählt der ganze Text.
MIN_MAIN_TEXT_CHARS = 200
MAX_LINK_TEXT_CHARS = 80

FETCH_URL_DEFINITION: dict[str, Any] = {
    "type": "function",
    "name": FETCH_TOOL,
    "description": (
        "Ruft eine Seite aus dem Internet ab (HTTP-GET, nur öffentliche http- und https-Adressen) und liefert "
        "ihren Text: bei HTML den lesbaren Inhalt ohne Navigation und Skripte, bei JSON, XML und Text den Inhalt "
        "selbst, gekürzt auf einige tausend Zeichen. Mit links=true kommen stattdessen die Links der Seite "
        "(Linktext und absolute URL), um von einer Übersichtsseite weiterzugehen. Eine Suchmaschine gibt es "
        "nicht; zum Suchen eignet sich die Wikipedia-API, etwa "
        "https://de.wikipedia.org/w/api.php?action=query&list=search&format=json&srsearch=<Begriffe>."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Vollständige URL mit http:// oder https://."},
            "links": {
                "type": "boolean",
                "description": "true liefert die Links der Seite statt ihres Texts. Standard false.",
            },
        },
        "required": ["url"],
    },
}

Resolver = Callable[[str, int], list[str]]


class FetchError(ValueError):
    """Die Seite ließ sich nicht abrufen oder nicht als Text lesen."""


class BlockedAddressError(FetchError):
    """Das Ziel ist keine öffentliche Adresse."""


def resolve_public(host: str, port: int) -> list[str]:
    """Löst den Namen auf; ist auch nur eine Adresse nicht öffentlich, wird nichts abgerufen."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise FetchError(f"Name nicht auflösbar: {host}") from exc
    addresses = list(dict.fromkeys(info[4][0] for info in infos))
    for address in addresses:
        # is_global ist False für Loopback, private Netze, Link-Local (auch 169.254.169.254),
        # Carrier-Grade-NAT (Tailscale 100.64/10) und auf IPv6 abgebildete IPv4-Adressen.
        if not ipaddress.ip_address(address.split("%", 1)[0]).is_global:
            raise BlockedAddressError(
                f"{host} ist keine öffentliche Adresse ({address}); abgerufen wird nur das Internet."
            )
    if not addresses:
        raise FetchError(f"Name nicht auflösbar: {host}")
    return addresses


def connect_pinned(
    addresses: list[str], address: tuple[str, int], timeout: Any = None, source_address: Any = None
) -> socket.socket:
    """Ersetzt socket.create_connection: verbindet nur zu den geprüften Adressen, nicht zum Namen."""
    last_error: OSError | None = None
    for ip in addresses:
        try:
            return socket.create_connection((ip, address[1]), timeout, source_address)
        except OSError as exc:
            last_error = exc
    raise last_error or OSError("keine Adresse zum Verbinden")


class PageText(HTMLParser):
    """Lesbarer Text und Links einer HTML-Seite, nur mit der Standardbibliothek."""

    SKIP = {
        "script", "style", "noscript", "template", "svg", "iframe", "head", "nav", "footer", "aside", "form",
        "button", "select",
    }  # fmt: skip
    MAIN = {"main", "article"}
    BLOCK = {
        "p", "div", "br", "hr", "li", "dt", "dd", "tr", "table", "ul", "ol", "pre", "blockquote", "section",
        "article", "main", "header", "figure", "figcaption", "h1", "h2", "h3", "h4", "h5", "h6",
    }  # fmt: skip
    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}

    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.title = ""
        self.links: list[tuple[str, str]] = []
        # Links aus main und article: der Inhalt der Seite, nicht ihre Navigation.
        self.main_links: list[tuple[str, str]] = []
        self._parts: list[str] = []
        self._main_parts: list[str] = []
        self._skip = 0
        self._main = 0
        self._pre = 0
        self._in_title = False
        self._link: tuple[str, list[str], bool] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title":
            # Nur der erste: Auch Grafiken (svg) haben title-Elemente.
            self._in_title = not self.title
        if tag == "body":
            # </head> darf fehlen; spätestens hier ist der Kopf vorbei.
            self._skip = 0
        if tag in self.VOID:
            if tag in self.BLOCK:
                self._add("\n")
            return
        if tag in self.SKIP:
            self._skip += 1
        if tag in self.MAIN:
            self._main += 1
        if tag == "pre":
            self._pre += 1
        if tag in self.BLOCK:
            self._add("\n")
        if tag == "li":
            self._add("- ")
        if tag == "a":
            self._link = (dict(attrs).get("href") or "", [], self._main > 0)

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        if tag in self.MAIN and self._main:
            self._main -= 1
        if tag == "pre" and self._pre:
            self._pre -= 1
        if tag in self.BLOCK:
            self._add("\n")
        if tag == "a" and self._link is not None:
            self._finish_link()

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
            return
        if not self._pre:
            # Wie im Browser: Zeilenumbrüche im Quelltext sind Leerzeichen, Absätze machen die Block-Tags.
            data = re.sub(r"\s+", " ", data)
        if self._link is not None:
            self._link[1].append(data)
        self._add(data)

    def _add(self, text: str) -> None:
        if self._skip:
            return
        self._parts.append(text)
        if self._main:
            self._main_parts.append(text)

    def _finish_link(self) -> None:
        href, parts, in_main = self._link or ("", [], False)
        self._link = None
        href = href.strip()
        if not href or href.startswith("#"):
            return
        url = urldefrag(urljoin(self.base_url, href)).url
        if urlsplit(url).scheme not in {"http", "https"}:
            return
        text = re.sub(r"\s+", " ", "".join(parts)).strip()[:MAX_LINK_TEXT_CHARS]
        self.links.append((text, url))
        if in_main:
            self.main_links.append((text, url))

    def text(self) -> str:
        main = tidy("".join(self._main_parts))
        return main if len(main) >= MIN_MAIN_TEXT_CHARS else tidy("".join(self._parts))


def tidy(text: str) -> str:
    """Leerzeichen je Zeile zusammenziehen, Leerzeilen weglassen: Die Ausgabe hat nur wenige tausend Zeichen."""
    # [^\S\n] ist jeder Leerraum außer dem Zeilenumbruch, auch das geschützte Leerzeichen.
    lines = (re.sub(r"[^\S\n]+", " ", line).strip() for line in text.split("\n"))
    return "\n".join(line for line in lines if line)


def decode_body(body: bytes, content_type: str, charset: str | None) -> str:
    if not charset and "html" in content_type:
        match = re.search(rb"<meta[^>]+charset=[\"']?([A-Za-z0-9_-]+)", body[:4096], flags=re.IGNORECASE)
        charset = match.group(1).decode("ascii") if match else None
    try:
        return body.decode(charset or "utf-8", errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")


def is_text(content_type: str) -> bool:
    return (
        not content_type
        or content_type.startswith("text/")
        or content_type.endswith(("/json", "+json", "/xml", "+xml", "/javascript"))
    )


def read_limited(response: http.client.HTTPResponse, deadline: float) -> bytes:
    chunks: list[bytes] = []
    size = 0
    while size <= MAX_FETCH_BYTES:
        if time.monotonic() > deadline:
            raise FetchError(f"Abruf dauerte länger als {FETCH_DEADLINE_SECONDS:g} Sekunden.")
        # read1 kehrt zurück, sobald Daten da sind; read(n) wartete, bis n Bytes beisammen sind.
        chunk = response.read1(64 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
        size += len(chunk)
    return b"".join(chunks)[:MAX_FETCH_BYTES]


def decompress(body: bytes, encoding: str) -> bytes:
    if encoding in {"", "identity"}:
        return body
    if encoding not in {"gzip", "x-gzip", "deflate"}:
        raise FetchError(f"Kodierung {encoding} wird nicht unterstützt.")
    # wbits 47: gzip und zlib automatisch; die Grenze schützt vor Archiven, die sich riesig entpacken.
    wbits = 47 if encoding != "deflate" else 15
    try:
        return zlib.decompressobj(wbits).decompress(body, MAX_FETCH_BYTES)
    except zlib.error as exc:
        if encoding == "deflate":
            # Manche Server schicken deflate ohne zlib-Kopf.
            try:
                return zlib.decompressobj(-15).decompress(body, MAX_FETCH_BYTES)
            except zlib.error:
                pass
        raise FetchError(f"Inhalt ließ sich nicht entpacken ({encoding}): {exc}") from exc


def get(url: str, resolve: Resolver, deadline: float) -> tuple[str, int, http.client.HTTPMessage, bytes]:
    """HTTP-GET mit geprüften Weiterleitungen; liefert finale URL, Status, Header und Rumpf."""
    for _ in range(MAX_REDIRECTS + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise FetchError(f"Abruf dauerte länger als {FETCH_DEADLINE_SECONDS:g} Sekunden.")
        parts = urlsplit(url)
        if parts.scheme not in {"http", "https"}:
            raise FetchError(f"Nur http- und https-URLs sind erlaubt, nicht: {url[:200]}")
        if not parts.hostname:
            raise FetchError(f"URL ohne Host: {url[:200]}")
        if parts.username or parts.password:
            raise FetchError("URLs mit Zugangsdaten werden nicht abgerufen.")
        try:
            port = parts.port or (443 if parts.scheme == "https" else 80)
        except ValueError as exc:
            raise FetchError(f"Ungültiger Port in {url[:200]}") from exc
        addresses = resolve(parts.hostname, port)

        connection_class = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
        connection = connection_class(parts.hostname, port, timeout=min(FETCH_TIMEOUT_SECONDS, remaining))
        # http.client verbindet über dieses Attribut; so geht die Verbindung an die geprüfte IP,
        # Host-Header, SNI und Zertifikatsprüfung behalten den Namen.
        connection._create_connection = partial(connect_pinned, addresses)  # type: ignore[attr-defined]
        path = parts.path or "/"
        if parts.query:
            path += f"?{parts.query}"
        try:
            connection.request(
                "GET",
                path,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,text/plain;q=0.9,*/*;q=0.5",
                    "Accept-Language": "de,en;q=0.8",
                    "Accept-Encoding": "gzip, deflate",
                },
            )
            response = connection.getresponse()
            location = response.getheader("Location")
            if response.status in {301, 302, 303, 307, 308} and location:
                url = urljoin(url, location)
                continue
            body = read_limited(response, deadline)
            return url, response.status, response.headers, body
        except (OSError, http.client.HTTPException) as exc:
            raise FetchError(f"Verbindung zu {parts.hostname} fehlgeschlagen: {type(exc).__name__}: {exc}") from exc
        finally:
            connection.close()
    raise FetchError(f"Mehr als {MAX_REDIRECTS} Weiterleitungen.")


def fit(header: str, body: str, max_chars: int) -> str:
    """Kopf plus so viel vom Rumpf, wie in max_chars passt, mit Hinweis, wenn gekürzt wurde."""
    text = f"{header}\n\n{body}" if body else header
    if len(text) <= max_chars:
        return text
    note = f"\n… gekürzt, die Seite hat {len(body)} Zeichen."
    return text[: max(0, max_chars - len(note))].rstrip() + note


def fetch_url(url: str, *, links: bool = False, max_chars: int = 4000, resolve: Resolver = resolve_public) -> str:
    """Das Werkzeug fetch_url: Text oder Links einer öffentlichen Seite, höchstens max_chars Zeichen."""
    url = url.strip()
    if not url:
        raise FetchError("url fehlt oder ist leer.")
    deadline = time.monotonic() + FETCH_DEADLINE_SECONDS
    final_url, status, headers, body = get(url, resolve, deadline)
    if status >= 400:
        raise FetchError(f"HTTP {status} für {final_url}")

    content_type = (headers.get_content_type() if headers.get("Content-Type") else "").lower()
    if not is_text(content_type):
        raise FetchError(f"Kein Text, sondern {content_type} ({len(body)} Bytes): {final_url}")
    body = decompress(body, (headers.get("Content-Encoding") or "").strip().lower())
    content = decode_body(body, content_type, headers.get_content_charset())

    header = f"URL: {final_url}"
    html = "html" in content_type or (
        not content_type and content.lstrip()[:15].lower().startswith(("<!doctype", "<html"))
    )
    if not html:
        if links:
            raise FetchError(f"links=true geht nur bei HTML-Seiten, diese ist {content_type or 'ohne Inhaltstyp'}.")
        return fit(header, content.strip(), max_chars)

    page = PageText(final_url)
    page.feed(content)
    page.close()
    title = re.sub(r"\s+", " ", page.title).strip()
    if title:
        header += f"\nTitel: {title}"
    if not links:
        return fit(header, page.text() or "(kein lesbarer Text)", max_chars)

    unique = list(dict.fromkeys([*page.main_links, *page.links]))
    lines: list[str] = []
    used = len(header) + 40
    for text, link in unique:
        line = f"- {text or '(ohne Text)'} → {link}"
        if used + len(line) + 1 > max_chars:
            break
        lines.append(line)
        used += len(line) + 1
    return f"{header}\nLinks ({len(lines)} von {len(unique)}):\n" + "\n".join(lines)
