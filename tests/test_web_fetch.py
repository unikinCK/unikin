from __future__ import annotations

import gzip
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import web_fetch

PAGE = """<!doctype html>
<html><head><title>Testseite  Kuramoto</title>
<script>var geheim = "nicht anzeigen";</script>
<style>body { color: red; }</style>
</head>
<body>
<nav><a href="/start">Start</a> <a href="/archiv">Archiv</a></nav>
<main>
<h1>Kopplung von Oszillatoren</h1>
<p>Das Kuramoto-Modell beschreibt, wie sich
   gekoppelte Oszillatoren synchronisieren.</p>
<ul><li>Erster Punkt</li><li>Zweiter Punkt</li></ul>
<p>Weiter mit <a href="artikel/2?x=1#abschnitt">dem zweiten Artikel</a> oder
<a href="https://example.org/extern">einer externen Quelle</a>.</p>
<p>Kontakt: <a href="mailto:a@example.org">Mail</a>, <a href="#oben">nach oben</a>.</p>
<pre>x = 1
y = 2</pre>
<p>Noch etwas Text, damit der Hauptteil lang genug ist, um als eigentlicher Inhalt der Seite zu gelten.</p>
</main>
<footer>Impressum und Fußzeile</footer>
</body></html>"""


class Site:
    """Ein kleiner HTTP-Server auf 127.0.0.1 mit festen Antworten je Pfad."""

    def __init__(self) -> None:
        self.routes: dict[str, tuple[int, dict[str, str], bytes]] = {}
        self.requests: list[tuple[str, dict[str, str]]] = []
        site = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                site.requests.append((self.path, dict(self.headers)))
                status, headers, body = site.routes.get(self.path, (404, {"Content-Type": "text/plain"}, b"fehlt"))
                self.send_response(status)
                for name, value in headers.items():
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_: object) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.base = f"http://127.0.0.1:{self.port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def add(
        self, path: str, body: str | bytes, content_type: str = "text/html; charset=utf-8", status: int = 200, **headers: str
    ) -> None:
        data = body.encode() if isinstance(body, str) else body
        self.routes[path] = (status, {"Content-Type": content_type, **headers}, data)


@pytest.fixture
def site():
    server = Site()
    yield server
    server.server.shutdown()
    server.server.server_close()


def allow_test_server(host: str, port: int) -> list[str]:
    """Wie resolve_public, nur der Testserver auf 127.0.0.1 ist erlaubt."""
    if host == "127.0.0.1":
        return ["127.0.0.1"]
    return web_fetch.resolve_public(host, port)


def fetch(url: str, **kwargs):
    return web_fetch.fetch_url(url, resolve=allow_test_server, **kwargs)


def test_html_page_becomes_readable_text(site):
    site.add("/seite", PAGE)

    text = fetch(f"{site.base}/seite")

    assert text.startswith(f"URL: {site.base}/seite\nTitel: Testseite Kuramoto\n\n")
    assert "Das Kuramoto-Modell beschreibt, wie sich gekoppelte Oszillatoren synchronisieren." in text
    assert "- Erster Punkt\n- Zweiter Punkt" in text
    assert "x = 1\ny = 2" in text
    for hidden in ("geheim", "color: red", "Archiv", "Impressum"):
        assert hidden not in text


def test_links_mode_lists_absolute_links(site):
    site.add("/seite", PAGE)

    text = fetch(f"{site.base}/seite", links=True)

    # Erst die Links aus dem Hauptinhalt, dann die übrigen.
    assert text.split("\n")[3:] == [
        f"- dem zweiten Artikel → {site.base}/artikel/2?x=1",
        "- einer externen Quelle → https://example.org/extern",
        f"- Start → {site.base}/start",
        f"- Archiv → {site.base}/archiv",
    ]
    assert "mailto:" not in text
    assert "#oben" not in text
    assert "Links (4 von 4):" in text
    assert "Kuramoto-Modell beschreibt" not in text


def test_json_is_returned_as_is(site):
    site.add("/api", json.dumps({"query": {"search": [{"title": "Kuramoto-Modell"}]}}), "application/json")

    text = fetch(f"{site.base}/api")

    assert json.loads(text.split("\n\n", 1)[1])["query"]["search"][0]["title"] == "Kuramoto-Modell"


def test_long_page_is_cut_with_note(site):
    site.add("/lang", "<p>" + "Wort " * 3000 + "</p>", "text/html")

    text = fetch(f"{site.base}/lang", max_chars=500)

    assert len(text) <= 500
    assert text.endswith("… gekürzt, die Seite hat 14999 Zeichen.")


def test_gzip_body_is_decoded(site):
    site.add("/gz", gzip.compress("<p>Komprimierter Inhalt</p>".encode()), "text/html", **{"Content-Encoding": "gzip"})

    assert "Komprimierter Inhalt" in fetch(f"{site.base}/gz")


def test_http_error_is_reported(site):
    with pytest.raises(web_fetch.FetchError, match="HTTP 404"):
        fetch(f"{site.base}/gibt-es-nicht")


def test_binary_content_is_refused(site):
    site.add("/bild.png", b"\x89PNG\r\n\x1a\n" + bytes(100), "image/png")

    with pytest.raises(web_fetch.FetchError, match="Kein Text, sondern image/png"):
        fetch(f"{site.base}/bild.png")


def test_relative_redirect_is_followed(site):
    site.add("/alt", "", status=301, Location="/seite")
    site.add("/seite", PAGE)

    assert fetch(f"{site.base}/alt").startswith(f"URL: {site.base}/seite\n")


@pytest.mark.parametrize(
    "target",
    [
        "http://192.168.1.1/admin",
        "http://host.docker.internal:1234/v1/models",
        "http://169.254.169.254/latest/meta-data",
    ],
)
def test_redirect_into_private_network_is_blocked(site, target):
    site.add("/weiter", "", status=302, Location=target)

    with pytest.raises(web_fetch.FetchError):
        fetch(f"{site.base}/weiter")


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/",
        "http://127.0.0.1/",
        "http://10.0.0.1/",
        "http://192.168.0.10/",
        "http://172.18.0.2:8812/mcp",
        "http://169.254.169.254/latest/meta-data",
        "http://100.100.100.100/",  # Tailscale
        "http://[::1]/",
        "http://0.0.0.0/",
    ],
)
def test_private_addresses_are_blocked(site, url):
    with pytest.raises(web_fetch.BlockedAddressError):
        web_fetch.fetch_url(url)
    assert site.requests == []


@pytest.mark.parametrize(
    "url", ["file:///etc/passwd", "ftp://example.org/datei", "example.org", "http://nutzer:pw@example.org/"]
)
def test_only_plain_http_urls_are_fetched(url):
    with pytest.raises(web_fetch.FetchError):
        web_fetch.fetch_url(url, resolve=lambda host, port: pytest.fail("darf nicht auflösen"))


def test_connection_goes_to_checked_address_not_to_the_name(site):
    """Gegen DNS-Rebinding: verbunden wird mit der geprüften IP, Host-Header und Pfad behalten den Namen."""
    site.add("/seite", PAGE)

    text = web_fetch.fetch_url(f"http://seite.example:{site.port}/seite", resolve=lambda host, port: ["127.0.0.1"])

    assert "Kuramoto-Modell" in text
    ((path, headers),) = site.requests
    assert path == "/seite"
    assert headers["Host"] == f"seite.example:{site.port}"
    assert headers["User-Agent"] == web_fetch.USER_AGENT
