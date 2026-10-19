"""OAuth 2.1 für Unikins MCP-Server über HTTP: eigener Autorisierungsserver mit Login-Seite.

Folgt dem Muster von mcp_imap und OpenBuchhaltung: dynamische Client-Registrierung, PKCE (S256),
undurchsichtige Tokens, von denen nur der SHA-256-Hash gespeichert wird, und rotierende
Refresh-Tokens. Die Endpunkte /authorize, /token, /register und /revoke sowie die Discovery
liefert das MCP-SDK; hier stehen Speicher, Provider und die Login- und Zustimmungsseite.
Konten stehen mit PBKDF2-Hash in UNIKIN_OAUTH_USERS (siehe passwords.py).
"""

from __future__ import annotations

import hashlib
import hmac
import html
import json
import logging
import secrets
import sqlite3
import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import anyio
from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    AuthorizeError,
    RefreshToken,
    RegistrationError,
    TokenError,
    construct_redirect_uri,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse, Response

from passwords import hash_password, verify_password

SCOPE = "unikin"
ACCESS_TOKEN_SECONDS = 3600
REFRESH_TOKEN_SECONDS = 30 * 24 * 3600
AUTH_CODE_SECONDS = 600
LOGIN_REQUEST_SECONDS = 600
MAX_FAILED_LOGINS = 10
FAILED_LOGIN_WINDOW_SECONDS = 900
LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS clients (client_id TEXT PRIMARY KEY, info TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS login_requests (
    id TEXT PRIMARY KEY, client_id TEXT NOT NULL, params TEXT NOT NULL, csrf TEXT NOT NULL, expires_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS codes (code_hash TEXT PRIMARY KEY, data TEXT NOT NULL, expires_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS tokens (
    token_hash TEXT PRIMARY KEY, kind TEXT NOT NULL, family TEXT NOT NULL, client_id TEXT NOT NULL,
    scopes TEXT NOT NULL, resource TEXT, subject TEXT, expires_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS tokens_family ON tokens (family);
"""


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def redirect_uri_error(uri: str, allowed_hosts: set[str]) -> str | None:
    """Prüft eine Redirect-URI bei der Registrierung; liefert den Grund oder None."""
    parsed = urlparse(uri)
    host = (parsed.hostname or "").lower()
    if parsed.fragment:
        return "Redirect-URI darf kein Fragment enthalten."
    if parsed.scheme != "https" and not (parsed.scheme == "http" and host in LOOPBACK_HOSTS):
        return "Redirect-URI muss HTTPS verwenden (HTTP nur auf localhost)."
    if host not in allowed_hosts:
        return f"Redirect-Host {host!r} ist nicht freigegeben."
    return None


class OAuthStore:
    """Clients, Login-Anfragen, Codes und Tokens in SQLite; Tokens und Codes nur als Hash."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.executescript(SCHEMA)
        path.chmod(0o600)

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            conn = sqlite3.connect(self.path, timeout=10)
            conn.row_factory = sqlite3.Row
            try:
                with conn:
                    yield conn
            finally:
                conn.close()

    def _purge(self, db: sqlite3.Connection) -> None:
        now = time.time()
        db.execute("DELETE FROM login_requests WHERE expires_at < ?", (now,))
        db.execute("DELETE FROM codes WHERE expires_at < ?", (now,))
        db.execute("DELETE FROM tokens WHERE expires_at < ?", (now,))

    def get_client(self, client_id: str) -> str | None:
        with self._db() as db:
            row = db.execute("SELECT info FROM clients WHERE client_id = ?", (client_id,)).fetchone()
        return row["info"] if row else None

    def save_client(self, client_id: str, info: str) -> None:
        with self._db() as db:
            db.execute("INSERT OR REPLACE INTO clients (client_id, info) VALUES (?, ?)", (client_id, info))

    def save_login_request(self, request_id: str, client_id: str, params: str, csrf: str) -> None:
        with self._db() as db:
            self._purge(db)
            db.execute(
                "INSERT INTO login_requests (id, client_id, params, csrf, expires_at) VALUES (?, ?, ?, ?, ?)",
                (request_id, client_id, params, csrf, time.time() + LOGIN_REQUEST_SECONDS),
            )

    def get_login_request(self, request_id: str) -> sqlite3.Row | None:
        with self._db() as db:
            return db.execute(
                "SELECT * FROM login_requests WHERE id = ? AND expires_at >= ?", (request_id, time.time())
            ).fetchone()

    def delete_login_request(self, request_id: str) -> bool:
        with self._db() as db:
            return db.execute("DELETE FROM login_requests WHERE id = ?", (request_id,)).rowcount == 1

    def save_code(self, code: str, data: str, expires_at: float) -> None:
        with self._db() as db:
            self._purge(db)
            db.execute(
                "INSERT INTO codes (code_hash, data, expires_at) VALUES (?, ?, ?)", (token_hash(code), data, expires_at)
            )

    def get_code(self, code: str) -> str | None:
        with self._db() as db:
            row = db.execute("SELECT data FROM codes WHERE code_hash = ?", (token_hash(code),)).fetchone()
        return row["data"] if row else None

    def consume_code(self, code: str) -> bool:
        with self._db() as db:
            return db.execute("DELETE FROM codes WHERE code_hash = ?", (token_hash(code),)).rowcount == 1

    def save_token_pair(
        self, access: str, refresh: str, client_id: str, scopes: list[str], resource: str | None, subject: str | None
    ) -> None:
        family = secrets.token_urlsafe(16)
        now = int(time.time())
        with self._db() as db:
            self._purge(db)
            for token, kind, lifetime in ((access, "access", ACCESS_TOKEN_SECONDS), (refresh, "refresh", REFRESH_TOKEN_SECONDS)):
                db.execute(
                    "INSERT INTO tokens (token_hash, kind, family, client_id, scopes, resource, subject, expires_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (token_hash(token), kind, family, client_id, " ".join(scopes), resource, subject, now + lifetime),
                )

    def get_token(self, token: str, kind: str) -> sqlite3.Row | None:
        with self._db() as db:
            return db.execute(
                "SELECT * FROM tokens WHERE token_hash = ? AND kind = ? AND expires_at >= ?",
                (token_hash(token), kind, int(time.time())),
            ).fetchone()

    def revoke_family(self, token: str) -> bool:
        """Löscht Access- und Refresh-Token einer Ausgabe; True, wenn es sie noch gab."""
        with self._db() as db:
            row = db.execute("SELECT family FROM tokens WHERE token_hash = ?", (token_hash(token),)).fetchone()
            if row is None:
                return False
            db.execute("DELETE FROM tokens WHERE family = ?", (row["family"],))
            return True


class UnikinOAuthProvider:
    """Autorisierungsserver für das MCP-SDK (OAuthAuthorizationServerProvider)."""

    def __init__(self, store: OAuthStore, public_url: str, allowed_redirect_hosts: set[str]) -> None:
        self.store = store
        self.public_url = public_url.rstrip("/")
        self.resource_url = f"{self.public_url}/mcp"
        self.allowed_redirect_hosts = {host.lower() for host in allowed_redirect_hosts}

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        info = self.store.get_client(client_id)
        return OAuthClientInformationFull.model_validate_json(info) if info else None

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        for uri in client_info.redirect_uris or []:
            error = redirect_uri_error(str(uri), self.allowed_redirect_hosts)
            if error:
                raise RegistrationError("invalid_redirect_uri", error)
        self.store.save_client(client_info.client_id, client_info.model_dump_json())
        logging.info("OAuth-Client registriert: %s (%s)", client_info.client_name, client_info.client_id)

    async def authorize(self, client: OAuthClientInformationFull, params: AuthorizationParams) -> str:
        # Tokens gelten nur für diesen Server; fehlt resource (RFC 8707), wird er eingesetzt.
        resource = params.resource or self.resource_url
        if resource.rstrip("/") != self.resource_url:
            raise AuthorizeError("invalid_target", "Dieser Server vergibt nur Tokens für seinen eigenen /mcp-Endpunkt.")
        request_id = secrets.token_urlsafe(32)
        params = params.model_copy(update={"resource": self.resource_url, "scopes": params.scopes or [SCOPE]})
        self.store.save_login_request(request_id, client.client_id, params.model_dump_json(), secrets.token_urlsafe(32))
        return f"{self.public_url}/login?request={request_id}"

    async def load_authorization_code(self, client: OAuthClientInformationFull, authorization_code: str) -> AuthorizationCode | None:
        data = self.store.get_code(authorization_code)
        return AuthorizationCode.model_validate({**json.loads(data), "code": authorization_code}) if data else None

    async def exchange_authorization_code(self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode) -> OAuthToken:
        if not self.store.consume_code(authorization_code.code):
            raise TokenError("invalid_grant", "Der Code wurde bereits eingelöst.")
        return self._issue(client.client_id, authorization_code.scopes, authorization_code.resource, authorization_code.subject)

    async def load_refresh_token(self, client: OAuthClientInformationFull, refresh_token: str) -> RefreshToken | None:
        row = self.store.get_token(refresh_token, "refresh")
        if row is None:
            return None
        return RefreshToken(
            token=refresh_token,
            client_id=row["client_id"],
            scopes=row["scopes"].split(),
            expires_at=row["expires_at"],
            resource=row["resource"],
            subject=row["subject"],
        )

    async def exchange_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: RefreshToken, scopes: list[str]
    ) -> OAuthToken:
        # Rotation: Der alte Refresh-Token und sein Access-Token werden ungültig.
        if not self.store.revoke_family(refresh_token.token):
            raise TokenError("invalid_grant", "Der Refresh-Token ist ungültig.")
        return self._issue(client.client_id, scopes, refresh_token.resource, refresh_token.subject)

    async def load_access_token(self, token: str) -> AccessToken | None:
        row = self.store.get_token(token, "access")
        if row is None:
            return None
        return AccessToken(
            token=token,
            client_id=row["client_id"],
            scopes=row["scopes"].split(),
            expires_at=row["expires_at"],
            resource=row["resource"],
            subject=row["subject"],
        )

    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        self.store.revoke_family(token.token)

    def _issue(self, client_id: str, scopes: list[str], resource: str | None, subject: str | None) -> OAuthToken:
        access = "uka_" + secrets.token_urlsafe(32)
        refresh = "ukr_" + secrets.token_urlsafe(32)
        self.store.save_token_pair(access, refresh, client_id, scopes, resource, subject)
        return OAuthToken(
            access_token=access,
            token_type="Bearer",
            expires_in=ACCESS_TOKEN_SECONDS,
            scope=" ".join(scopes),
            refresh_token=refresh,
        )


class FailedLogins:
    """Begrenzt Fehlversuche: Nach MAX_FAILED_LOGINS im Zeitfenster ist der Login gesperrt."""

    def __init__(self) -> None:
        self._failures: deque[float] = deque()

    def _prune(self) -> None:
        cutoff = time.monotonic() - FAILED_LOGIN_WINDOW_SECONDS
        while self._failures and self._failures[0] < cutoff:
            self._failures.popleft()

    def blocked(self) -> bool:
        self._prune()
        return len(self._failures) >= MAX_FAILED_LOGINS

    def record(self) -> None:
        self._failures.append(time.monotonic())


PAGE_HEADERS = {
    "Cache-Control": "no-store",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; frame-ancestors 'none'",
}

PAGE_TEMPLATE = """<!doctype html>
<html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Unikin – Anmelden</title>
<style>
body {{ font-family: system-ui, sans-serif; background: #f4f4f5; color: #18181b; margin: 0; padding: 16px; }}
main {{ max-width: 420px; margin: 8vh auto; background: #fff; border-radius: 12px; padding: 28px; box-shadow: 0 1px 4px #0002; }}
h1 {{ font-size: 1.3rem; margin: 0 0 12px; }}
p {{ line-height: 1.45; }}
label {{ display: block; margin: 14px 0 4px; font-weight: 600; }}
input {{ width: 100%; box-sizing: border-box; padding: 9px; border: 1px solid #a1a1aa; border-radius: 6px; font-size: 1rem; }}
.buttons {{ display: flex; gap: 10px; margin-top: 20px; }}
button {{ flex: 1; padding: 10px; border-radius: 6px; border: 0; font-size: 1rem; cursor: pointer; }}
.allow {{ background: #18181b; color: #fff; }} .deny {{ background: #e4e4e7; }}
.error {{ background: #fee2e2; color: #991b1b; padding: 8px 10px; border-radius: 6px; }}
@media (prefers-color-scheme: dark) {{
  body {{ background: #18181b; color: #f4f4f5; }} main {{ background: #27272a; }}
  input {{ background: #18181b; color: #f4f4f5; border-color: #52525b; }}
  .allow {{ background: #f4f4f5; color: #18181b; }} .deny {{ background: #3f3f46; color: #f4f4f5; }}
}}
</style></head><body><main>{body}</main></body></html>"""


def render_page(body: str, status_code: int = 200) -> HTMLResponse:
    return HTMLResponse(PAGE_TEMPLATE.format(body=body), status_code=status_code, headers=PAGE_HEADERS)


def render_message(text: str, status_code: int) -> HTMLResponse:
    return render_page(f"<h1>Unikin</h1><p>{html.escape(text)}</p>", status_code)


class LoginPage:
    """Login und Zustimmung in einem Schritt; danach geht es mit Code zurück zum Client."""

    def __init__(self, provider: UnikinOAuthProvider, users: dict[str, str]) -> None:
        self.provider = provider
        self.users = users
        self.failures = FailedLogins()
        # Für unbekannte Namen: gleich teure Prüfung, damit die Antwortzeit nicht verrät, welche Konten es gibt.
        self.dummy_hash = hash_password(secrets.token_urlsafe(16))

    async def handle(self, request: Request) -> Response:
        if request.method == "POST":
            return await self._submit(request)
        login = self.provider.store.get_login_request(request.query_params.get("request", ""))
        if login is None:
            return render_message("Diese Anmeldung ist abgelaufen oder unbekannt. Bitte im Client neu verbinden.", 400)
        return await self._form(login)

    async def _form(self, login: sqlite3.Row, error: str = "", status_code: int = 200) -> HTMLResponse:
        client = await self.provider.get_client(login["client_id"])
        params = AuthorizationParams.model_validate_json(login["params"])
        client_name = (client.client_name if client else None) or "Ein MCP-Client"
        target = urlparse(str(params.redirect_uri)).hostname or "?"
        error_html = f'<p class="error">{html.escape(error)}</p>' if error else ""
        body = f"""<h1>Zugriff auf Unikin</h1>
<p><strong>{html.escape(client_name)}</strong> ({html.escape(target)}) möchte Unikins Gedächtnis lesen,
ihm Nachrichten schicken und Zyklen anstoßen.</p>
{error_html}
<form method="post" action="/login">
<input type="hidden" name="request" value="{html.escape(login["id"])}">
<input type="hidden" name="csrf" value="{html.escape(login["csrf"])}">
<label for="username">Benutzername</label><input id="username" name="username" autocomplete="username" required>
<label for="password">Passwort</label>
<input id="password" name="password" type="password" autocomplete="current-password" required>
<div class="buttons"><button class="allow" name="action" value="allow">Anmelden und erlauben</button>
<button class="deny" name="action" value="deny" formnovalidate>Ablehnen</button></div>
</form>"""
        return render_page(body, status_code)

    async def _submit(self, request: Request) -> Response:
        form = await request.form()
        login = self.provider.store.get_login_request(str(form.get("request", "")))
        if login is None or not hmac.compare_digest(str(form.get("csrf", "")), login["csrf"]):
            return render_message("Diese Anmeldung ist abgelaufen oder ungültig. Bitte im Client neu verbinden.", 400)
        params = AuthorizationParams.model_validate_json(login["params"])

        if form.get("action") == "deny":
            self.provider.store.delete_login_request(login["id"])
            return RedirectResponse(
                construct_redirect_uri(str(params.redirect_uri), error="access_denied", state=params.state), status_code=302
            )

        if self.failures.blocked():
            return render_message("Zu viele Fehlversuche. Bitte später erneut versuchen.", 429)
        username = str(form.get("username", "")).strip()
        password = str(form.get("password", ""))
        # PBKDF2 dauert absichtlich lange: nicht im Event-Loop rechnen.
        valid = await anyio.to_thread.run_sync(verify_password, password, self.users.get(username, self.dummy_hash))
        if not valid or username not in self.users:
            self.failures.record()
            logging.warning("OAuth-Login fehlgeschlagen für %r", username[:80])
            return await self._form(login, "Benutzername oder Passwort ist falsch.", 401)

        if not self.provider.store.delete_login_request(login["id"]):
            return render_message("Diese Anmeldung wurde bereits verwendet. Bitte im Client neu verbinden.", 400)
        code = secrets.token_urlsafe(32)
        expires_at = time.time() + AUTH_CODE_SECONDS
        data: dict[str, Any] = {
            "scopes": params.scopes or [SCOPE],
            "expires_at": expires_at,
            "client_id": login["client_id"],
            "code_challenge": params.code_challenge,
            "redirect_uri": str(params.redirect_uri),
            "redirect_uri_provided_explicitly": params.redirect_uri_provided_explicitly,
            "resource": params.resource,
            "subject": username,
        }
        self.provider.store.save_code(code, json.dumps(data), expires_at)
        logging.info("OAuth-Login: %s erlaubt Zugriff für Client %s", username, login["client_id"])
        return RedirectResponse(
            construct_redirect_uri(str(params.redirect_uri), code=code, state=params.state), status_code=302
        )
