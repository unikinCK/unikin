from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import secrets
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import anyio
import pytest
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client

import passwords
import unikin

ROOT = Path(__file__).resolve().parent.parent
PASSWORD = "richtig-langes-passwort"
REDIRECT_URI = "http://127.0.0.1/callback"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


OPENER = urllib.request.build_opener(NoRedirect)


def request(method, url, *, form=None, json_body=None, headers=None):
    """Liefert (Status, Header, Text); Weiterleitungen werden nicht verfolgt."""
    data = None
    headers = dict(headers or {})
    if form is not None:
        data = urlencode(form).encode()
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    if json_body is not None:
        data = json.dumps(json_body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with OPENER.open(req, timeout=15) as response:
            return response.status, response.headers, response.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers, exc.read().decode()


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def server_env(config, base_url: str, users: str | None) -> dict[str, str]:
    env = {
        **os.environ,
        "MEMORY_FILE": str(config.memory_file),
        "LOG_FILE": str(config.log_file),
        "UNIKIN_MCP_PUBLIC_URL": base_url,
        "UNIKIN_OAUTH_ALLOWED_REDIRECT_HOSTS": "127.0.0.1",
    }
    env.pop("UNIKIN_OAUTH_USERS", None)
    if users is not None:
        env["UNIKIN_OAUTH_USERS"] = users
    return env


@pytest.fixture
def base_url(config, tmp_path):
    unikin.load_memory(config)
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    # Wenige Iterationen: Der Test soll nicht auf PBKDF2 warten.
    users = f"christoph:{passwords.hash_password(PASSWORD, iterations=1000)}"
    process = subprocess.Popen(
        [sys.executable, str(ROOT / "mcp_server.py"), "--http", "--host", "127.0.0.1", "--port", str(port)],
        env=server_env(config, url, users),
        cwd=tmp_path,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 15
        while True:
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                break
            except OSError:
                assert process.poll() is None, "HTTP-Server ist beim Start abgestürzt"
                assert time.monotonic() < deadline, "HTTP-Server ist nicht gestartet"
                time.sleep(0.1)
        yield url
    finally:
        process.terminate()
        process.wait(timeout=10)


def register(base_url, redirect_uri=REDIRECT_URI):
    return request(
        "POST",
        f"{base_url}/register",
        json_body={
            "redirect_uris": [redirect_uri],
            "token_endpoint_auth_method": "none",
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "client_name": "Testclient",
        },
    )


def open_login_page(base_url, client_id, verifier, resource=None):
    """Startet /authorize und liefert die Felder der Login-Seite (request, csrf)."""
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    query = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": REDIRECT_URI,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": "zustand-123",
        "resource": resource or f"{base_url}/mcp",
    }
    status, headers, _ = request("GET", f"{base_url}/authorize?{urlencode(query)}")
    assert status == 302
    location = headers["Location"]
    if not location.startswith(f"{base_url}/login"):
        return location, None
    status, _, page = request("GET", location)
    assert status == 200
    fields = dict(re.findall(r'name="(request|csrf)" value="([^"]+)"', page))
    return location, fields


def login(base_url, fields, password=PASSWORD, action="allow"):
    return request(
        "POST",
        f"{base_url}/login",
        form={**fields, "username": "christoph", "password": password, "action": action},
    )


def exchange_code(base_url, client_id, code, verifier):
    return request(
        "POST",
        f"{base_url}/token",
        form={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": REDIRECT_URI,
            "client_id": client_id,
            "code_verifier": verifier,
            "resource": f"{base_url}/mcp",
        },
    )


def mcp_status(base_url, access_token):
    return request(
        "POST",
        f"{base_url}/mcp",
        json_body={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}},
        },
        headers={"Accept": "application/json, text/event-stream", "Authorization": f"Bearer {access_token}"},
    )[0]


def test_discovery_and_unauthenticated_request(base_url):
    status, _, body = request("GET", f"{base_url}/.well-known/oauth-protected-resource/mcp")
    assert status == 200
    resource = json.loads(body)
    assert resource["resource"] == f"{base_url}/mcp"
    assert [server.rstrip("/") for server in resource["authorization_servers"]] == [base_url]

    status, _, body = request("GET", f"{base_url}/.well-known/oauth-authorization-server")
    metadata = json.loads(body)
    assert metadata["registration_endpoint"] == f"{base_url}/register"
    assert metadata["code_challenge_methods_supported"] == ["S256"]

    status, headers, _ = request("POST", f"{base_url}/mcp", json_body={})
    assert status == 401
    assert "resource_metadata" in headers["WWW-Authenticate"]


def test_full_authorization_code_flow(base_url, config):
    status, _, body = register(base_url)
    assert status == 201
    client_id = json.loads(body)["client_id"]
    verifier = secrets.token_urlsafe(48)
    _, fields = open_login_page(base_url, client_id, verifier)

    status, headers, page = login(base_url, fields, password="falsch")
    assert status == 401
    assert "falsch" in page and "Location" not in headers

    status, headers, _ = login(base_url, fields)
    assert status == 302
    redirect = urlparse(headers["Location"])
    params = parse_qs(redirect.query)
    assert f"{redirect.scheme}://{redirect.netloc}{redirect.path}" == REDIRECT_URI
    assert params["state"] == ["zustand-123"]
    code = params["code"][0]

    status, _, body = exchange_code(base_url, client_id, code, verifier)
    assert status == 200
    tokens = json.loads(body)
    assert tokens["token_type"] == "Bearer" and tokens["scope"] == "unikin"
    assert exchange_code(base_url, client_id, code, verifier)[0] == 400

    async def use_tools():
        http_client = create_mcp_http_client(headers={"Authorization": f"Bearer {tokens['access_token']}"})
        async with Client(streamable_http_client(f"{base_url}/mcp", http_client=http_client)) as client:
            names = {tool.name for tool in (await client.list_tools()).tools}
            await client.call_tool("send_message", {"text": "Hallo per OAuth", "sender": "Test"})
            return names

    assert anyio.run(use_tools) == {"get_state", "get_journal", "search_journal", "send_message", "step"}
    assert [message["text"] for _, message in unikin.read_inbox(config)] == ["Hallo per OAuth"]

    status, _, body = request(
        "POST",
        f"{base_url}/token",
        form={"grant_type": "refresh_token", "refresh_token": tokens["refresh_token"], "client_id": client_id},
    )
    assert status == 200
    refreshed = json.loads(body)
    assert mcp_status(base_url, refreshed["access_token"]) == 200
    assert mcp_status(base_url, tokens["access_token"]) == 401
    status, _, _ = request(
        "POST",
        f"{base_url}/token",
        form={"grant_type": "refresh_token", "refresh_token": tokens["refresh_token"], "client_id": client_id},
    )
    assert status == 400


def test_deny_redirects_with_access_denied(base_url):
    client_id = json.loads(register(base_url)[2])["client_id"]
    _, fields = open_login_page(base_url, client_id, secrets.token_urlsafe(48))

    status, headers, _ = login(base_url, fields, action="deny")

    assert status == 302
    params = parse_qs(urlparse(headers["Location"]).query)
    assert params["error"] == ["access_denied"] and params["state"] == ["zustand-123"]


def test_token_for_foreign_resource_is_refused(base_url):
    client_id = json.loads(register(base_url)[2])["client_id"]

    location, fields = open_login_page(base_url, client_id, secrets.token_urlsafe(48), resource="https://evil.example/mcp")

    assert fields is None
    assert parse_qs(urlparse(location).query)["error"] == ["invalid_target"]


def test_registration_rejects_foreign_redirect_host(base_url):
    status, _, body = register(base_url, redirect_uri="https://evil.example/callback")

    assert status == 400
    assert json.loads(body)["error"] == "invalid_redirect_uri"


def test_http_refuses_to_start_without_accounts(config, tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "mcp_server.py"), "--http", "--port", str(free_port())],
        env=server_env(config, "http://127.0.0.1:1", None),
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode != 0
    assert "UNIKIN_OAUTH_USERS" in result.stderr


def test_password_hashes():
    encoded = passwords.hash_password("geheim-und-lang", iterations=1000)

    assert "$" not in encoded
    assert passwords.verify_password("geheim-und-lang", encoded)
    assert not passwords.verify_password("falsch", encoded)
    assert passwords.parse_users(f"christoph:{encoded}") == {"christoph": encoded}
    with pytest.raises(ValueError):
        passwords.parse_users("christoph:klartext-passwort")
