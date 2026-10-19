#!/usr/bin/env python3
"""Passwort-Hashes für die OAuth-Konten des MCP-Servers (UNIKIN_OAUTH_USERS).

Nur Standardbibliothek, damit `python3 passwords.py` überall läuft, auch ohne die
Abhängigkeiten des Projekts. Format: pbkdf2-sha256.<iterationen>.<salt>.<hash> (Base64url),
ohne "$", weil Docker Compose ein "$" in der .env als Variable auswerten würde.
"""

from __future__ import annotations

import base64
import getpass
import hashlib
import hmac
import secrets
import sys

PASSWORD_HASH_PREFIX = "pbkdf2-sha256"
PBKDF2_ITERATIONS = 600_000
MIN_PASSWORD_CHARS = 12


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def hash_password(password: str, iterations: int = PBKDF2_ITERATIONS) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return f"{PASSWORD_HASH_PREFIX}.{iterations}.{_b64(salt)}.{_b64(digest)}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        prefix, iterations, salt, digest = encoded.split(".")
        if prefix != PASSWORD_HASH_PREFIX:
            return False
        candidate = hashlib.pbkdf2_hmac("sha256", password.encode(), _unb64(salt), int(iterations))
        return hmac.compare_digest(candidate, _unb64(digest))
    except (ValueError, TypeError):
        return False


def parse_users(value: str) -> dict[str, str]:
    """Liest UNIKIN_OAUTH_USERS: name:hash, mehrere durch Komma getrennt."""
    users: dict[str, str] = {}
    for entry in value.split(","):
        entry = entry.strip()
        if not entry:
            continue
        name, separator, encoded = entry.partition(":")
        if not separator or not name.strip() or not encoded.startswith(PASSWORD_HASH_PREFIX + "."):
            raise ValueError(f"Ungültiger Eintrag in UNIKIN_OAUTH_USERS: {name.strip() or entry[:20]!r}")
        users[name.strip()] = encoded.strip()
    return users


def main() -> None:
    password = getpass.getpass("Neues Passwort: ")
    if len(password) < MIN_PASSWORD_CHARS:
        sys.exit(f"Das Passwort braucht mindestens {MIN_PASSWORD_CHARS} Zeichen.")
    if getpass.getpass("Wiederholen: ") != password:
        sys.exit("Die Passwörter stimmen nicht überein.")
    print(hash_password(password))


if __name__ == "__main__":
    main()
