from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_every_file_the_image_copies_is_in_the_build_context():
    # .dockerignore ist eine Allowlist: Fehlt dort eine Datei, scheitert der Build erst auf dem Ziel.
    allowed = {
        line[1:].strip()
        for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.startswith("!")
    }
    copied = {
        name
        for match in re.finditer(r"^COPY (.+) \./$", (ROOT / "Dockerfile").read_text(encoding="utf-8"), re.MULTILINE)
        for name in match.group(1).split()
    }

    assert copied
    assert copied <= allowed
