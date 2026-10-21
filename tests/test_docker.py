from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_every_file_the_image_copies_is_in_the_build_context():
    """.dockerignore ist eine Allowlist: Was dort fehlt, findet COPY beim Build nicht."""
    allowed = {line[1:] for line in (ROOT / ".dockerignore").read_text().splitlines() if line.startswith("!")}
    copied = [
        source
        for line in (ROOT / "Dockerfile").read_text().splitlines()
        if line.startswith("COPY ")
        for source in line.split()[1:-1]
    ]
    assert copied
    assert [source for source in copied if source not in allowed] == []
