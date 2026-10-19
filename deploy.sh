#!/usr/bin/env bash
# Synchronisiert den Quellcode per rsync auf einen Docker-Host und startet den
# Container dort neu. Standardziel: SSH-Host "macstudio", Verzeichnis ~/unikin.
# UNIKIN_DIR ist relativ zum Home-Verzeichnis des SSH-Benutzers oder absolut.
#
#   ./deploy.sh
#   UNIKIN_HOST=anderer-host UNIKIN_DIR=agents/unikin ./deploy.sh
#
# .env, memory/, logs/ und artifacts/ auf dem Ziel bleiben unangetastet.
set -euo pipefail

HOST="${UNIKIN_HOST:-macstudio}"
DIR="${UNIKIN_DIR:-unikin}"
SSH="ssh -o ConnectTimeout=15"

case "$DIR" in
  *[[:space:]]*) echo "UNIKIN_DIR darf keine Leerzeichen enthalten (openrsync kennt kein --protect-args): '$DIR'" >&2; exit 1 ;;
esac

cd "$(dirname "$0")"

# Der Python-Runner ist ein Git-Submodul; ohne ihn schlägt der Build auf dem Ziel fehl.
if [ ! -f runner/Dockerfile ]; then
  echo "runner/ ist leer: zuerst 'git submodule update --init' ausführen." >&2
  exit 1
fi

$SSH "$HOST" "mkdir -p '$DIR'"

rsync -a -e "$SSH" \
  --exclude .git --exclude .github --exclude .claude \
  --exclude .venv --exclude venv --exclude __pycache__ --exclude .pytest_cache \
  --exclude .env --exclude memory --exclude logs --exclude artifacts \
  ./ "$HOST:$DIR/"

$SSH "$HOST" "export PATH=\"\$PATH:/opt/homebrew/bin:/usr/local/bin\"; \
  cd '$DIR' \
  && { [ -f .env ] || { cp .env.example .env && echo 'Hinweis: .env aus .env.example angelegt'; }; } \
  && mkdir -p memory logs artifacts \
  && docker compose up -d --build \
  && docker compose ps"
