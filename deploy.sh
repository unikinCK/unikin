#!/usr/bin/env bash
# Synchronisiert den Quellcode per rsync auf einen Docker-Host und startet den
# Container dort neu. Standardziel: SSH-Host "macstudio", Verzeichnis ~/unikin.
#
#   ./deploy.sh
#   UNIKIN_HOST=anderer-host UNIKIN_DIR=/opt/unikin ./deploy.sh
#
# .env, memory/ und logs/ auf dem Ziel bleiben unangetastet.
set -euo pipefail

HOST="${UNIKIN_HOST:-macstudio}"
DIR="${UNIKIN_DIR:-unikin}"

cd "$(dirname "$0")"

rsync -a \
  --exclude .git --exclude .github --exclude .claude \
  --exclude .venv --exclude venv --exclude __pycache__ \
  --exclude .env --exclude memory --exclude logs \
  ./ "$HOST:$DIR/"

ssh "$HOST" "export PATH=\"\$PATH:/opt/homebrew/bin:/usr/local/bin\"; \
  cd '$DIR' \
  && { [ -f .env ] || { cp .env.example .env && echo 'Hinweis: .env aus .env.example angelegt'; }; } \
  && mkdir -p memory logs \
  && docker compose up -d --build \
  && docker compose ps"
