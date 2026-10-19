# unikin

`unikin` ist ein autonomer Python-Agent, der dauerhaft (`while True`) über eine lokal laufende OpenAI-kompatible LLM-API arbeitet.

## Features

- Echte Endlosschleife ohne fachliche Abbruchbedingung
- Konfigurierbare Identität (`Unikin`) und Mission
- Persistentes Memory in JSON (`memory/unikin_state.json`)
- Robustes Logging auf Konsole und in Datei (`logs/unikin.log`)
- Strukturierte Modellantwort mit JSON-Feldern:
  - `thought`
  - `state_update`
  - `next_focus`
  - `urgency` (`low|medium|high`)
- Fallback-Parsing bei fehlerhaftem JSON
- Adaptive Drosselung zwischen `MIN_INTERVAL_SECONDS` und `MAX_INTERVAL_SECONDS`

## Voraussetzungen

- Python 3.10+
- Eine lokale OpenAI-kompatible Instanz, z. B.:
  - LM Studio (OpenAI API kompatibler Server)
  - Ollama via kompatiblem Gateway
  - vLLM mit OpenAI-kompatiblem Endpoint

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Passe danach die `.env` an dein lokales Setup an.

## Beispiel `.env`

```dotenv
OPENAI_BASE_URL=http://localhost:1234/v1
OPENAI_API_KEY=local-key
OPENAI_MODEL=local-model
OPENAI_TIMEOUT_SECONDS=120

UNIKIN_IDENTITY=Unikin
UNIKIN_MISSION=Kontinuierliche reflektierte Arbeit in kleinen Schritten.

MIN_INTERVAL_SECONDS=5
MAX_INTERVAL_SECONDS=300
DEFAULT_INTERVAL_SECONDS=30

MEMORY_FILE=memory/unikin_state.json
LOG_FILE=logs/unikin.log
```


## MCP Server & MCP Client

Zusätzlich enthält `unikin` jetzt eine MCP-Integration:

- `mcp_server.py`: Ein lokaler MCP-Server (JSON-RPC über STDIN/STDOUT) mit den Tools:
  - `unikin.get_state`
  - `unikin.step`
- `mcp_client.py`: Ein einfacher CLI-Client, der den Server startet und Tools aufruft.

### MCP-Server starten

```bash
python mcp_server.py
```

### MCP-Client nutzen

```bash
# Tool-Liste
python mcp_client.py tools

# Aktuellen Zustand lesen
python mcp_client.py state

# Einen autonomen Schritt ausführen
python mcp_client.py step --note "manueller MCP-Trigger"
```

## Start

```bash
python unikin.py
```

## Docker (Mac Studio + LM Studio)

Der Agent läuft als Container, LM Studio bleibt nativ auf dem Host (Metal/MLX) und wird aus dem Container über `host.docker.internal:1234` angesprochen. Getestet auf dem Mac Studio mit Colima; mit Docker Desktop funktioniert es genauso.

### Voraussetzungen auf dem Host

- Docker mit Compose v2 (Colima oder Docker Desktop)
- LM Studio mit laufendem Server auf Port 1234 (`lms server status`). Unter Colima muss der Server auf allen Interfaces lauschen („Serve on Local Network“ bzw. `networkInterface: 0.0.0.0`), weil Container den Host über die VM-Brücke erreichen; unter Docker Desktop reicht auch `127.0.0.1`.
- Das Modell aus `OPENAI_MODEL` ist geladen (`lms ps`) oder Just-in-Time-Laden ist in LM Studio aktiv.

### Starten

```bash
cp .env.example .env          # OPENAI_MODEL anpassen; OPENAI_BASE_URL darf auf localhost bleiben
docker compose up -d --build
docker compose logs -f
```

- Die Basis-URL setzt `docker-compose.yml` für den Container selbst auf `http://host.docker.internal:1234/v1`; `OPENAI_BASE_URL` in `.env` gilt nur für den nativen Start. Läuft LM Studio auf einem anderen Rechner, `LMSTUDIO_URL=http://<host>:1234/v1` in `.env` setzen.
- `memory/` und `logs/` sind Bind-Mounts im Projektverzeichnis und überleben Neustarts, Rebuilds und `docker compose down`. `logs/unikin.log` rotiert bei 10 MB (fünf Generationen).
- Jeder Modellaufruf hat ein Timeout von `OPENAI_TIMEOUT_SECONDS` (Standard 120 s, danach bis zu drei Versuche). `stop_grace_period` (150 s) ist darauf abgestimmt, damit `docker compose stop` einen laufenden Aufruf zu Ende kommen lässt, statt den Agenten hart zu beenden.
- `restart: unless-stopped`: Der Container kommt nach einem Docker-Neustart von selbst wieder, solange er nicht mit `docker compose stop` angehalten wurde.

### Stoppen, Modell wechseln

```bash
docker compose stop     # SIGTERM; in der Wartephase endet der Agent innerhalb einer Sekunde
docker compose down     # Container entfernen; memory/ und logs/ bleiben erhalten
```

Modell wechseln: `OPENAI_MODEL` in `.env` ändern und `docker compose up -d` ausführen (kein Rebuild nötig). Verfügbare IDs liefert `curl http://localhost:1234/v1/models`. Bei Just-in-Time-Laden bleibt das alte Modell bis zum Ablauf seiner TTL im Speicher; wer den Platz sofort braucht, entlädt es gezielt mit `lms unload <alte-modell-id>` (nicht `--all`, falls andere Clients LM Studio nutzen).

### Dauerbetrieb auf dem Mac

Docker (Colima bzw. Docker Desktop) und LM Studio laufen in der Benutzersitzung. Nach einem Neustart des Macs sind sie nur dann wieder da, wenn der Benutzer automatisch angemeldet wird, der Mac nicht schläft, Colima als Dienst startet (`brew services start colima`) und LM Studio den Server beim App-Start aktiviert (Developer-Bereich, `autoStartOnLaunch`). Bis LM Studio erreichbar ist, protokolliert der Agent Verbindungsfehler und wartet mit wachsendem Intervall (bis `MAX_INTERVAL_SECONDS`).

### Deployment per rsync (z. B. vom Laptop auf den Mac Studio)

```bash
./deploy.sh                                       # Ziel: SSH-Host "macstudio", Verzeichnis ~/unikin
UNIKIN_HOST=anderer-host UNIKIN_DIR=/opt/unikin ./deploy.sh
```

Das Skript synchronisiert den Quellcode (ohne `.git`, `.env`, `memory/`, `logs/`) und führt auf dem Ziel `docker compose up -d --build` aus. Eine vorhandene `.env` auf dem Ziel bleibt unangetastet; fehlt sie, wird sie aus `.env.example` angelegt.

### MCP-Server im Container

```bash
# Einmal-Container für den MCP-Server (unabhängig vom laufenden Loop-Container)
python mcp_client.py --server-cmd "docker compose run --rm -T unikin python mcp_server.py" state
python mcp_client.py --server-cmd "docker compose run --rm -T unikin python mcp_server.py" step --note "manueller MCP-Trigger"
```

Hinweis: MCP-Server und Loop-Container teilen sich `memory/unikin_state.json`. Ein `unikin.step` parallel zum laufenden Loop kann sich mit dessen Schreibvorgang überschneiden (letzter Schreiber gewinnt).

### Fehlersuche

- `Connection error` im Log: Der LM-Studio-Server läuft nicht (`lms server start`) oder lauscht unter Colima nur auf `127.0.0.1`.
- `Request timed out`: Ein Aufruf hat länger als `OPENAI_TIMEOUT_SECONDS` gedauert, z. B. weil LM Studio ein großes Modell per JIT lädt. Der Agent wiederholt bis zu dreimal; bei sehr großen Modellen das Timeout erhöhen und `stop_grace_period` in `docker-compose.yml` mitziehen.
- Reasoning-Modelle (z. B. Qwen 3.6, Nemotron) liefern das JSON im `message`-Teil der Antwort; der Denkteil wird nicht ausgewertet.

## Adaptive Drosselung

Der Agent passt das nächste Abfrageintervall autonom an. Signale:

1. **`urgency` aus dem Modell**
   - `high` → kürzeres Intervall
   - `low` → längeres Intervall
2. **Fehlerhäufigkeit / aufeinanderfolgende Fehler**
   - exponentieller Backoff
3. **Gedankliche Wiederholung**
   - einfache Ähnlichkeitsheuristik (`difflib.SequenceMatcher`) zwischen letzter und aktueller `thought`
   - bei wiederholter Ähnlichkeit wird das Intervall schrittweise erhöht
4. **Antwortzeit (optional)**
   - sehr langsame Antworten erhöhen das Intervall leicht

Zusätzlich wird ein kleiner Jitter genutzt, um starre Polling-Muster zu vermeiden.

## Persistenz & Logs

- Memory wird pro Zyklus geladen/aktualisiert/gespeichert.
- Bei fehlender oder defekter Memory-Datei wird sauber reinitialisiert.
- Logs landen sowohl in der Konsole als auch in der Datei `logs/unikin.log`.
- Ordner werden automatisch angelegt.

## Robustheit

- Normale Laufzeitfehler beenden den Agenten nicht dauerhaft.
- API-/JSON-/Netzwerkprobleme werden mit Retry und Fallback behandelt.
- `KeyboardInterrupt`/`SIGTERM` werden kontrolliert verarbeitet.

## Wichtiger Hinweis

Der Agent besitzt **keine automatische fachliche Abbruchbedingung** (kein „done“, kein „completed“, kein Max-Steps-Cutoff). Er läuft kontinuierlich, bis du ihn manuell stoppst.
