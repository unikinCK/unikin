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
- Inbox für Nachrichten von außen: Eine neue Nachricht weckt den Loop sofort, Unikin antwortet im Feld `reply`
- Standardkonformer MCP-Server (offizielles MCP-Python-SDK) zum Lesen des Zustands und Senden von Nachrichten

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


## MCP-Server & MCP-Client

`mcp_server.py` ist ein MCP-Server nach Spezifikation auf Basis des offiziellen [MCP-Python-SDK](https://github.com/modelcontextprotocol/python-sdk) (`mcp`, Transport stdio). Er ist die Schnittstelle zum laufenden Loop: Er liest dessen Gedächtnis und legt Nachrichten in die Inbox, schreibt aber nie selbst in `unikin_state.json`. Der Loop bleibt so der einzige Schreiber, parallele Aufrufe überschreiben keinen Zustand.

| Tool | Wirkung |
|---|---|
| `get_state` | Gespeicherter Zustand: Mission, letzter Gedanke, nächster Fokus, Metriken, Journal (nur lesend) |
| `get_journal` | Die neuesten Journal-Einträge, `limit` 1–40, Standard 10 (nur lesend) |
| `send_message` | Legt eine Nachricht (`text`, optional `sender`) in die Inbox und kehrt sofort zurück |
| `step` | Stößt sofort einen Zyklus an, optional mit Nachricht (`note`), und wartet auf dessen Journal-Eintrag (`timeout_seconds`, Standard 240) |

### Inbox

Jede Nachricht liegt als eigene Datei in `memory/inbox/`. Eine neue Nachricht weckt den wartenden Loop sofort, frühestens aber `MIN_INTERVAL_SECONDS` nach dem letzten Zyklus; steckt er gerade in einem Modellaufruf, ist sie im Zyklus danach dran. Pro Zyklus liest Unikin bis zu fünf Nachrichten (je höchstens 4000 Zeichen) im Prompt-Feld `inbox` und antwortet im Feld `reply`.

Nachrichten und Antwort landen im Journal-Eintrag des Zyklus (`inbox`, `inbox_ids`, `reply`); erst danach wird die Datei gelöscht. Ist LM Studio nicht erreichbar, bleibt die Nachricht liegen und kommt im nächsten Zyklus erneut. Läuft der Loop gar nicht, liefert `step` nach `timeout_seconds` den Status `pending`. Unlesbare Dateien werden als `.rejected` beiseitegelegt.

### Kommandozeilen-Client

```bash
python mcp_client.py tools                            # Tool-Liste mit Schemas
python mcp_client.py state                            # gespeicherter Zustand
python mcp_client.py journal --limit 3                # neueste Journal-Einträge
python mcp_client.py send "Hallo Unikin"              # Nachricht in die Inbox, kehrt sofort zurück
python mcp_client.py step --note "Wie geht es dir?"   # Zyklus anstoßen und auf die Antwort warten
```

Der Client startet den Server selbst als Subprozess (`--server-cmd`, Standard: `mcp_server.py` neben dem Client).

### In Claude Code oder Claude Desktop einbinden

Jeder MCP-Client, der stdio-Server starten kann, funktioniert. Für den Container auf dem Mac Studio, vom Laptop aus über SSH (der SSH-Schlüssel muss ohne Rückfrage funktionieren):

```bash
claude mcp add unikin -- ssh macstudio 'cd ~/unikin && PATH="$PATH:/opt/homebrew/bin" docker compose run --rm -T unikin python mcp_server.py'
```

In Claude Desktop (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "unikin": {
      "command": "ssh",
      "args": ["macstudio", "cd ~/unikin && PATH=\"$PATH:/opt/homebrew/bin\" docker compose run --rm -T unikin python mcp_server.py"]
    }
  }
}
```

## Start

```bash
python unikin.py
```

## Docker (Mac Studio + LM Studio)

Der Agent läuft als Container, LM Studio bleibt nativ auf dem Host (Metal/MLX) und wird aus dem Container über `host.docker.internal:1234` angesprochen. Getestet auf dem Mac Studio mit Colima; mit Docker Desktop funktioniert es genauso.

### Voraussetzungen auf dem Host

- Docker mit Compose ≥ 2.24 (Colima oder Docker Desktop)
- LM Studio mit laufendem Server auf Port 1234 (`lms server status`). Unter Colima muss der Server auf allen Interfaces lauschen („Serve on Local Network“ bzw. `networkInterface: 0.0.0.0`), weil Container den Host über die VM-Brücke erreichen; unter Docker Desktop reicht auch `127.0.0.1`.
- Das Modell aus `OPENAI_MODEL` ist geladen (`lms ps`) oder Just-in-Time-Laden ist in LM Studio aktiv.

### Starten

```bash
cp .env.example .env          # OPENAI_MODEL anpassen; OPENAI_BASE_URL darf auf localhost bleiben
mkdir -p memory logs          # Bind-Mounts; müssen vor dem Start existieren
docker compose up -d --build
docker compose logs -f
```

- Ohne `.env` startet Compose absichtlich nicht; ohne `memory/` und `logs/` ebenfalls nicht. Auf Linux-Hosts müssen die beiden Verzeichnisse für UID 1000 beschreibbar sein (der Container läuft nicht als root).
- Die Basis-URL setzt `docker-compose.yml` für den Container selbst auf `http://host.docker.internal:1234/v1`; `OPENAI_BASE_URL` in `.env` gilt nur für den nativen Start. Läuft LM Studio auf einem anderen Rechner, `LMSTUDIO_URL=http://<host>:1234/v1` in `.env` setzen.
- `memory/` und `logs/` sind Bind-Mounts im Projektverzeichnis und überleben Neustarts, Rebuilds und `docker compose down`. `logs/unikin.log` rotiert bei 10 MB (fünf Generationen); der MCP-Server schreibt nach `logs/unikin-mcp.log`.
- Jeder Modellaufruf hat ein Timeout von `OPENAI_TIMEOUT_SECONDS` (Standard 120 s; insgesamt bis zu drei Versuche pro Schritt). `stop_grace_period` (150 s) ist darauf abgestimmt, damit `docker compose stop` einen laufenden Aufruf zu Ende kommen lässt, statt den Agenten hart zu beenden.
- `restart: unless-stopped`: Der Container kommt nach einem Docker-Neustart von selbst wieder, solange er nicht mit `docker compose stop` angehalten wurde.

### Stoppen, Modell wechseln

```bash
docker compose stop     # SIGTERM; in der Wartephase endet der Agent innerhalb einer Sekunde
docker compose down     # Container entfernen; memory/ und logs/ bleiben erhalten
```

Modell wechseln: `OPENAI_MODEL` in `.env` ändern und `docker compose up -d` ausführen (kein Rebuild nötig). Verfügbare IDs liefert `curl http://localhost:1234/v1/models`. Das alte Modell bleibt in LM Studio geladen (bis zum Ablauf seiner TTL, ohne TTL dauerhaft); wer den Platz sofort braucht, entlädt es gezielt mit `lms unload <alte-modell-id>` (nicht `--all`, falls andere Clients LM Studio nutzen).

### Dauerbetrieb auf dem Mac

Docker (Colima bzw. Docker Desktop) und LM Studio laufen in der Benutzersitzung. Nach einem Neustart des Macs sind sie nur dann wieder da, wenn der Benutzer automatisch angemeldet wird, der Mac nicht schläft, Colima als Dienst startet (`brew services start colima`), LM Studio als Anmeldeobjekt (Systemeinstellungen → Allgemein → Anmeldeobjekte) gestartet wird und dort der Server beim App-Start aktiviert ist (Developer-Bereich, `autoStartOnLaunch`). Bis LM Studio erreichbar ist, protokolliert der Agent Verbindungsfehler und wartet mit wachsendem Intervall (bis `MAX_INTERVAL_SECONDS`).

### Deployment per rsync (z. B. vom Laptop auf den Mac Studio)

```bash
./deploy.sh                                          # Ziel: SSH-Host "macstudio", Verzeichnis ~/unikin
UNIKIN_HOST=anderer-host UNIKIN_DIR=agents/unikin ./deploy.sh   # Verzeichnis relativ zum Home oder absolut
```

Das Skript synchronisiert den Quellcode samt `Dockerfile`, `docker-compose.yml` und `deploy.sh` (ohne `.git`, `.env`, `memory/`, `logs/`) und führt auf dem Ziel `docker compose up -d --build` aus. Eine vorhandene `.env` auf dem Ziel bleibt unangetastet; fehlt sie, wird sie aus `.env.example` angelegt. Änderungen an `docker-compose.yml` (z. B. `stop_grace_period`) gehören deshalb ins Repository, nicht auf das Ziel.

### MCP-Server im Container

```bash
# Client und Server zusammen in einem Einmal-Container (kein Python auf dem Host nötig)
docker compose run --rm unikin python mcp_client.py state
docker compose run --rm unikin python mcp_client.py step --note "Hallo Unikin"

# Oder: Client auf dem Host (pip install -r requirements.txt), MCP-Server als Einmal-Container (stdio über -T)
python3 mcp_client.py --server-cmd "docker compose run --rm -T unikin python mcp_server.py" state
```

Der Einmal-Container teilt sich `memory/` mit dem Loop-Container: Er liest `unikin_state.json` und legt Nachrichten in `memory/inbox/`, von denen der Loop aufwacht. `step` und `send_message` brauchen deshalb einen laufenden Loop-Container.

### Fehlersuche

- `Connection error` im Log: Der LM-Studio-Server läuft nicht (`lms server start`) oder lauscht unter Colima nur auf `127.0.0.1`.
- `Request timed out`: Ein Aufruf hat länger als `OPENAI_TIMEOUT_SECONDS` gedauert, z. B. weil LM Studio ein großes Modell per JIT lädt. Der Agent versucht es bis zu dreimal; bei sehr großen Modellen das Timeout in `.env` erhöhen und `stop_grace_period` in `docker-compose.yml` (im Repository, dann `./deploy.sh`) mitziehen.
- Reasoning-Modelle (z. B. Qwen 3.6, Nemotron) liefern das JSON im `message`-Teil der Antwort; der Denkteil wird nicht ausgewertet.
- `step` liefert `pending`: Der Loop-Container läuft nicht (`docker compose ps`) oder LM Studio ist nicht erreichbar. Die Nachricht bleibt in `memory/inbox/` und wird nachgeholt, sobald ein Zyklus das Modell erreicht.

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
- Nachrichten an Unikin liegen bis zur Zustellung als einzelne Dateien in `memory/inbox/`.
- Logs landen sowohl in der Konsole als auch in der Datei `logs/unikin.log`.
- Ordner werden automatisch angelegt.

## Robustheit

- Normale Laufzeitfehler beenden den Agenten nicht dauerhaft.
- API-/JSON-/Netzwerkprobleme werden mit Retry und Fallback behandelt.
- `KeyboardInterrupt`/`SIGTERM` werden kontrolliert verarbeitet.

## Wichtiger Hinweis

Der Agent besitzt **keine automatische fachliche Abbruchbedingung** (kein „done“, kein „completed“, kein Max-Steps-Cutoff). Er läuft kontinuierlich, bis du ihn manuell stoppst.
