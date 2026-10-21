# unikin

`unikin` ist ein autonomer Python-Agent, der dauerhaft (`while True`) über eine lokal laufende OpenAI-kompatible LLM-API arbeitet.

## Features

- Echte Endlosschleife ohne fachliche Abbruchbedingung
- Konfigurierbare Identität (`Unikin`) und Mission
- Persistentes Memory in JSON (`memory/unikin_state.json`)
- Durchsuchbares Langzeitgedächtnis: Jeder Journal-Eintrag landet in `memory/unikin_archive.sqlite3` (SQLite mit Volltextsuche). Unikin durchsucht es mit dem Werkzeug `search_memory`, und passende ältere Einträge zum aktuellen Fokus stehen als `related_memory` im Prompt
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
- Werkzeug `run_python`: Unikin führt Python-Code in einer netzwerkisolierten Sandbox aus (python-http-mcp-runner per MCP)

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
OPENAI_TIMEOUT_SECONDS=300
OPENAI_MAX_OUTPUT_TOKENS=12000

UNIKIN_IDENTITY=Unikin
UNIKIN_MISSION=Kontinuierliche reflektierte Arbeit in kleinen Schritten. Baue auf früheren Ergebnissen auf: Durchsuche dein Langzeitgedächtnis, bevor du etwas erneut prüfst.

MIN_INTERVAL_SECONDS=5
MAX_INTERVAL_SECONDS=300
DEFAULT_INTERVAL_SECONDS=30

MEMORY_FILE=memory/unikin_state.json
LOG_FILE=logs/unikin.log
RELATED_MEMORY_HITS=3
```


## MCP-Server & MCP-Client

`mcp_server.py` ist ein MCP-Server nach Spezifikation auf Basis des offiziellen [MCP-Python-SDK](https://github.com/modelcontextprotocol/python-sdk) (`mcp`). Standard ist stdio; mit `--http` läuft er dauerhaft über Streamable HTTP mit OAuth (siehe [unten](#über-http-mit-oauth-z-b-librechat)). Er ist die Schnittstelle zum laufenden Loop: Er liest dessen Gedächtnis und legt Nachrichten in die Inbox, schreibt aber nie selbst in `unikin_state.json`. Der Loop bleibt so der einzige Schreiber, parallele Aufrufe überschreiben keinen Zustand.

| Tool | Wirkung |
|---|---|
| `get_state` | Gespeicherter Zustand: Mission, letzter Gedanke, nächster Fokus, Metriken, Journal (nur lesend) |
| `get_journal` | Die neuesten Journal-Einträge, `limit` 1–40, Standard 10 (nur lesend) |
| `search_journal` | Volltextsuche im [Langzeitgedächtnis](#langzeitgedächtnis), auch in Einträgen, die nicht mehr im Journal stehen; `query`, `limit` 1–20, Standard 10, bester Treffer zuerst (nur lesend) |
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
python mcp_client.py search "Kuramoto" --limit 5      # alle archivierten Einträge durchsuchen
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

### Über HTTP mit OAuth (z. B. LibreChat)

`python mcp_server.py --http` startet den Server dauerhaft über Streamable HTTP auf Port 8020. Er ist dann zugleich sein eigener OAuth-2.1-Autorisierungsserver, nach dem Muster von OpenBuchhaltung und mcp_imap:

- Discovery (`/.well-known/oauth-protected-resource/mcp`, `/.well-known/oauth-authorization-server`), dynamische Client-Registrierung (`/register`), `/authorize`, `/token` und `/revoke` liefert das MCP-SDK.
- `/login` ist die Login- und Zustimmungsseite. Die Konten stehen mit PBKDF2-Hash in `UNIKIN_OAUTH_USERS`. Nach 10 Fehlversuchen in 15 Minuten ist der Login gesperrt.
- PKCE S256, Access-Token 1 h, Refresh-Token 30 Tage und rotierend. Gespeichert wird nur der SHA-256-Hash der Tokens, in `memory/oauth.sqlite3`.
- Tokens gelten nur für `<UNIKIN_MCP_PUBLIC_URL>/mcp` (RFC 8707). Redirect-URIs müssen HTTPS sein (HTTP nur auf localhost) und auf einem freigegebenen Host liegen.

| Variable | Bedeutung |
|---|---|
| `UNIKIN_MCP_PUBLIC_URL` | HTTPS-Origin, unter der Clients und Browser den Server erreichen; Issuer und Ressource leiten sich davon ab |
| `UNIKIN_OAUTH_USERS` | Konten als `name:hash`, mehrere durch Komma getrennt |
| `UNIKIN_OAUTH_ALLOWED_REDIRECT_HOSTS` | Erlaubte Hosts für Redirect-URIs, Standard: der Host von `UNIKIN_MCP_PUBLIC_URL` |

Einrichten auf dem Mac Studio:

1. Passwort-Hash erzeugen. Das Skript fragt das Passwort verdeckt ab und braucht nur die Standardbibliothek, läuft also auch auf dem Laptop: `python3 passwords.py`
2. In `~/unikin/.env` eintragen:
   ```dotenv
   COMPOSE_FILE=docker-compose.yml:docker-compose.mcp-http.yml
   UNIKIN_MCP_PUBLIC_URL=https://macstudio.tail717550.ts.net:8020
   UNIKIN_OAUTH_USERS=christoph:<Hash aus Schritt 1>
   ```
3. `./deploy.sh` startet zusätzlich den Container `unikin-mcp`, erreichbar nur auf `127.0.0.1:18020`.
4. Ins Tailnet bringen (nur Tailnet, kein Funnel): `tailscale serve --bg --https=8020 http://127.0.0.1:18020`

In LibreChat (rebelchat, Instanz-Overlay `config/instances/unikin.yaml`):

```yaml
mcpSettings:
  allowedDomains:
    - 'https://macstudio.tail717550.ts.net:8020'
  allowedAddresses:
    - 'macstudio.tail717550.ts.net:8020'

mcpServers:
  unikin:
    title: 'Unikin'
    description: 'Autonomer Agent auf dem Mac Studio: Gedächtnis lesen, Nachrichten schicken, Zyklen anstoßen'
    type: streamable-http
    url: https://macstudio.tail717550.ts.net:8020/mcp
    requiresOAuth: true
    startup: false
    serverInstructions: true
    initTimeout: 15000
    timeout: 300000   # step wartet bis zu 240 s auf einen Zyklus
```

## Werkzeuge: Python-Runner

Unikin kann Python-Code ausführen. Das Werkzeug liefert der [python-http-mcp-runner](https://github.com/unikinCK/python-http-mcp-runner) (privat), ein MCP-Server über Streamable HTTP mit dem Tool `run_python`. Er ist als Git-Submodul unter `runner/` eingebunden und läuft im Docker-Betrieb als eigener Container neben Unikin.

So läuft ein Zyklus mit Werkzeugen:

1. Unikin holt die Tool-Liste vom Runner und bietet sie dem Modell an (Responses-API, Function Calling).
2. Ruft das Modell ein Werkzeug auf, führt Unikin den Aufruf über MCP aus und gibt die Ausgabe zurück. Das wiederholt sich, bis das Modell mit seinem JSON antwortet.
3. Pro Zyklus sind höchstens `MAX_TOOL_CALLS_PER_CYCLE` Aufrufe erlaubt (Standard 3). Ist das Budget aufgebraucht, stellt Unikin eine frische Schlussanfrage ohne Werkzeuge, in der die Ergebnisse als `tool_results` stehen. Qwen ruft nach einem Werkzeugergebnis sonst weiter Werkzeuge auf (auch mit `tool_choice: "none"`) oder denkt ohne angebotene Werkzeuge endlos nach. Dieselbe Schlussanfrage stellt Unikin auch, wenn das Modell nach Werkzeugaufrufen von selbst aufhört, aber nicht mit JSON antwortet (etwa mit Zeilen wie `thought: "…"`). Ohne Werkzeugaufrufe bleibt es bei einer Anfrage.
4. Code und Ausgabe landen gekürzt im Journal-Eintrag (`tool_calls`). Ist der Runner nicht erreichbar, läuft der Zyklus ohne seine Werkzeuge weiter; `search_memory` (siehe [Langzeitgedächtnis](#langzeitgedächtnis)) bleibt verfügbar.

| Variable | Bedeutung |
|---|---|
| `RUNNER_MCP_URL` | MCP-Endpunkt des Runners. Im Docker-Betrieb setzt `docker-compose.yml` `http://runner:8812/mcp`; leer (`RUNNER_MCP_URL=` in `.env`) schaltet die Werkzeuge ab |
| `MAX_TOOL_CALLS_PER_CYCLE` | Höchstzahl der Werkzeugaufrufe pro Zyklus, `run_python` und `search_memory` zusammen (Standard 3, `0` schaltet ab) |

### Isolation

Die Sperrliste des Runners prüft nur Text (`"socket"`, `"urllib"` …) und lässt sich umgehen, etwa mit `importlib.import_module("sock" + "et")`. Für einen Agenten, der unbeaufsichtigt Code schreibt, sichert deshalb das Netz ab:

- Der Runner hängt nur im Docker-Netz `sandbox` (`internal: true`) und hat kein Port-Mapping. Er erreicht weder Internet noch DNS, weder den Host (LM Studio) noch das LAN, und von außen ist er nicht erreichbar. Nur der unikin-Container spricht ihn unter `http://runner:8812/mcp` an.
- Dazu kommen `cap_drop: ALL`, `no-new-privileges`, 768 MB Speicher und 128 Prozesse; im Runner selbst laufen Snippets mit Zeit-, CPU- und Speicherlimits in einem frischen Temp-Verzeichnis.
- Jeder Aufruf startet ohne Gedächtnis; Unikins eigene Dateien sind für den Runner nicht sichtbar. Vom Host sieht er nur `artifacts/` (siehe [Erzeugte Dateien](#erzeugte-dateien)).

Auf dem Mac Studio geprüft: Code, der die Sperrliste umgeht, scheitert an `1.1.1.1:443`, `host.docker.internal:1234` und dem LAN mit `Network is unreachable` bzw. ohne Namensauflösung.

### Erzeugte Dateien

Abbildungen (PNG, SVG …) und Berichte (docx, pptx, xlsx, pdf, csv …), die der Code in seinem Arbeitsverzeichnis anlegt, gibt der Runner als Artefakte zurück, höchstens fünf pro Aufruf. Im Docker-Betrieb liegen sie im Projektverzeichnis unter `artifacts/<id>/<datei>` (auf dem Mac Studio `~/unikin/artifacts/`), eingebunden als `/artifacts` im Runner. Sie überleben Neustarts, Rebuilds, `docker compose down` und `./deploy.sh`.

- Das Journal nennt jedes Artefakt als `/artifacts/<id>/<datei>` in der Werkzeugausgabe; derselbe Pfad unter `artifacts/` ist die Datei. Als Link sind sie nicht erreichbar, weil der Runner kein Port-Mapping hat.
- Ansehen: auf dem Mac Studio mit `open ~/unikin/artifacts` im Finder, vom Laptop aus mit `rsync -a macstudio:unikin/artifacts/ ./artifacts/` (von git und `deploy.sh` ignoriert).
- Der Runner löscht nichts selbst (`PYTHON_MCP_ARTIFACT_RETENTION_DAYS: "0"` in `docker-compose.yml`); `artifacts/` wächst, bis man von Hand alte Ordner entfernt. Mit einer Zahl > 0 löscht der Runner Ordner, die älter als so viele Tage sind.
- Die Dateien stammen aus Code, den das Modell geschrieben hat. Dieser Code kann alles unter `artifacts/` lesen, ändern und löschen, aber nichts sonst auf dem Host.

### Submodul

```bash
git clone --recurse-submodules https://github.com/unikinCK/unikin.git
# oder in einer bestehenden Kopie (auch in jedem neuen Worktree):
git submodule update --init
```

Neue Runner-Version übernehmen: `git -C runner pull origin main`, dann `git add runner` und committen.

## Langzeitgedächtnis

Das Journal in `unikin_state.json` behält nur die letzten 40 Einträge, und der Prompt zeigt davon sechs (`recent_journal`). Ohne Archiv vergaß Unikin nach etwa einer halben Stunde, was es geprüft und berechnet hatte: Am 20.10. wiederholte es dutzende Male dieselbe Prüfung der Sandbox (`os.listdir`, `code.py`), und seine Kuramoto-Simulationen aus der Nacht waren am Morgen aus seinem Gedächtnis verschwunden.

Das Archiv `memory/unikin_archive.sqlite3` behält jeden Eintrag. Es nutzt `sqlite3` aus der Standardbibliothek mit einem FTS5-Volltextindex, braucht also weder einen eigenen Dienst noch ein Embedding-Modell und schickt keine zusätzliche Anfrage an LM Studio.

- **Was gespeichert wird:** `ts`, `loop`, `thought`, `state_update`, `next_focus`, `urgency`, `inbox`, `reply` und die Werkzeugaufrufe, so gekürzt wie im Journal (Code bis 600, Ausgabe bis 400 Zeichen). Die Ausgabe von `search_memory` selbst wird nicht archiviert, sonst fände jede spätere Suche ihre eigenen alten Ergebnisse.
- **Wann:** Nach jedem Zyklus, im selben Moment wie `unikin_state.json`, in einer Transaktion. Schlüssel ist `ts`, jeder Eintrag kommt nur einmal hinein. Der Loop übergibt dabei das ganze Journal: Scheitert das Archivieren einmal, steht im Log `Journal konnte nicht archiviert werden, der Zyklus läuft weiter`, und der nächste Zyklus holt den Eintrag nach, solange er noch im Journal steht. Ein Archivfehler bricht nie einen Zyklus ab.
- **Backfill:** Beim Start übernimmt der Loop die vorhandenen Journal-Einträge. Bei ihnen bleibt `loop` leer, weil das Journal die Zyklusnummer erst seit dem Archiv speichert. Ältere Journale aus gesicherten Zuständen lassen sich nachtragen, doppelte Einträge bleiben dabei wirkungslos:
  ```bash
  docker compose run --rm unikin python journal_archive.py import memory/unikin_state.backup-*.json
  ```
- **Suche:** Jedes Wort zählt einzeln, ein Treffer braucht nicht alle. Ab vier Buchstaben findet ein Wort auch längere Formen („Ergebnis“ → „Ergebnisse“), Groß- und Kleinschreibung und Umlaute sind egal („Prufung“ findet „Prüfung“). Sortiert wird nach Relevanz (bm25), seltene Wörter wiegen mehr. Es ist eine Wortsuche: Synonyme oder sinnverwandte Begriffe findet sie nicht.
- **Für Unikin:** Das Werkzeug `search_memory` (`query`, `limit` 1–5, Standard 3) steht neben `run_python` in der Tool-Schleife und zählt zum selben Budget `MAX_TOOL_CALLS_PER_CYCLE`. Es läuft im Loop-Prozess und funktioniert auch, wenn der Runner nicht erreichbar ist. Jeder Treffer enthält `ts`, `loop`, `state_update` (bis 240 Zeichen) und `match`, die passende Textstelle. Fast gleiche Einträge (Unikin schreibt oft viele Zyklen lang denselben Gedanken) zählen als ein Treffer.
- **related_memory:** Qwen ruft Suchwerkzeuge selten von selbst auf. Deshalb sucht der Loop vor jedem Zyklus mit dem aktuellen `next_focus` im Archiv, ohne Modellaufruf, und legt die besten `RELATED_MEMORY_HITS` Treffer (Standard 3, höchstens 5, `0` schaltet ab) als `related_memory` in den Prompt. Einträge, die `recent_journal` ohnehin zeigt, fallen dabei weg. Auf dem Mac Studio sind das rund 1000 Zeichen zusätzlich pro Prompt.
- **Für Menschen:** `search_journal` im MCP-Server bzw. `python mcp_client.py search "…"` liefert ganze Einträge, bis zu 20. Der MCP-Server öffnet das Archiv schreibgeschützt.
- **Größe:** Das Archiv löscht nichts. Ein Eintrag braucht rund 2 KB, bei einem Zyklus alle fünf Minuten sind das etwa 0,6 MB pro Tag.

### Mission anpassen

Die Mission aus `UNIKIN_MISSION` beschreibt Unikin, was es wahrnimmt. Steht dort wie auf dem Mac Studio, sein Gedächtnis sei nur das, was es in `thought`, `state_update` und `next_focus` zurückgibt, widerspricht das dem Archiv. `./deploy.sh` ändert die `.env` auf dem Ziel nie, die Mission muss man dort von Hand ergänzen, etwa so:

```text
Dein Langzeitgedächtnis: Jeder Zyklus wird archiviert. recent_journal zeigt nur die letzten sechs Einträge; mit dem Werkzeug search_memory durchsuchst du alle früheren, auch Werkzeugaufrufe mit ihren Ergebnissen, und related_memory zeigt dir ältere Einträge zu deinem next_focus. Suche dort, bevor du etwas erneut prüfst oder berechnest, und baue auf deinen früheren Ergebnissen auf.
```

## Start

```bash
python unikin.py
```

Nativ gestartet nutzt Unikin Werkzeuge nur, wenn `RUNNER_MCP_URL` in `.env` auf einen laufenden Runner zeigt.

## Docker (Mac Studio + LM Studio)

Der Agent läuft als Container, LM Studio bleibt nativ auf dem Host (Metal/MLX) und wird aus dem Container über `host.docker.internal:1234` angesprochen. Getestet auf dem Mac Studio mit Colima; mit Docker Desktop funktioniert es genauso.

### Voraussetzungen auf dem Host

- Docker mit Compose ≥ 2.24 (Colima oder Docker Desktop)
- LM Studio mit laufendem Server auf Port 1234 (`lms server status`). Unter Colima muss der Server auf allen Interfaces lauschen („Serve on Local Network“ bzw. `networkInterface: 0.0.0.0`), weil Container den Host über die VM-Brücke erreichen; unter Docker Desktop reicht auch `127.0.0.1`.
- Das Modell aus `OPENAI_MODEL` ist geladen (`lms ps`) oder Just-in-Time-Laden ist in LM Studio aktiv.

### Starten

```bash
git submodule update --init   # Python-Runner unter runner/
cp .env.example .env          # OPENAI_MODEL anpassen; OPENAI_BASE_URL darf auf localhost bleiben
mkdir -p memory logs artifacts   # Bind-Mounts; müssen vor dem Start existieren
docker compose up -d --build
docker compose logs -f
```

- Compose startet zwei Container: `unikin` (der Loop) und `unikin-runner` (die Sandbox für `run_python`, siehe [Werkzeuge](#werkzeuge-python-runner)).
- Ohne `.env` startet Compose absichtlich nicht; ohne `memory/`, `logs/` und `artifacts/` ebenfalls nicht. Auf Linux-Hosts müssen `memory/` und `logs/` für UID 1000 beschreibbar sein, `artifacts/` für UID 10001 (`sudo chown 10001 artifacts`); die Container laufen nicht als root. Unter Colima ist das nicht nötig.
- Die Basis-URL setzt `docker-compose.yml` für den Container selbst auf `http://host.docker.internal:1234/v1`; `OPENAI_BASE_URL` in `.env` gilt nur für den nativen Start. Läuft LM Studio auf einem anderen Rechner, `LMSTUDIO_URL=http://<host>:1234/v1` in `.env` setzen.
- `memory/`, `logs/` und `artifacts/` sind Bind-Mounts im Projektverzeichnis und überleben Neustarts, Rebuilds und `docker compose down`. `logs/unikin.log` rotiert bei 10 MB (fünf Generationen); der MCP-Server schreibt nach `logs/unikin-mcp.log`.
- Jeder Modellaufruf hat ein Timeout von `OPENAI_TIMEOUT_SECONDS` (Standard 300 s). Nach einem Timeout gibt es keinen zweiten Versuch im selben Zyklus, weil LM Studio die abgebrochene Anfrage noch verarbeiten kann; die nächste Anfrage kommt erst nach der regulären Pause, die ab dem Ende des Zyklus zählt. So läuft nie mehr als eine Anfrage von Unikin gleichzeitig. Bei Verbindungs- und HTTP-Fehlern versucht es der Agent bis zu dreimal. `stop_grace_period` (330 s) ist darauf abgestimmt, damit `docker compose stop` einen laufenden Aufruf zu Ende kommen lässt, statt den Agenten hart zu beenden.
- Jede Anfrage darf höchstens `OPENAI_MAX_OUTPUT_TOKENS` Tokens erzeugen, Denken und Antwort zusammen (Standard 12000, `0` = keine Grenze). Qwen 3.6 gerät gelegentlich in eine Denkschleife ohne Ende; das Budget beendet sie, lange bevor das Timeout greift (12000 Tokens brauchen auf dem Mac Studio rund 125 s). Schneidet das Budget eine Antwort ab, fragt Unikin im selben Zyklus genau einmal ohne Denken nach (`reasoning: {"effort": "none"}`). Die erste Anfrage ist dann in LM Studio schon beendet, es läuft also auch hier nie mehr als eine gleichzeitig. Bleibt auch die Nachfrage ohne Antwort, endet der Zyklus mit „Modell hat das Denkbudget überschritten“.
- `restart: unless-stopped`: Der Container kommt nach einem Docker-Neustart von selbst wieder, solange er nicht mit `docker compose stop` angehalten wurde.

### Stoppen, Modell wechseln

```bash
docker compose stop     # SIGTERM; in der Wartephase endet der Agent innerhalb einer Sekunde
docker compose down     # Container entfernen; memory/, logs/ und artifacts/ bleiben erhalten
```

Modell wechseln: `OPENAI_MODEL` in `.env` ändern und `docker compose up -d` ausführen (kein Rebuild nötig). Verfügbare IDs liefert `curl http://localhost:1234/v1/models`. Das alte Modell bleibt in LM Studio geladen (bis zum Ablauf seiner TTL, ohne TTL dauerhaft); wer den Platz sofort braucht, entlädt es gezielt mit `lms unload <alte-modell-id>` (nicht `--all`, falls andere Clients LM Studio nutzen).

### Dauerbetrieb auf dem Mac

Docker (Colima bzw. Docker Desktop) und LM Studio laufen in der Benutzersitzung. Nach einem Neustart des Macs sind sie nur dann wieder da, wenn der Benutzer automatisch angemeldet wird, der Mac nicht schläft, Colima als Dienst startet (`brew services start colima`), LM Studio als Anmeldeobjekt (Systemeinstellungen → Allgemein → Anmeldeobjekte) gestartet wird und dort der Server beim App-Start aktiviert ist (Developer-Bereich, `autoStartOnLaunch`). Bis LM Studio erreichbar ist, protokolliert der Agent Verbindungsfehler und wartet mit wachsendem Intervall (bis `MAX_INTERVAL_SECONDS`).

### Deployment per rsync (z. B. vom Laptop auf den Mac Studio)

```bash
./deploy.sh                                          # Ziel: SSH-Host "macstudio", Verzeichnis ~/unikin
UNIKIN_HOST=anderer-host UNIKIN_DIR=agents/unikin ./deploy.sh   # Verzeichnis relativ zum Home oder absolut
```

Das Skript synchronisiert den Quellcode samt `Dockerfile`, `docker-compose.yml`, `deploy.sh` und dem Runner-Submodul `runner/` (ohne `.git`, `.env`, `memory/`, `logs/`, `artifacts/`) und führt auf dem Ziel `docker compose up -d --build` aus. Der Mac Studio braucht so keinen Zugang zum privaten Runner-Repository; ist `runner/` leer, bricht das Skript mit einem Hinweis ab. Eine vorhandene `.env` auf dem Ziel bleibt unangetastet; fehlt sie, wird sie aus `.env.example` angelegt. Änderungen an `docker-compose.yml` (z. B. `stop_grace_period`) gehören deshalb ins Repository, nicht auf das Ziel.

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
- `Request timed out`: Ein Aufruf hat länger als `OPENAI_TIMEOUT_SECONDS` gedauert, z. B. weil LM Studio ein großes Modell per JIT lädt. Der Zyklus endet dann ohne zweiten Versuch, der nächste folgt nach der Pause; bei sehr großen Modellen das Timeout in `.env` erhöhen und `stop_grace_period` in `docker-compose.yml` (im Repository, dann `./deploy.sh`) mitziehen.
- `Denkbudget von … Tokens überschritten, frage einmal ohne Denken nach` im Log: Das Modell hat bis zum Budget nur nachgedacht, meist in einer Endlosschleife. Das ist kein API-Fehler.
- Reasoning-Modelle (z. B. Qwen 3.6, Nemotron) liefern das JSON im `message`-Teil der Antwort; der Denkteil wird nicht ausgewertet. Bei Qwen 3.6 schaltet `reasoning: {"effort": "none"}` das Denken ab; `minimal`, `low` und `high` wirken in LM Studio nicht, das Modell kennt nur an oder aus.
- `Python-Runner nicht erreichbar, dieser Zyklus läuft ohne seine Werkzeuge` im Log: Der Runner-Container läuft nicht (`docker compose ps`, `docker compose logs runner`). Unikin arbeitet ohne `run_python` weiter und versucht es jeden Zyklus erneut.
- `Journal konnte nicht archiviert werden` oder `Langzeitgedächtnis nicht durchsuchbar` im Log: `memory/unikin_archive.sqlite3` ist nicht schreibbar oder beschädigt. Der Zyklus läuft trotzdem weiter. Prüfen lässt sich die Datei mit `sqlite3 memory/unikin_archive.sqlite3 'PRAGMA integrity_check'`; eine beschädigte Datei beiseitelegen, dann legt der Loop ein neues Archiv an und übernimmt die Einträge, die noch im Journal stehen.
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
- Jeder Journal-Eintrag landet zusätzlich im [Langzeitgedächtnis](#langzeitgedächtnis) `memory/unikin_archive.sqlite3`.
- Logs landen sowohl in der Konsole als auch in der Datei `logs/unikin.log`.
- Ordner werden automatisch angelegt.

## Robustheit

- Normale Laufzeitfehler beenden den Agenten nicht dauerhaft.
- API-/JSON-/Netzwerkprobleme werden mit Retry und Fallback behandelt.
- `KeyboardInterrupt`/`SIGTERM` werden kontrolliert verarbeitet.

## Wichtiger Hinweis

Der Agent besitzt **keine automatische fachliche Abbruchbedingung** (kein „done“, kein „completed“, kein Max-Steps-Cutoff). Er läuft kontinuierlich, bis du ihn manuell stoppst.
