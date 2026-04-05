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
