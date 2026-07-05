# Mafia Framework

An all-in-one toolkit for Pokemon Showdown Mafia: it ingests game logs, extracts
per-player behavioral "tells" (vote patterns, keyword usage, timing, etc.),
trains a model to predict town/mafia alignment, exposes a Streamlit dashboard
for reviewing games and training data, and runs a websocket bot that plays
live games on Showdown using the trained model.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Credentials

The bot needs a Pokemon Showdown username/password. **Never commit real
credentials** — `config.toml` and `.env` are both gitignored for this reason.
Pick whichever of these fits how you work:

**Option A — `config.toml` (quick, local-only)**

```bash
cp config.example.toml config.toml
```

Edit `config.toml` and fill in `[showdown].username` / `[showdown].password`.
This file is gitignored, so it stays local to your machine.

**Option B — `.env` file (recommended if others will run this too)**

```bash
cp .env.example .env
```

Edit `.env` and set `PS_USERNAME` / `PS_PASSWORD` (and optionally `PS_ROOM`).
These are loaded automatically and take priority over whatever is in
`config.toml`, so `config.toml` itself never needs to hold a real password —
contributors can each keep their own `.env` without touching a shared file.

## Usage

```bash
# Initialize the database
mafia-framework --db data/mafia.db init-db

# Ingest a game log
mafia-framework --db data/mafia.db ingest path/to/log.txt

# Train a model
mafia-framework --db data/mafia.db train

# Run the dashboard
mafia-dashboard

# Run the live bot
mafia-framework start-bot --config config.toml
```

Run `mafia-framework --help` for the full list of commands (aliases,
predictions, game management, etc.).

## Tests

```bash
pytest
```
