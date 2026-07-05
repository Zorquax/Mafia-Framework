from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Python 3.11+ has tomllib, fall back to toml package for older versions if needed
if sys.version_info >= (3, 11):
    import tomllib
else:
    import toml as tomllib


@dataclass
class ShowdownConfig:
    server_url: str = "ws://sim3.psim.us/showdown/websocket"
    login_url: str = "https://play.pokemonshowdown.com/action.php"
    username: str = ""
    password: str = ""
    room: str = "mafia"


@dataclass
class GameplayConfig:
    autojoin: bool = True
    min_confidence_to_vote: float = 0.55
    night_idle: bool = True
    update_suspicion_frequency_seconds: float = 60.0
    min_seconds_between_vote_actions: float = 3.0
    random_vote_chance: float = 0.4


@dataclass
class DatabaseConfig:
    db_path: str = "data/mafia.db"
    model_path: str = "data/model.pkl"
    model_d1_path: str = "data/model_d1.pkl"


@dataclass
class BotConfig:
    showdown: ShowdownConfig
    gameplay: GameplayConfig
    database: DatabaseConfig

    @classmethod
    def load_from_file(cls, path: str | Path) -> BotConfig:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            cwd_candidate = Path.cwd() / candidate
            if cwd_candidate.exists():
                candidate = cwd_candidate
            else:
                repo_candidate = Path(__file__).resolve().parents[2] / candidate
                if repo_candidate.exists():
                    candidate = repo_candidate

        if not candidate.exists():
            # Return default configs if file not found
            return cls(
                showdown=ShowdownConfig(),
                gameplay=GameplayConfig(),
                database=DatabaseConfig(),
            )

        with candidate.open("rb") as handle:
            data = tomllib.load(handle)

        showdown_data = data.get("showdown", {})
        gameplay_data = data.get("gameplay", {})
        database_data = data.get("database", {})

        # Override with environment variables if present
        username = os.environ.get("PS_USERNAME", showdown_data.get("username", ""))
        password = os.environ.get("PS_PASSWORD", showdown_data.get("password", ""))
        room = os.environ.get("PS_ROOM", showdown_data.get("room", "mafia"))

        return cls(
            showdown=ShowdownConfig(
                server_url=showdown_data.get("server_url", "ws://sim3.psim.us/showdown/websocket"),
                login_url=showdown_data.get("login_url", "https://play.pokemonshowdown.com/action.php"),
                username=username,
                password=password,
                room=room,
            ),
            gameplay=GameplayConfig(
                autojoin=gameplay_data.get("autojoin", True),
                min_confidence_to_vote=float(gameplay_data.get("min_confidence_to_vote", 0.55)),
                night_idle=gameplay_data.get("night_idle", True),
                update_suspicion_frequency_seconds=float(gameplay_data.get("update_suspicion_frequency_seconds", 60.0)),
                min_seconds_between_vote_actions=float(gameplay_data.get("min_seconds_between_vote_actions", 3.0)),
                random_vote_chance=float(gameplay_data.get("random_vote_chance", 0.4)),
            ),
            database=DatabaseConfig(
                db_path=database_data.get("db_path", "data/mafia.db"),
                model_path=database_data.get("model_path", "data/model.pkl"),
                model_d1_path=database_data.get("model_d1_path", "data/model_d1.pkl"),
            ),
        )
