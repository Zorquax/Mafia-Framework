from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from ..paths import resolve_repo_path

# Python 3.11+ has tomllib, fall back to toml package for older versions if needed
if sys.version_info >= (3, 11):
    import tomllib
else:
    import toml as tomllib

# Loads PS_USERNAME/PS_PASSWORD/PS_ROOM (and anything else) from a local
# .env file into the environment, if one exists. Safe no-op otherwise.
load_dotenv()


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
    vote_comment_chance: float = 0.75
    town_read_comment_chance: float = 0.75
    vote_reaction_chance: float = 0.75
    random_action_interval_seconds: list[float] = field(default_factory=lambda: [180.0, 300.0])
    first_evaluation_delay_seconds: float = 60.0
    plurality_claim_check_seconds: list[float] = field(default_factory=lambda: [30.0, 20.0, 10.0, 5.0])
    volo_min_confidence: float = 0.75
    silent_mode: bool = False
    auto_save_games: bool = True
    autojoin_delay_seconds: float = 5.0
    troll_mode: bool = False
    plurality_defense_min_confidence: float = 0.85


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
        candidate = resolve_repo_path(path)

        data = {}
        if candidate.exists():
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
                vote_comment_chance=float(gameplay_data.get("vote_comment_chance", 0.75)),
                town_read_comment_chance=float(gameplay_data.get("town_read_comment_chance", 0.75)),
                vote_reaction_chance=float(gameplay_data.get("vote_reaction_chance", 0.75)),
                random_action_interval_seconds=[
                    float(v) for v in gameplay_data.get("random_action_interval_seconds", [180.0, 300.0])
                ],
                first_evaluation_delay_seconds=float(gameplay_data.get("first_evaluation_delay_seconds", 60.0)),
                plurality_claim_check_seconds=[
                    float(v) for v in gameplay_data.get("plurality_claim_check_seconds", [30.0, 20.0, 10.0, 5.0])
                ],
                volo_min_confidence=float(gameplay_data.get("volo_min_confidence", 0.75)),
                silent_mode=bool(gameplay_data.get("silent_mode", False)),
                auto_save_games=bool(gameplay_data.get("auto_save_games", True)),
                autojoin_delay_seconds=float(gameplay_data.get("autojoin_delay_seconds", 5.0)),
                troll_mode=bool(gameplay_data.get("troll_mode", False)),
                plurality_defense_min_confidence=float(gameplay_data.get("plurality_defense_min_confidence", 0.85)),
            ),
            database=DatabaseConfig(
                db_path=str(resolve_repo_path(database_data.get("db_path", "data/mafia.db"))),
                model_path=str(resolve_repo_path(database_data.get("model_path", "data/model.pkl"))),
                model_d1_path=str(resolve_repo_path(database_data.get("model_d1_path", "data/model_d1.pkl"))),
            ),
        )
