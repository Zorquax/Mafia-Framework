from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class Message:
    player_name: str
    text: str
    timestamp: str | None = None
    day: int = 1


@dataclass
class Vote:
    voter_name: str
    target_name: str
    timestamp: str | None = None
    text: str | None = None
    day: int = 1
    action: str = "vote"


@dataclass
class Flip:
    player_name: str
    alignment: str


@dataclass
class LogEvent:
    player_name: str
    event_type: str  # "elimination" or "reveal"
    text: str
    day: int = 1
    timestamp: str | None = None


@dataclass
class GameSession:
    source: str
    raw_text: str
    players: list[str] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)
    votes: list[Vote] = field(default_factory=list)
    flips: list[Flip] = field(default_factory=list)
    phases: list[str] = field(default_factory=list)
    events: list[LogEvent] = field(default_factory=list)
    game_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def initialize_database(db_path: str | Path) -> None:
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(str(db_file)) as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS games (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT,
                raw_text TEXT NOT NULL,
                created_at TEXT NOT NULL,
                display_name TEXT
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS players (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                UNIQUE(game_id, name),
                FOREIGN KEY(game_id) REFERENCES games(id)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id INTEGER NOT NULL,
                player_id INTEGER NOT NULL,
                timestamp TEXT,
                day INTEGER,
                text TEXT NOT NULL,
                FOREIGN KEY(game_id) REFERENCES games(id),
                FOREIGN KEY(player_id) REFERENCES players(id)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS votes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id INTEGER NOT NULL,
                voter_id INTEGER NOT NULL,
                target_id INTEGER NOT NULL,
                timestamp TEXT,
                text TEXT,
                FOREIGN KEY(game_id) REFERENCES games(id),
                FOREIGN KEY(voter_id) REFERENCES players(id),
                FOREIGN KEY(target_id) REFERENCES players(id)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS flips (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id INTEGER NOT NULL,
                player_id INTEGER NOT NULL,
                alignment TEXT NOT NULL,
                FOREIGN KEY(game_id) REFERENCES games(id),
                FOREIGN KEY(player_id) REFERENCES players(id)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS player_aliases (
                alias TEXT PRIMARY KEY,
                canonical_name TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS player_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id INTEGER NOT NULL,
                player_id INTEGER NOT NULL,
                feature_set_version TEXT NOT NULL,
                features_json TEXT NOT NULL,
                UNIQUE(game_id, player_id, feature_set_version),
                FOREIGN KEY(game_id) REFERENCES games(id),
                FOREIGN KEY(player_id) REFERENCES players(id)
            )
            """
        )
        connection.commit()
        _migrate_games_display_name(connection)


def _migrate_games_display_name(connection: sqlite3.Connection) -> None:
    cursor = connection.cursor()
    cursor.execute("PRAGMA table_info(games)")
    columns = {row[1] for row in cursor.fetchall()}
    if "display_name" not in columns:
        cursor.execute("ALTER TABLE games ADD COLUMN display_name TEXT")
        connection.commit()
