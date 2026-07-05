from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from ..data.models import GameSession, initialize_database
from ..io.ingestion import delete_game, load_game, load_games, set_flip_alignment
from ..io.player_names import canonical_player_name, names_match, player_identity_key


def canonical_key(name: str) -> str:
    return player_identity_key(name)


def resolve_flip_map(session: GameSession) -> dict[str, str]:
    canonical_player_map: dict[str, str] = {}
    for player in session.players:
        key = player_identity_key(player)
        if key and key not in canonical_player_map:
            canonical_player_map[key] = canonical_player_name(player)

    canonical_flip_map: dict[str, str] = {}
    for flip in session.flips:
        key = player_identity_key(flip.player_name)
        if key:
            canonical_flip_map[key] = flip.alignment

    resolved: dict[str, str] = {}
    for key, player_name in canonical_player_map.items():
        if key in canonical_flip_map:
            resolved[player_name] = canonical_flip_map[key]
    return resolved


@dataclass
class GameSummary:
    game_id: int
    display_name: str | None
    source: str
    created_at: str
    player_count: int
    flip_count: int
    undefined_count: int
    needs_review: bool


@dataclass
class UndefinedPlayerRow:
    game_id: int
    display_name: str | None
    player_name: str
    has_messages: bool
    is_inferred_town_candidate: bool


def _ensure_display_name_column(db_path: str | Path) -> None:
    initialize_database(db_path)
    with sqlite3.connect(str(db_path)) as connection:
        cursor = connection.cursor()
        cursor.execute("PRAGMA table_info(games)")
        columns = {row[1] for row in cursor.fetchall()}
        if "display_name" not in columns:
            cursor.execute("ALTER TABLE games ADD COLUMN display_name TEXT")
            connection.commit()


def set_game_display_name(db_path: str | Path, game_id: int, display_name: str) -> None:
    _ensure_display_name_column(db_path)
    with sqlite3.connect(str(db_path)) as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT id FROM games WHERE id = ?", (game_id,))
        if not cursor.fetchone():
            raise FileNotFoundError(f"Game not found: {game_id}")
        cursor.execute(
            "UPDATE games SET display_name = ? WHERE id = ?",
            (display_name.strip(), game_id),
        )
        connection.commit()


def get_game_display_name(db_path: str | Path, game_id: int) -> str | None:
    _ensure_display_name_column(db_path)
    with sqlite3.connect(str(db_path)) as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT display_name FROM games WHERE id = ?", (game_id,))
        row = cursor.fetchone()
        return row[0] if row else None


def list_game_summaries(db_path: str | Path) -> list[GameSummary]:
    _ensure_display_name_column(db_path)
    sessions = load_games(db_path)
    summaries: list[GameSummary] = []

    with sqlite3.connect(str(db_path)) as connection:
        cursor = connection.cursor()
        for session in sessions:
            game_id = session.game_id or 0
            cursor.execute(
                "SELECT display_name, source, created_at FROM games WHERE id = ?",
                (game_id,),
            )
            row = cursor.fetchone()
            display_name = row[0] if row else None
            source = row[1] if row else session.source
            created_at = row[2] if row else ""

            resolved = resolve_flip_map(session)
            undefined = sorted(set(session.players) - set(resolved.keys()))
            flip_count = len(resolved)
            player_count = len(session.players)
            needs_review = flip_count == 0 or bool(undefined)

            summaries.append(
                GameSummary(
                    game_id=game_id,
                    display_name=display_name,
                    source=source,
                    created_at=created_at,
                    player_count=player_count,
                    flip_count=flip_count,
                    undefined_count=len(undefined),
                    needs_review=needs_review,
                )
            )
    return summaries


def find_undefined_players(db_path: str | Path, game_id: int | None = None) -> list[UndefinedPlayerRow]:
    sessions = load_games(db_path)
    if game_id is not None:
        sessions = [s for s in sessions if s.game_id == game_id]

    rows: list[UndefinedPlayerRow] = []
    for session in sessions:
        if session.game_id is None:
            continue
        display_name = get_game_display_name(db_path, session.game_id)
        resolved = resolve_flip_map(session)
        inferred_town = set(session.players) - {flip.player_name for flip in session.flips}
        undefined = sorted(set(session.players) - set(resolved.keys()))
        message_players = {message.player_name for message in session.messages}

        for player_name in undefined:
            rows.append(
                UndefinedPlayerRow(
                    game_id=session.game_id,
                    display_name=display_name,
                    player_name=player_name,
                    has_messages=player_name in message_players,
                    is_inferred_town_candidate=player_name in inferred_town,
                )
            )
    return rows


def get_session(db_path: str | Path, game_id: int | None = None) -> GameSession | None:
    return load_game(db_path, game_id)


def remove_game(db_path: str | Path, game_id: int) -> None:
    delete_game(db_path, game_id)


def assign_player_role(
    db_path: str | Path,
    game_id: int,
    player_name: str,
    alignment: str,
) -> None:
    set_flip_alignment(db_path, game_id, player_name, alignment)


def assign_player_roles_bulk(
    db_path: str | Path,
    assignments: list[tuple[int, str, str]],
) -> int:
    applied = 0
    for game_id, player_name, alignment in assignments:
        if not alignment:
            continue
        assign_player_role(db_path, game_id, player_name, alignment)
        applied += 1
    return applied
