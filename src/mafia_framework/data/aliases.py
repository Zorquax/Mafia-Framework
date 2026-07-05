from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ..io.parser import normalize_player_name
from ..io.player_names import canonical_player_name, names_match, player_identity_key, resolve_to_roster
from .models import Flip, GameSession, initialize_database


def resolve_name(name: str, alias_map: dict[str, str]) -> str:
    if not alias_map:
        return canonical_player_name(name)

    identity_to_canonical: dict[str, str] = {}
    for alias, canonical in alias_map.items():
        resolved = resolve_to_roster(canonical, identity_to_canonical)
        identity_to_canonical[player_identity_key(alias)] = resolved

    current = resolve_to_roster(name, identity_to_canonical)
    key = player_identity_key(current)
    if key in identity_to_canonical:
        return identity_to_canonical[key]

    seen: set[str] = set()
    lookup = canonical_player_name(name)
    while lookup in alias_map:
        target = canonical_player_name(alias_map[lookup])
        if target == lookup or lookup in seen:
            break
        seen.add(lookup)
        lookup = target
    return lookup


def _dedupe_preserve_order(names: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for name in names:
        key = player_identity_key(name)
        if key and key not in seen:
            seen.add(key)
            result.append(canonical_player_name(name))
    return result


def apply_player_aliases(session: GameSession, alias_map: dict[str, str]) -> GameSession:
    if not alias_map:
        return session

    def resolve(name: str) -> str:
        return resolve_name(name, alias_map)

    session.players = _dedupe_preserve_order(resolve(player) for player in session.players)

    for message in session.messages:
        message.player_name = resolve(message.player_name)

    for vote in session.votes:
        vote.voter_name = resolve(vote.voter_name)
        vote.target_name = resolve(vote.target_name)

    flip_by_player: dict[str, str] = {}
    for flip in session.flips:
        flip_by_player[resolve(flip.player_name)] = flip.alignment
    session.flips = [
        Flip(player_name=player_name, alignment=alignment)
        for player_name, alignment in flip_by_player.items()
    ]

    if hasattr(session, "events") and session.events:
        for event in session.events:
            old_name = event.player_name
            new_name = resolve(old_name)
            event.player_name = new_name
            event.text = event.text.replace(old_name, new_name)

    return session


def load_alias_map(db_path: str | Path) -> dict[str, str]:
    db_file = Path(db_path)
    if not db_file.exists():
        return {}

    with sqlite3.connect(str(db_file)) as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT alias, canonical_name FROM player_aliases")
        return {
            player_identity_key(alias): canonical_player_name(canonical)
            for alias, canonical in cursor.fetchall()
        }


def set_player_alias(db_path: str | Path, alias: str, canonical: str) -> None:
    db_file = Path(db_path)
    initialize_database(db_file)
    normalized_alias = canonical_player_name(alias)
    normalized_canonical = canonical_player_name(canonical)

    with sqlite3.connect(str(db_file)) as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            INSERT INTO player_aliases (alias, canonical_name)
            VALUES (?, ?)
            ON CONFLICT(alias) DO UPDATE SET canonical_name = excluded.canonical_name
            """,
            (normalized_alias, normalized_canonical),
        )
        connection.commit()


def remove_player_alias(db_path: str | Path, alias: str) -> bool:
    db_file = Path(db_path)
    if not db_file.exists():
        return False

    normalized_alias = canonical_player_name(alias)
    with sqlite3.connect(str(db_file)) as connection:
        cursor = connection.cursor()
        cursor.execute("DELETE FROM player_aliases WHERE alias = ?", (normalized_alias,))
        connection.commit()
        return cursor.rowcount > 0


def list_player_aliases(db_path: str | Path) -> list[tuple[str, str]]:
    db_file = Path(db_path)
    if not db_file.exists():
        return []

    with sqlite3.connect(str(db_file)) as connection:
        cursor = connection.cursor()
        cursor.execute(
            "SELECT alias, canonical_name FROM player_aliases ORDER BY canonical_name, alias"
        )
        return [(row[0], row[1]) for row in cursor.fetchall()]


def import_aliases_from_json(db_path: str | Path, json_path: str | Path) -> int:
    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Alias JSON must be an object mapping alias -> canonical name.")

    count = 0
    for alias, canonical in payload.items():
        if not isinstance(alias, str) or not isinstance(canonical, str):
            continue
        set_player_alias(db_path, alias, canonical)
        count += 1
    return count
