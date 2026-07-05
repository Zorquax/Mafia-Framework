import sqlite3
from pathlib import Path
from typing import Iterable

from ..data.aliases import apply_player_aliases, load_alias_map
from ..data.flips import extract_flips
from ..io.player_names import canonical_player_name, player_identity_key
from ..data.models import Flip, GameSession, Message, Vote, initialize_database
from ..io.google_docs import fetch_published_google_doc
from ..io.parser import parse_showdown_log


def _get_or_create_player_id(connection: sqlite3.Connection, game_id: int, username: str) -> int:
    cursor = connection.cursor()
    cursor.execute(
        "SELECT id FROM players WHERE game_id = ? AND name = ?",
        (game_id, username),
    )
    row = cursor.fetchone()
    if row:
        return int(row[0])

    cursor.execute(
        "INSERT INTO players (game_id, name) VALUES (?, ?)",
        (game_id, username),
    )
    return int(cursor.lastrowid)


def insert_game(db_path: str | Path, session: GameSession) -> int:
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    initialize_database(db_file)

    with sqlite3.connect(str(db_file)) as connection:
        cursor = connection.cursor()
        cursor.execute(
            "INSERT INTO games (source, raw_text, created_at) VALUES (?, ?, datetime('now'))",
            (session.source, session.raw_text),
        )
        game_id = int(cursor.lastrowid)

        for player_name in set(session.players):
            _get_or_create_player_id(connection, game_id, player_name)

        for message in session.messages:
            player_id = _get_or_create_player_id(connection, game_id, message.player_name)
            cursor.execute(
                "INSERT INTO messages (game_id, player_id, timestamp, day, text) VALUES (?, ?, ?, ?, ?)",
                (game_id, player_id, message.timestamp, message.day, message.text),
            )

        for vote in session.votes:
            voter_id = _get_or_create_player_id(connection, game_id, vote.voter_name)
            target_id = _get_or_create_player_id(connection, game_id, vote.target_name)
            cursor.execute(
                "INSERT INTO votes (game_id, voter_id, target_id, timestamp, text) VALUES (?, ?, ?, ?, ?)",
                (game_id, voter_id, target_id, vote.timestamp, vote.text),
            )

        for flip in session.flips:
            player_id = _get_or_create_player_id(connection, game_id, flip.player_name)
            cursor.execute(
                "INSERT INTO flips (game_id, player_id, alignment) VALUES (?, ?, ?)",
                (game_id, player_id, flip.alignment),
            )

        connection.commit()
        return game_id


def set_flip_alignment(db_path: str | Path, game_id: int, player_name: str, alignment: str) -> None:
    db_file = Path(db_path)
    if not db_file.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    with sqlite3.connect(str(db_file)) as connection:
        player_id = _get_or_create_player_id(connection, game_id, player_name)
        cursor = connection.cursor()
        cursor.execute(
            "SELECT id FROM flips WHERE game_id = ? AND player_id = ?",
            (game_id, player_id),
        )
        row = cursor.fetchone()
        if row:
            cursor.execute(
                "UPDATE flips SET alignment = ? WHERE id = ?",
                (alignment, int(row[0])),
            )
        else:
            cursor.execute(
                "INSERT INTO flips (game_id, player_id, alignment) VALUES (?, ?, ?)",
                (game_id, player_id, alignment),
            )
        connection.commit()


def ingest_log(db_path: str | Path, raw_text: str, source: str = "unknown") -> int:
    session = parse_showdown_log(raw_text, source=source)
    session.flips = extract_flips(raw_text)
    return insert_game(db_path, session)


def ingest_google_doc(db_path: str | Path, doc_url: str) -> int:
    raw_text = fetch_published_google_doc(doc_url)
    session = parse_showdown_log(raw_text, source=doc_url)
    session.flips = extract_flips(raw_text)
    return insert_game(db_path, session)


def _load_flips(connection: sqlite3.Connection, game_id: int) -> list[Flip]:
    cursor = connection.cursor()
    cursor.execute(
        "SELECT p.name, f.alignment FROM flips f JOIN players p ON f.player_id = p.id WHERE f.game_id = ?",
        (game_id,),
    )
    # Normalize player names loaded from DB so they match parser-normalized session.players
    return [Flip(player_name=canonical_player_name(row[0]), alignment=row[1]) for row in cursor.fetchall()]


def _finalize_session(session: GameSession, alias_map: dict[str, str]) -> GameSession:
    return apply_player_aliases(session, alias_map)


def load_games(db_path: str | Path) -> list[GameSession]:
    db_file = Path(db_path)
    if not db_file.exists():
        return []

    alias_map = load_alias_map(db_file)
    with sqlite3.connect(str(db_file)) as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT id, source, raw_text FROM games ORDER BY id")
        loaded: list[GameSession] = []
        for game_id, source, raw_text in cursor.fetchall():
            session = parse_showdown_log(raw_text, source=source)
            session.game_id = int(game_id)
            session.flips = _load_flips(connection, game_id)
            loaded.append(_finalize_session(session, alias_map))
        return loaded


def load_game(db_path: str | Path, game_id: int | None = None) -> GameSession | None:
    db_file = Path(db_path)
    if not db_file.exists():
        return None

    with sqlite3.connect(str(db_file)) as connection:
        cursor = connection.cursor()
        if game_id is None:
            cursor.execute("SELECT id FROM games ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            if not row:
                return None
            game_id = int(row[0])

        cursor.execute("SELECT source, raw_text FROM games WHERE id = ?", (game_id,))
        row = cursor.fetchone()
        if not row:
            return None

        source, raw_text = row
        session = parse_showdown_log(raw_text, source=source)
        session.game_id = int(game_id)
        session.flips = _load_flips(connection, game_id)
        alias_map = load_alias_map(db_file)
        return _finalize_session(session, alias_map)


def delete_game(db_path: str | Path, game_id: int) -> None:
    db_file = Path(db_path)
    if not db_file.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    with sqlite3.connect(str(db_file)) as connection:
        cursor = connection.cursor()
        # Ensure game exists
        cursor.execute("SELECT id FROM games WHERE id = ?", (game_id,))
        if not cursor.fetchone():
            raise FileNotFoundError(f"Game not found: {game_id}")

        # Delete dependent rows first
        cursor.execute("DELETE FROM player_features WHERE game_id = ?", (game_id,))
        cursor.execute("DELETE FROM flips WHERE game_id = ?", (game_id,))
        cursor.execute("DELETE FROM votes WHERE game_id = ?", (game_id,))
        cursor.execute("DELETE FROM messages WHERE game_id = ?", (game_id,))
        cursor.execute("DELETE FROM players WHERE game_id = ?", (game_id,))
        cursor.execute("DELETE FROM games WHERE id = ?", (game_id,))
        connection.commit()
