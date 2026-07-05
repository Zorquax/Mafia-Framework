from __future__ import annotations

from typing import Iterable

import pandas as pd

from ..analysis.tells import aggregate_tells
from ..analysis.tells.registry import (
    DAY_ONE_FEATURE_NAMES,
    FEATURE_NAMES,
    day_one_tells,
    default_tells,
)
from ..data.models import GameSession
from ..io.ingestion import load_game, load_games
from ..models.feature_engineering import build_feature_dataframe, build_training_dataset
from .game_service import resolve_flip_map


def compute_tell_dataframe(
    session: GameSession,
    *,
    day_one: bool = False,
    db_path: str | None = None,
) -> pd.DataFrame:
    extractors = day_one_tells(db_path) if day_one else default_tells(db_path)
    feature_names = DAY_ONE_FEATURE_NAMES if day_one else FEATURE_NAMES
    tell_results = aggregate_tells(session, extractors)
    return build_feature_dataframe(tell_results, feature_names=feature_names)


def compute_all_game_tells(
    db_path: str,
    *,
    day_one: bool = False,
) -> pd.DataFrame:
    sessions = load_games(db_path)
    frames: list[pd.DataFrame] = []
    for session in sessions:
        frame = compute_tell_dataframe(session, day_one=day_one, db_path=db_path)
        if frame.empty:
            continue
        frame.insert(0, "game_id", session.game_id)
        resolved = resolve_flip_map(session)
        frame["alignment"] = frame["player_name"].map(resolved)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_labeled_training_frame(
    db_path: str,
    *,
    day_one: bool = False,
    exclude_neutral_games: bool = False,
) -> tuple[pd.DataFrame, list[str], list[int]]:
    sessions = load_games(db_path)
    extractors = day_one_tells(db_path) if day_one else default_tells(db_path)
    feature_names = DAY_ONE_FEATURE_NAMES if day_one else FEATURE_NAMES
    return build_training_dataset(
        sessions,
        extractors,
        feature_names=feature_names,
        binary_only=True,
        exclude_neutral_games=exclude_neutral_games,
    )


def get_session_tells(db_path: str, game_id: int, *, day_one: bool = False) -> pd.DataFrame:
    session = load_game(db_path, game_id)
    if session is None:
        return pd.DataFrame()
    return compute_tell_dataframe(session, day_one=day_one, db_path=db_path)


def list_canonical_players(db_path: str) -> list[str]:
    from ..data.aliases import load_alias_map, resolve_name

    alias_map = load_alias_map(db_path)
    players: set[str] = set()
    for session in load_games(db_path):
        for player in session.players:
            players.add(resolve_name(player, alias_map) if alias_map else player)
    return sorted(players)
