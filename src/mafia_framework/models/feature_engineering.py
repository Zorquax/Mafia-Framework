from __future__ import annotations

from typing import Iterable

import pandas as pd

from ..analysis.tells.base import TellFeatures, aggregate_tells
from ..analysis.tells.registry import FEATURE_NAMES
from ..data.models import GameSession

BINARY_ALIGNMENTS = frozenset({"town", "mafia"})


def build_feature_dataframe(
    tell_results: Iterable[TellFeatures],
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    columns = feature_names if feature_names is not None else FEATURE_NAMES
    records = []
    for tell in tell_results:
        record = {"player_name": tell.player_name}
        record.update(tell.features)
        records.append(record)

    if not records:
        return pd.DataFrame(columns=["player_name", *columns])

    frame = pd.DataFrame(records).fillna(0.0)
    for column in columns:
        if column not in frame.columns:
            frame[column] = 0.0
    return frame[["player_name", *columns]]


def build_training_dataset(
    sessions: Iterable[GameSession],
    tell_extractors: Iterable,
    *,
    feature_names: list[str] | None = None,
    binary_only: bool = True,
    exclude_neutral_games: bool = False,
) -> tuple[pd.DataFrame, list[str], list[int]]:
    columns = feature_names if feature_names is not None else FEATURE_NAMES
    rows: list[dict[str, float]] = []
    labels: list[str] = []
    groups: list[int] = []

    for session_index, session in enumerate(sessions):
        game_id = session.game_id if session.game_id is not None else session_index
        flip_map = {flip.player_name: flip.alignment for flip in session.flips}
        if not flip_map:
            continue

        if exclude_neutral_games and any(alignment == "neutral" for alignment in flip_map.values()):
            continue

        tell_results = aggregate_tells(session, tell_extractors)
        dataframe = build_feature_dataframe(tell_results, feature_names=columns)

        player_names = set(session.players) if session.players else set(dataframe["player_name"].unique())
        inferred_town = player_names - set(flip_map.keys())
        for player_name in inferred_town:
            flip_map[player_name] = "town"

        for _, row in dataframe.iterrows():
            player_name = row["player_name"]
            alignment = flip_map.get(player_name)
            if alignment is None:
                continue
            if binary_only and alignment not in BINARY_ALIGNMENTS:
                continue
            labels.append(alignment)
            groups.append(game_id)
            rows.append(row.drop(labels=["player_name"]).to_dict())

    if not rows:
        return pd.DataFrame(columns=columns), [], []

    return pd.DataFrame(rows, columns=columns).fillna(0.0), labels, groups
