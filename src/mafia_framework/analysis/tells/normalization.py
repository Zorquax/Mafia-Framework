from __future__ import annotations

import statistics
from collections import defaultdict

from .base import BaseTell, TellFeatures
from ...data.models import GameSession


def _zscore(value: float, values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = statistics.mean(values)
    stdev = statistics.pstdev(values)
    if stdev == 0.0:
        return 0.0
    return (value - mean) / stdev


class NormalizationTell(BaseTell):
    name = "normalization"

    def extract(self, session: GameSession) -> list[TellFeatures]:
        line_counts: dict[str, int] = defaultdict(int)
        cast_counts: dict[str, int] = defaultdict(int)
        receive_counts: dict[str, int] = defaultdict(int)

        for message in session.messages:
            line_counts[message.player_name] += 1

        for vote in session.votes:
            cast_counts[vote.voter_name] += 1
            receive_counts[vote.target_name] += 1

        players = set(line_counts) | set(cast_counts) | set(receive_counts) | set(session.players)
        if not players:
            return []

        total_lines = sum(line_counts.values()) or 1
        total_casts = sum(cast_counts.values()) or 1
        total_received = sum(receive_counts.values()) or 1

        line_values = [float(line_counts.get(p, 0)) for p in players]
        cast_values = [float(cast_counts.get(p, 0)) for p in players]

        results: list[TellFeatures] = []
        for player_name in players:
            line_count = float(line_counts.get(player_name, 0))
            cast_count = float(cast_counts.get(player_name, 0))
            received_count = float(receive_counts.get(player_name, 0))
            results.append(
                TellFeatures(
                    player_name=player_name,
                    features={
                        "line_count_share": line_count / total_lines,
                        "vote_cast_share": cast_count / total_casts,
                        "vote_received_share": received_count / total_received,
                        "line_count_zscore": _zscore(line_count, line_values),
                        "vote_cast_zscore": _zscore(cast_count, cast_values),
                    },
                )
            )
        return results
