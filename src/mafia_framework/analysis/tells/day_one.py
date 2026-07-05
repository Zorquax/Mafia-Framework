from __future__ import annotations

from collections import defaultdict

from .base import BaseTell, TellFeatures
from ...data.models import GameSession


class DayOneTell(BaseTell):
    name = "day_one"

    def extract(self, session: GameSession) -> list[TellFeatures]:
        total_counts: dict[str, int] = defaultdict(int)
        day_one_counts: dict[str, int] = defaultdict(int)

        for message in session.messages:
            total_counts[message.player_name] += 1
            if message.day == 1:
                day_one_counts[message.player_name] += 1

        players = set(total_counts) | set(session.players)
        results: list[TellFeatures] = []
        for player_name in players:
            total = total_counts.get(player_name, 0)
            ratio = float(day_one_counts.get(player_name, 0)) / float(total) if total else 0.0
            results.append(
                TellFeatures(
                    player_name=player_name,
                    features={"day_one_ratio": ratio},
                )
            )
        return results
