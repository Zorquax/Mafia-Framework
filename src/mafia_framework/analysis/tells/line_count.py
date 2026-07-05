from __future__ import annotations

from collections import defaultdict

from .base import BaseTell, TellFeatures
from ...data.models import GameSession


class LineCountTell(BaseTell):
    name = "line_count"

    def extract(self, session: GameSession) -> list[TellFeatures]:
        counts: dict[str, int] = defaultdict(int)
        for message in session.messages:
            counts[message.player_name] += 1

        players = set(counts) | set(session.players)
        return [
            TellFeatures(
                player_name=player_name,
                features={"line_count": float(counts.get(player_name, 0))},
            )
            for player_name in players
        ]
