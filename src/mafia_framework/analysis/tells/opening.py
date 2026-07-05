from collections import defaultdict

from .base import BaseTell, TellFeatures
from ...data.models import GameSession


class OpeningTell(BaseTell):
    name = "opening"

    def __init__(self, opening_limit: int = 5) -> None:
        self.opening_limit = opening_limit

    def extract(self, session: GameSession) -> list[TellFeatures]:
        per_player_counts: dict[str, int] = defaultdict(int)
        per_player_seen: dict[str, int] = defaultdict(int)

        for message in session.messages:
            player = message.player_name
            if per_player_seen[player] < self.opening_limit:
                per_player_counts[player] += 1
                per_player_seen[player] += 1

        return [
            TellFeatures(
                player_name=player_name,
                features={"opening_line_count": float(count)},
            )
            for player_name, count in per_player_counts.items()
        ]
