from __future__ import annotations

from collections import defaultdict

from .base import BaseTell, TellFeatures
from ...data.models import GameSession


class VoteCountTell(BaseTell):
    name = "vote_count"

    def extract(self, session: GameSession) -> list[TellFeatures]:
        cast_counts: dict[str, int] = defaultdict(int)
        received_counts: dict[str, int] = defaultdict(int)

        for vote in session.votes:
            if vote.action != "unvote":
                cast_counts[vote.voter_name] += 1
            if vote.target_name:
                received_counts[vote.target_name] += 1

        players = set(cast_counts) | set(received_counts) | set(session.players)
        results: list[TellFeatures] = []
        for player_name in players:
            results.append(
                TellFeatures(
                    player_name=player_name,
                    features={
                        "vote_cast_count": float(cast_counts.get(player_name, 0)),
                        "vote_received_count": float(received_counts.get(player_name, 0)),
                    },
                )
            )
        return results
