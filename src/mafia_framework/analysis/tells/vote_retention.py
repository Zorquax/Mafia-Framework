from __future__ import annotations

from collections import defaultdict

from .base import BaseTell, TellFeatures
from ...data.models import GameSession


class VoteRetentionTell(BaseTell):
    name = "vote_retention"

    def extract(self, session: GameSession) -> list[TellFeatures]:
        voter_history: dict[str, list[str]] = defaultdict(list)
        received_total: dict[str, int] = defaultdict(int)
        received_retained: dict[str, int] = defaultdict(int)

        for vote in session.votes:
            voter = vote.voter_name
            target = vote.target_name
            prior_targets = voter_history[voter]
            if target in prior_targets:
                received_retained[target] += 1
            received_total[target] += 1
            voter_history[voter].append(target)

        players = set(received_total) | set(session.players)
        results: list[TellFeatures] = []
        for player_name in players:
            total = received_total.get(player_name, 0)
            retained = received_retained.get(player_name, 0)
            ratio = float(retained) / float(total) if total > 0 else 0.0
            results.append(
                TellFeatures(
                    player_name=player_name,
                    features={"vote_retention_received_ratio": ratio},
                )
            )
        return results
