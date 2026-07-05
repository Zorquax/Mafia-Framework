from collections import defaultdict

from .base import BaseTell, TellFeatures
from ...data.models import GameSession


class VoteShiftTell(BaseTell):
    name = "vote_shift"

    def extract(self, session: GameSession) -> list[TellFeatures]:
        unvotes: dict[str, int] = defaultdict(int)
        shifts: dict[str, int] = defaultdict(int)

        for vote in session.votes:
            if vote.action == "unvote":
                unvotes[vote.voter_name] += 1
            elif vote.action == "shift":
                shifts[vote.voter_name] += 1

        players = set(unvotes) | set(shifts) | set(session.players)
        results: list[TellFeatures] = []
        for player_name in players:
            unvote_count = float(unvotes.get(player_name, 0))
            shift_count = float(shifts.get(player_name, 0))
            total = unvote_count + shift_count
            unvote_ratio = unvote_count / total if total > 0 else 0.0
            shift_ratio = shift_count / total if total > 0 else 0.0
            results.append(
                TellFeatures(
                    player_name=player_name,
                    features={
                        "unvote_count": unvote_count,
                        "vote_shift_count": shift_count,
                        "unvote_ratio": unvote_ratio,
                        "vote_shift_ratio": shift_ratio,
                    },
                )
            )
        return results
