from __future__ import annotations

import re
from collections import defaultdict

from .base import BaseTell, TellFeatures
from ...data.models import GameSession


DIRECTED_RE = re.compile(r"(^|\s)[@+][A-Za-z0-9_][A-Za-z0-9_\-]*")


class DirectedTalkTell(BaseTell):
    name = "directed_talk"

    def extract(self, session: GameSession) -> list[TellFeatures]:
        total_counts: dict[str, int] = defaultdict(int)
        directed_counts: dict[str, int] = defaultdict(int)

        for message in session.messages:
            total_counts[message.player_name] += 1
            if DIRECTED_RE.search(message.text):
                directed_counts[message.player_name] += 1

        players = set(total_counts) | set(session.players)
        results: list[TellFeatures] = []
        for player_name in players:
            total = total_counts.get(player_name, 0)
            ratio = float(directed_counts.get(player_name, 0)) / float(total) if total else 0.0
            results.append(
                TellFeatures(
                    player_name=player_name,
                    features={"directed_talk_ratio": ratio},
                )
            )
        return results
