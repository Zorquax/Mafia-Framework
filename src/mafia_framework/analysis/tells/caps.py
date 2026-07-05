from __future__ import annotations

from collections import defaultdict

from .base import BaseTell, TellFeatures
from ...data.models import GameSession


class CapsTell(BaseTell):
    name = "caps"

    def extract(self, session: GameSession) -> list[TellFeatures]:
        upper_counts: dict[str, int] = defaultdict(int)
        letter_counts: dict[str, int] = defaultdict(int)

        for message in session.messages:
            for char in message.text:
                if not char.isalpha():
                    continue
                letter_counts[message.player_name] += 1
                if char.isupper():
                    upper_counts[message.player_name] += 1

        players = set(letter_counts) | set(session.players)
        results: list[TellFeatures] = []
        for player_name in players:
            letters = letter_counts.get(player_name, 0)
            ratio = float(upper_counts.get(player_name, 0)) / float(letters) if letters else 0.0
            results.append(
                TellFeatures(
                    player_name=player_name,
                    features={"caps_ratio": ratio},
                )
            )
        return results
