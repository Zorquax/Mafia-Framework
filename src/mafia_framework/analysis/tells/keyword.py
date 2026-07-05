from __future__ import annotations

import re
from collections import defaultdict

from .base import BaseTell, TellFeatures
from ...data.models import GameSession


TOKEN_RE = re.compile(r"\b[a-zA-Z']+\b")
SCUM_WORDS = {"scum", "mafia", "wolf", "partner", "bussing", "bus", "fake"}
TOWN_WORDS = {"town", "townie", "villager", "solve", "solving", "read", "reads"}


class KeywordTell(BaseTell):
    name = "keyword"

    def extract(self, session: GameSession) -> list[TellFeatures]:
        scores: dict[str, float] = defaultdict(float)

        for message in session.messages:
            tokens = {token.lower() for token in TOKEN_RE.findall(message.text)}
            scores[message.player_name] += float(len(tokens & SCUM_WORDS))
            scores[message.player_name] -= float(len(tokens & TOWN_WORDS)) * 0.5

        players = set(scores) | set(session.players)
        return [
            TellFeatures(
                player_name=player_name,
                features={"keyword_score": float(scores.get(player_name, 0.0))},
            )
            for player_name in players
        ]
