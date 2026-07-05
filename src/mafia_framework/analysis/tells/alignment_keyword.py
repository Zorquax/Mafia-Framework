from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from .base import BaseTell, TellFeatures
from ...data.models import GameSession

TOKEN_RE = re.compile(r"\b([a-zA-Z']{3,})\b")
STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "have", "from", "your", "you",
}


class AlignmentKeywordTell(BaseTell):
    name = "alignment_keyword"

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = db_path
        self._town_words: set[str] | None = None
        self._mafia_words: set[str] | None = None

    def _load_word_sets(self) -> tuple[set[str], set[str]]:
        if self._town_words is not None and self._mafia_words is not None:
            return self._town_words, self._mafia_words
        from ...services.keyword_corpus import get_alignment_word_sets
        if self.db_path and Path(self.db_path).exists():
            self._town_words, self._mafia_words = get_alignment_word_sets(self.db_path)
        else:
            self._town_words, self._mafia_words = set(), set()
        return self._town_words, self._mafia_words

    def extract(self, session: GameSession) -> list[TellFeatures]:
        town_words, mafia_words = self._load_word_sets()
        mafia_hits: dict[str, int] = defaultdict(int)
        town_hits: dict[str, int] = defaultdict(int)
        exclusive_mafia: dict[str, int] = defaultdict(int)
        exclusive_town: dict[str, int] = defaultdict(int)

        for message in session.messages:
            tokens = [token.lower() for token in TOKEN_RE.findall(message.text) if token.lower() not in STOPWORDS]
            player = message.player_name
            for token in tokens:
                if token in mafia_words:
                    mafia_hits[player] += 1
                    if token not in town_words:
                        exclusive_mafia[player] += 1
                if token in town_words:
                    town_hits[player] += 1
                    if token not in mafia_words:
                        exclusive_town[player] += 1

        players = set(mafia_hits) | set(town_hits) | set(session.players)
        results: list[TellFeatures] = []
        for player_name in players:
            mafia_count = float(mafia_hits.get(player_name, 0))
            town_count = float(town_hits.get(player_name, 0))
            ex_mafia = float(exclusive_mafia.get(player_name, 0))
            ex_town = float(exclusive_town.get(player_name, 0))
            total = ex_mafia + ex_town
            bias_ratio = (ex_mafia - ex_town) / total if total > 0 else 0.0
            results.append(
                TellFeatures(
                    player_name=player_name,
                    features={
                        "alignment_mafia_word_hits": mafia_count,
                        "alignment_town_word_hits": town_count,
                        "alignment_exclusive_mafia_words": ex_mafia,
                        "alignment_exclusive_town_words": ex_town,
                        "alignment_word_bias_ratio": bias_ratio,
                    },
                )
            )
        return results
