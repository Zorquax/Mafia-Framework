from __future__ import annotations

from copy import deepcopy

from .base import BaseTell, TellFeatures
from ...data.models import GameSession


class DayScopedTell(BaseTell):
    """Run an inner tell against messages/votes from a single day only."""

    def __init__(self, inner: BaseTell, day: int = 1, prefix: str = "d1_") -> None:
        self.inner = inner
        self.day = day
        self.prefix = prefix
        self.name = f"{prefix.rstrip('_')}_{inner.name}"

    def extract(self, session: GameSession) -> list[TellFeatures]:
        filtered = GameSession(
            source=session.source,
            raw_text=session.raw_text,
            players=list(session.players),
            messages=[message for message in session.messages if message.day == self.day],
            votes=[vote for vote in session.votes if vote.day == self.day],
            flips=list(session.flips),
            phases=list(session.phases),
            game_id=session.game_id,
            metadata=deepcopy(session.metadata),
        )
        results = self.inner.extract(filtered)
        remapped: list[TellFeatures] = []
        for result in results:
            features = {f"{self.prefix}{key}": value for key, value in result.features.items()}
            remapped.append(TellFeatures(player_name=result.player_name, features=features))
        return remapped
