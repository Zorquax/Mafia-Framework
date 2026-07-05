from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Protocol

from ...data.models import GameSession


@dataclass
class TellFeatures:
    player_name: str
    features: dict[str, float] = field(default_factory=dict)


class BaseTell(Protocol):
    name: str

    def extract(self, session: GameSession) -> list[TellFeatures]:
        ...


def aggregate_tells(
    session: GameSession,
    tell_extractors: Iterable[BaseTell],
) -> list[TellFeatures]:
    by_player: dict[str, dict[str, float]] = {
        player_name: {} for player_name in session.players
    }

    for extractor in tell_extractors:
        for result in extractor.extract(session):
            player_features = by_player.setdefault(result.player_name, {})
            player_features.update(result.features)

    return [
        TellFeatures(player_name=player_name, features=features)
        for player_name, features in by_player.items()
    ]
