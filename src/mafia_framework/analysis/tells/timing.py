from __future__ import annotations

from collections import defaultdict

from .base import BaseTell, TellFeatures
from ...data.models import GameSession


def _timestamp_to_seconds(timestamp: str | None) -> int | None:
    if not timestamp:
        return None
    parts = timestamp.split(":")
    try:
        if len(parts) == 2:
            hours, minutes = (int(part) for part in parts)
            return hours * 3600 + minutes * 60
        if len(parts) == 3:
            hours, minutes, seconds = (int(part) for part in parts)
            return hours * 3600 + minutes * 60 + seconds
    except ValueError:
        return None
    return None


class TimingTell(BaseTell):
    name = "timing"

    def extract(self, session: GameSession) -> list[TellFeatures]:
        times_by_player: dict[str, list[int]] = defaultdict(list)

        for message in session.messages:
            seconds = _timestamp_to_seconds(message.timestamp)
            if seconds is not None:
                times_by_player[message.player_name].append(seconds)

        players = set(times_by_player) | set(session.players)
        results: list[TellFeatures] = []
        for player_name in players:
            times = sorted(times_by_player.get(player_name, []))
            gaps = [later - earlier for earlier, later in zip(times, times[1:]) if later >= earlier]
            avg = float(sum(gaps)) / float(len(gaps)) if gaps else 0.0
            results.append(
                TellFeatures(
                    player_name=player_name,
                    features={"avg_response_time": avg},
                )
            )
        return results
