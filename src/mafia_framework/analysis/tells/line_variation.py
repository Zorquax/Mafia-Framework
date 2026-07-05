from __future__ import annotations

import statistics
from collections import defaultdict

from .base import BaseTell, TellFeatures
from ...data.models import GameSession


class LineVariationTell(BaseTell):
    name = "line_variation"

    def extract(self, session: GameSession) -> list[TellFeatures]:
        by_player_day: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))

        for message in session.messages:
            by_player_day[message.player_name][message.day] += 1

        results: list[TellFeatures] = []
        for player_name, day_counts in by_player_day.items():
            daily = list(day_counts.values())
            total = sum(daily)
            if total == 0:
                continue

            day_std = float(statistics.pstdev(daily)) if len(daily) > 1 else 0.0
            sorted_days = sorted(day_counts.keys())
            median_day = sorted_days[len(sorted_days) // 2]
            early = sum(count for day, count in day_counts.items() if day <= median_day)
            late = total - early
            late_ratio = float(late) / float(early) if early > 0 else 0.0

            results.append(
                TellFeatures(
                    player_name=player_name,
                    features={
                        "line_count_day_std": day_std,
                        "late_line_ratio": late_ratio,
                    },
                )
            )
        return results
