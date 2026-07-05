from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..data.models import GameSession
from ..io.player_names import names_match, player_identity_key


LogMode = Literal["lines", "votes", "both"]


@dataclass
class GameLogEntry:
    day: int
    entry_type: str
    player_name: str | None
    timestamp: str | None
    text: str


def parse_player_query(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def match_players_in_session(session: GameSession, queries: list[str]) -> set[str]:
    if not queries:
        return set()

    matched: set[str] = set()
    roster = set(session.players)
    roster_keys = {player_identity_key(name): name for name in roster if player_identity_key(name)}

    for query in queries:
        query_key = player_identity_key(query)
        if query_key and query_key in roster_keys:
            matched.add(roster_keys[query_key])
            continue
        for player in roster:
            if names_match(query, player):
                matched.add(player)
    return matched


def build_game_log(
    session: GameSession,
    player_queries: list[str],
    mode: LogMode = "both",
) -> list[GameLogEntry]:
    selected_players = match_players_in_session(session, player_queries)
    if not selected_players:
        return []

    entries: list[GameLogEntry] = []
    phase_index = 0
    phases_by_day: dict[int, list[str]] = {}
    for phase in session.phases:
        day = _phase_day(phase, phase_index)
        phases_by_day.setdefault(day, []).append(phase)
        phase_index += 1

    max_day = max(
        [
            session.messages[-1].day if session.messages else 1,
            session.votes[-1].day if session.votes else 1,
            max((e.day for e in session.events), default=1) if hasattr(session, "events") and session.events else 1,
            max(phases_by_day.keys(), default=1),
        ]
    )

    for day in range(1, max_day + 1):
        for phase in phases_by_day.get(day, []):
            entries.append(
                GameLogEntry(
                    day=day,
                    entry_type="phase",
                    player_name=None,
                    timestamp=None,
                    text=phase,
                )
            )

        if mode in {"lines", "both"}:
            for message in session.messages:
                if message.day != day:
                    continue
                if message.player_name not in selected_players:
                    continue
                entries.append(
                    GameLogEntry(
                        day=day,
                        entry_type="message",
                        player_name=message.player_name,
                        timestamp=message.timestamp,
                        text=message.text,
                    )
                )

        if mode in {"votes", "both"}:
            for vote in session.votes:
                if vote.day != day:
                    continue
                if vote.voter_name not in selected_players and vote.target_name not in selected_players:
                    continue
                if vote.action == "unvote":
                    text = vote.text or f"{vote.voter_name} unvoted"
                elif vote.action == "shift":
                    text = vote.text or f"{vote.voter_name} shifted to {vote.target_name}"
                else:
                    text = vote.text or f"{vote.voter_name} -> {vote.target_name}"
                entries.append(
                    GameLogEntry(
                        day=day,
                        entry_type="vote",
                        player_name=vote.voter_name,
                        timestamp=vote.timestamp,
                        text=text,
                    )
                )

        if mode in {"lines", "both"} and hasattr(session, "events") and session.events:
            for event in session.events:
                if event.day != day:
                    continue
                if event.player_name not in selected_players:
                    continue
                entries.append(
                    GameLogEntry(
                        day=day,
                        entry_type=event.event_type,
                        player_name=event.player_name,
                        timestamp=event.timestamp,
                        text=event.text,
                    )
                )

    return entries


def _phase_day(phase: str, fallback_index: int) -> int:
    import re

    match = re.search(r"day\s+(\d+)", phase, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return fallback_index + 1
