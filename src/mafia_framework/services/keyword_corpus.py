from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from ..io.player_names import names_match
from ..io.ingestion import load_games
from .game_service import resolve_flip_map

TOKEN_RE = re.compile(r"\b([a-zA-Z']{3,})\b")
STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "have", "from", "your", "you",
    "are", "was", "but", "not", "all", "can", "just", "like", "what", "when",
    "who", "how", "why", "its", "it's", "imo", "tbh", "lol", "vote", "unvote",
}


@dataclass
class AlignmentWordStat:
    word: str
    town_count: int
    mafia_count: int
    town_rate: float
    mafia_rate: float
    alignment_bias: str


def build_alignment_word_stats(db_path: str | Path, *, min_total: int = 5) -> list[AlignmentWordStat]:
    town_counts: dict[str, int] = defaultdict(int)
    mafia_counts: dict[str, int] = defaultdict(int)

    for session in load_games(db_path):
        flip_map = resolve_flip_map(session)
        for message in session.messages:
            alignment = flip_map.get(message.player_name)
            if alignment not in {"town", "mafia"}:
                continue
            tokens = [token.lower() for token in TOKEN_RE.findall(message.text)]
            for token in tokens:
                if token in STOPWORDS:
                    continue
                if alignment == "town":
                    town_counts[token] += 1
                else:
                    mafia_counts[token] += 1

    total_town = sum(town_counts.values()) or 1
    total_mafia = sum(mafia_counts.values()) or 1
    words = set(town_counts) | set(mafia_counts)

    stats: list[AlignmentWordStat] = []
    for word in words:
        town = town_counts[word]
        mafia = mafia_counts[word]
        if town + mafia < min_total:
            continue
        town_rate = town / total_town
        mafia_rate = mafia / total_mafia
        if mafia_rate >= town_rate * 1.5 and mafia >= 3:
            bias = "mafia"
        elif town_rate >= mafia_rate * 1.5 and town >= 3:
            bias = "town"
        else:
            continue
        stats.append(
            AlignmentWordStat(
                word=word,
                town_count=town,
                mafia_count=mafia,
                town_rate=town_rate,
                mafia_rate=mafia_rate,
                alignment_bias=bias,
            )
        )
    return sorted(stats, key=lambda row: abs(row.mafia_rate - row.town_rate), reverse=True)


def get_alignment_word_sets(db_path: str | Path) -> tuple[set[str], set[str]]:
    stats = build_alignment_word_stats(db_path)
    mafia_words = {row.word for row in stats if row.alignment_bias == "mafia"}
    town_words = {row.word for row in stats if row.alignment_bias == "town"}
    return town_words, mafia_words


def player_alignment_word_usage(
    db_path: str | Path,
    player_name: str,
) -> list[dict[str, int | str]]:
    from ..data.aliases import load_alias_map, resolve_name

    alias_map = load_alias_map(db_path)
    canonical = resolve_name(player_name, alias_map) if alias_map else player_name
    town_words, mafia_words = get_alignment_word_sets(db_path)
    usage: dict[str, dict[str, int | str]] = {}

    for session in load_games(db_path):
        flip_map = resolve_flip_map(session)
        for message in session.messages:
            resolved = resolve_name(message.player_name, alias_map) if alias_map else message.player_name
            if resolved != canonical and not names_match(resolved, canonical):
                continue
            alignment = flip_map.get(message.player_name, "unknown")
            for token in TOKEN_RE.findall(message.text):
                word = token.lower()
                if word in STOPWORDS:
                    continue
                if word not in town_words and word not in mafia_words:
                    continue
                bucket = usage.setdefault(
                    word,
                    {"word": word, "town_uses": 0, "mafia_uses": 0, "bias": "neutral"},
                )
                if alignment == "town":
                    bucket["town_uses"] = int(bucket["town_uses"]) + 1
                elif alignment == "mafia":
                    bucket["mafia_uses"] = int(bucket["mafia_uses"]) + 1
                if word in mafia_words:
                    bucket["bias"] = "mafia"
                elif word in town_words:
                    bucket["bias"] = "town"

    rows = list(usage.values())
    rows.sort(key=lambda row: int(row["town_uses"]) + int(row["mafia_uses"]), reverse=True)
    return rows


from ..io.player_names import names_match