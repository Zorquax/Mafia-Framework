import re
from typing import Iterable

from ..io.parser import split_combined_system_lines
from ..io.player_names import canonical_player_name, player_identity_key
from .models import Flip

FLIP_RE = re.compile(
    r"(?m)^(?P<player>[^\n]+?)(?:['’]\s*s)?\s+role was\s+(?P<role>[^\n\.]+)",
    re.IGNORECASE,
)

MAFIA_KEYWORDS = [
    "mafia",
    "mafioso",
    "godfather",
    "goon",
    "framer",
    "consigliere",
    "blackmailer",
    "janitor",
    "ambusher",
    "cultist",
    "scum",
]
TOWN_KEYWORDS = [
    "Town",
    "Villager",
    "Vanilla Townie",
]
NEUTRAL_KEYWORDS = [
    "neutral",
    "serial killer",
    "sk",
    "executioner",
    "jester",
    "survivor",
]


def normalize_alignment(role: str) -> str:
    normalized = role.lower().strip()
    if not normalized:
        return "unknown"

    for keyword in MAFIA_KEYWORDS:
        if keyword in normalized:
            return "mafia"

    for keyword in NEUTRAL_KEYWORDS:
        if keyword in normalized:
            return "neutral"

    return "town"


def extract_flips(raw_text: str) -> list[Flip]:
    results: list[Flip] = []
    for match in FLIP_RE.finditer(split_combined_system_lines(raw_text)):
        # Normalize player names the same way we normalize chat/player lists
        # so flips match session.player entries (remove prefix symbols, trim).
        player = canonical_player_name(match.group("player"))
        alignment = normalize_alignment(match.group("role"))
        results.append(Flip(player_name=player, alignment=alignment))
    return results
