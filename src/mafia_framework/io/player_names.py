from __future__ import annotations

import re

PREFIX_SYMBOL_RE = re.compile(r"^[+%@]+")


def normalize_player_name(username: str) -> str:
    """Strip leading address symbols and whitespace."""
    return PREFIX_SYMBOL_RE.sub("", username).strip()


def player_identity_key(name: str) -> str:
    """Identity from English letters/digits only (case/decorators/spacing ignored)."""
    cleaned = normalize_player_name(name)
    return re.sub(r"[^a-zA-Z0-9]", "", cleaned).lower()


def canonical_player_name(name: str) -> str:
    """Store/display name using English letters and digits only."""
    cleaned = normalize_player_name(name)
    letters = re.sub(r"[^a-zA-Z0-9]", "", cleaned)
    return letters if letters else cleaned


def names_match(left: str, right: str) -> bool:
    left_key = player_identity_key(left)
    right_key = player_identity_key(right)
    return bool(left_key) and left_key == right_key


def resolve_to_roster(name: str, identity_to_canonical: dict[str, str]) -> str:
    """Map any name variant to the roster canonical name for its identity."""
    key = player_identity_key(name)
    if not key:
        return canonical_player_name(name)
    if key in identity_to_canonical:
        return identity_to_canonical[key]
    canonical = canonical_player_name(name)
    identity_to_canonical[key] = canonical
    return canonical
