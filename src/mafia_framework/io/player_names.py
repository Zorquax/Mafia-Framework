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
    """Store/display name exactly as it appears (spaces, punctuation,
    unicode and all) -- only a leading rank/prefix symbol and surrounding
    whitespace are stripped. This has to match what's actually stored as a
    player's raw name in the database, or the same person ends up filed
    under two different spellings (e.g. "Lucid Daydream" vs
    "LucidDaydream"). Use player_identity_key/names_match, not this, for
    identity comparisons -- this is for storage/display only.
    """
    return normalize_player_name(name)


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
