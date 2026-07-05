"""Tell registry and feature schema versioning."""

from __future__ import annotations

from pathlib import Path

from .alignment_keyword import AlignmentKeywordTell
from .base import BaseTell
from .caps import CapsTell
from .day_scoped import DayScopedTell
from .directed_talk import DirectedTalkTell
from .keyword import KeywordTell
from .line_count import LineCountTell
from .normalization import NormalizationTell
from .opening import OpeningTell
from .timing import TimingTell
from .vote_count import VoteCountTell
from .vote_retention import VoteRetentionTell
from .vote_shift import VoteShiftTell

FEATURE_SET_VERSION = "2026-06-01-v2"
DAY_ONE_FEATURE_SET_VERSION = "2026-06-01-d1-v2"

FEATURE_NAMES: list[str] = [
    "line_count",
    "opening_line_count",
    "day_one_ratio",
    "directed_talk_ratio",
    "caps_ratio",
    "keyword_score",
    "vote_cast_count",
    "vote_received_count",
    "avg_response_time",
    "line_count_share",
    "vote_cast_share",
    "vote_received_share",
    "line_count_zscore",
    "vote_cast_zscore",
    "line_count_day_std",
    "late_line_ratio",
    "vote_retention_received_ratio",
    "unvote_count",
    "vote_shift_count",
    "unvote_ratio",
    "vote_shift_ratio",
    "alignment_mafia_word_hits",
    "alignment_town_word_hits",
    "alignment_exclusive_mafia_words",
    "alignment_exclusive_town_words",
    "alignment_word_bias_ratio",
]

DAY_ONE_BASE_FEATURES: list[str] = [
    "line_count",
    "opening_line_count",
    "directed_talk_ratio",
    "caps_ratio",
    "keyword_score",
    "vote_cast_count",
    "vote_received_count",
    "avg_response_time",
    "line_count_share",
    "vote_cast_share",
    "vote_received_share",
    "line_count_zscore",
    "vote_cast_zscore",
    "vote_retention_received_ratio",
    "unvote_count",
    "vote_shift_count",
    "unvote_ratio",
    "vote_shift_ratio",
    "alignment_mafia_word_hits",
    "alignment_town_word_hits",
    "alignment_exclusive_mafia_words",
    "alignment_exclusive_town_words",
    "alignment_word_bias_ratio",
]

DAY_ONE_FEATURE_NAMES: list[str] = [f"d1_{name}" for name in DAY_ONE_BASE_FEATURES]

OPENING_LINES_PER_PLAYER = 5


def default_tells(db_path: str | Path | None = None) -> list[BaseTell]:
    from .day_one import DayOneTell
    from .line_variation import LineVariationTell

    return [
        LineCountTell(),
        OpeningTell(opening_limit=OPENING_LINES_PER_PLAYER),
        DayOneTell(),
        DirectedTalkTell(),
        CapsTell(),
        KeywordTell(),
        VoteCountTell(),
        TimingTell(),
        NormalizationTell(),
        LineVariationTell(),
        VoteRetentionTell(),
        VoteShiftTell(),
        AlignmentKeywordTell(db_path=db_path),
    ]


def day_one_tells(db_path: str | Path | None = None) -> list[BaseTell]:
    return [
        DayScopedTell(LineCountTell()),
        DayScopedTell(OpeningTell(opening_limit=OPENING_LINES_PER_PLAYER)),
        DayScopedTell(DirectedTalkTell()),
        DayScopedTell(CapsTell()),
        DayScopedTell(KeywordTell()),
        DayScopedTell(VoteCountTell()),
        DayScopedTell(TimingTell()),
        DayScopedTell(NormalizationTell()),
        DayScopedTell(VoteRetentionTell()),
        DayScopedTell(VoteShiftTell()),
        DayScopedTell(AlignmentKeywordTell(db_path=db_path)),
    ]
