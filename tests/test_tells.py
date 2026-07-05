import unittest

from mafia_framework.analysis.tells.base import aggregate_tells
from mafia_framework.analysis.tells.caps import CapsTell
from mafia_framework.analysis.tells.day_one import DayOneTell
from mafia_framework.analysis.tells.directed_talk import DirectedTalkTell
from mafia_framework.analysis.tells.keyword import KeywordTell
from mafia_framework.analysis.tells.line_count import LineCountTell
from mafia_framework.analysis.tells.line_variation import LineVariationTell
from mafia_framework.analysis.tells.normalization import NormalizationTell
from mafia_framework.analysis.tells.opening import OpeningTell
from mafia_framework.analysis.tells.registry import FEATURE_NAMES, default_tells
from mafia_framework.analysis.tells.vote_count import VoteCountTell
from mafia_framework.analysis.tells.timing import TimingTell
from mafia_framework.analysis.tells.vote_retention import VoteRetentionTell
from mafia_framework.data.models import GameSession, Message, Vote
from mafia_framework.models.feature_engineering import build_feature_dataframe


class TestTells(unittest.TestCase):

    def test_tell_extractors(self):
        session = GameSession(
            source="test",
            raw_text="",
            messages=[
                Message(player_name="Alice", text="I think we should vote Bob", timestamp="00:01", day=1),
                Message(player_name="Bob", text="That seems scum", timestamp="00:02", day=1),
                Message(player_name="Alice", text="I AGREE @Bob", timestamp="00:03", day=1),
                Message(player_name="Charlie", text="HELLO everyone", timestamp="00:04", day=2),
                Message(player_name="Alice", text="Hello +Bob", timestamp="00:05", day=2),
            ],
            votes=[Vote(voter_name="Alice", target_name="Bob", timestamp="00:03", text="vote Bob", day=1)],
        )
        extractors = [
            LineCountTell(),
            OpeningTell(opening_limit=5),
            DayOneTell(),
            DirectedTalkTell(),
            CapsTell(),
            KeywordTell(),
            VoteCountTell(),
            TimingTell(),
            NormalizationTell(),
            LineVariationTell(),
            VoteRetentionTell(),
        ]
        results = aggregate_tells(session, extractors)

        alice = next(r for r in results if r.player_name == "Alice")
        bob = next(r for r in results if r.player_name == "Bob")
        charlie = next(r for r in results if r.player_name == "Charlie")

        self.assertEqual(alice.features["line_count"], 3.0)
        self.assertEqual(bob.features["line_count"], 1.0)
        self.assertEqual(charlie.features["line_count"], 1.0)
        self.assertEqual(alice.features["opening_line_count"], 3.0)
        self.assertEqual(bob.features["opening_line_count"], 1.0)
        self.assertEqual(charlie.features["opening_line_count"], 1.0)
        self.assertEqual(alice.features["day_one_ratio"], 2.0 / 3.0)
        self.assertEqual(bob.features["day_one_ratio"], 1.0)
        self.assertEqual(charlie.features["day_one_ratio"], 0.0)
        self.assertEqual(alice.features["directed_talk_ratio"], 2.0 / 3.0)
        self.assertEqual(bob.features["directed_talk_ratio"], 0.0)
        self.assertEqual(charlie.features["directed_talk_ratio"], 0.0)
        self.assertEqual(alice.features["vote_cast_count"], 1.0)
        self.assertEqual(bob.features["vote_received_count"], 1.0)
        self.assertAlmostEqual(charlie.features["caps_ratio"], 5.0 / 13.0, places=4)
        self.assertEqual(alice.features["avg_response_time"], 120.0)
        self.assertAlmostEqual(alice.features["line_count_share"], 3.0 / 5.0)
        self.assertGreater(alice.features["line_count_zscore"], 0.0)

    def test_opening_per_player_not_global(self):
        session = GameSession(
            source="test",
            raw_text="",
            messages=[
                Message(player_name="Alice", text="a", day=1),
                Message(player_name="Alice", text="b", day=1),
                Message(player_name="Alice", text="c", day=1),
                Message(player_name="Alice", text="d", day=1),
                Message(player_name="Alice", text="e", day=1),
                Message(player_name="Alice", text="f", day=1),
                Message(player_name="Bob", text="only bob", day=1),
            ],
        )
        results = aggregate_tells(session, [OpeningTell(opening_limit=5)])
        alice = next(r for r in results if r.player_name == "Alice")
        bob = next(r for r in results if r.player_name == "Bob")
        self.assertEqual(alice.features["opening_line_count"], 5.0)
        self.assertEqual(bob.features["opening_line_count"], 1.0)

    def test_default_tells_feature_schema(self):
        session = GameSession(
            source="test",
            raw_text="",
            messages=[Message(player_name="Alice", text="hello", day=1)],
        )
        results = aggregate_tells(session, default_tells())
        frame = build_feature_dataframe(results, feature_names=FEATURE_NAMES)
        self.assertEqual(list(frame.columns), ["player_name", *FEATURE_NAMES])
