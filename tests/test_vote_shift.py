import unittest

from mafia_framework.analysis.tells.vote_shift import VoteShiftTell
from mafia_framework.analysis.tells.base import aggregate_tells
from mafia_framework.data.models import GameSession, Vote
from mafia_framework.io.parser import parse_showdown_log


class TestVoteShift(unittest.TestCase):

    def test_parser_detects_unvote_and_shift(self):
        raw = (
            "[17:11:41] |c:|1779642701|~|**Players (3)**: Alice, Bob, Charlie\n"
            "[00:01] |c:|1|~|Alice has voted Bob.\n"
            "[00:02] |c:|2|~|Alice has unvoted.\n"
            "[00:03] |c:|3|~|Alice has voted Bob.\n"
            "[00:04] |c:|4|~|Alice has voted Charlie.\n"
        )
        session = parse_showdown_log(raw)
        actions = [vote.action for vote in session.votes]
        self.assertIn("unvote", actions)
        self.assertIn("shift", actions)

    def test_vote_shift_tell_features(self):
        session = GameSession(
            source="test",
            raw_text="",
            players=["Alice"],
            votes=[
                Vote(voter_name="Alice", target_name="Bob", day=1, action="unvote"),
                Vote(voter_name="Alice", target_name="Charlie", day=1, action="shift"),
                Vote(voter_name="Alice", target_name="Charlie", day=1, action="shift"),
            ],
        )
        result = aggregate_tells(session, [VoteShiftTell()])[0]
        self.assertEqual(result.features["unvote_count"], 1.0)
        self.assertEqual(result.features["vote_shift_count"], 2.0)
        self.assertAlmostEqual(result.features["unvote_ratio"], 1.0 / 3.0)
