import json
import tempfile
import unittest
from pathlib import Path

from mafia_framework.analysis.tells.day_scoped import DayScopedTell
from mafia_framework.analysis.tells.line_count import LineCountTell
from mafia_framework.analysis.tells.registry import DAY_ONE_FEATURE_NAMES, day_one_tells
from mafia_framework.data.models import GameSession, Message, Vote, initialize_database
from mafia_framework.io.ingestion import ingest_log
from mafia_framework.services.format_service import format_percent, format_probability
from mafia_framework.services.game_service import (
    find_undefined_players,
    list_game_summaries,
    resolve_flip_map,
    set_game_display_name,
)


class TestFormatService(unittest.TestCase):

    def test_format_probability(self):
        formatted = format_probability({"town": 0.6234, "mafia": 0.3766})
        self.assertEqual(formatted["town"], "62.34%")
        self.assertEqual(formatted["mafia"], "37.66%")

    def test_format_percent(self):
        self.assertEqual(format_percent(0.5), "50.00%")


class TestDayScopedTell(unittest.TestCase):

    def test_day_scoped_filters_messages(self):
        session = GameSession(
            source="test",
            raw_text="",
            messages=[
                Message(player_name="Alice", text="d1", day=1),
                Message(player_name="Alice", text="d2", day=2),
            ],
            votes=[
                Vote(voter_name="Alice", target_name="Bob", day=1),
                Vote(voter_name="Alice", target_name="Bob", day=2),
            ],
        )
        result = DayScopedTell(LineCountTell()).extract(session)[0]
        self.assertEqual(result.features["d1_line_count"], 1.0)

    def test_day_one_registry(self):
        self.assertEqual(len(day_one_tells()), 11)
        self.assertTrue(all(name.startswith("d1_") for name in DAY_ONE_FEATURE_NAMES))


class TestGameService(unittest.TestCase):

    def test_display_name_and_undefined(self):
        raw_text = (
            "[17:11:41] |c:|1779642701|~|**Players (2)**: Alice, Bob\n"
            "[00:01] Alice: hello\n"
            "[00:02] Bob: hi\n"
            "Alice's role was town"
        )
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "mafia.db"
            initialize_database(db_path)
            game_id = ingest_log(
                db_path,
                raw_text,
                source="test",
            )
            set_game_display_name(db_path, game_id, "Test Game")
            summaries = list_game_summaries(db_path)
            self.assertEqual(summaries[0].display_name, "Test Game")
            self.assertEqual(summaries[0].undefined_count, 1)

            undefined = find_undefined_players(db_path, game_id)
            self.assertEqual(undefined[0].player_name, "Bob")

            session = GameSession(
                source="x",
                raw_text="",
                players=["Alice", "Bob"],
                flips=[],
            )
            session.flips = []
            from mafia_framework.data.models import Flip

            session.flips = [Flip(player_name="Alice", alignment="town")]
            resolved = resolve_flip_map(session)
            self.assertEqual(resolved["Alice"], "town")
            self.assertNotIn("Bob", resolved)
