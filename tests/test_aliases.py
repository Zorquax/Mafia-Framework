import json
import tempfile
import unittest
from pathlib import Path

from mafia_framework.analysis.tells.line_count import LineCountTell
from mafia_framework.analysis.tells.base import aggregate_tells
from mafia_framework.io.player_names import player_identity_key
from mafia_framework.data.aliases import (
    apply_player_aliases,
    import_aliases_from_json,
    list_player_aliases,
    load_alias_map,
    resolve_name,
    set_player_alias,
)
from mafia_framework.data.models import Flip, GameSession, Message, Vote, initialize_database
from mafia_framework.io.ingestion import load_game, ingest_log
from mafia_framework.models.feature_engineering import build_training_dataset


class TestAliases(unittest.TestCase):

    def test_resolve_name_follows_chain(self):
        alias_map = {
            "linduana": "Linda",
            "Ailura": "Linda",
            "Linda": "Linda",
        }
        self.assertEqual(resolve_name("linduana", alias_map), "Linda")
        self.assertEqual(resolve_name("Ailura", alias_map), "Linda")

    def test_apply_player_aliases_merges_messages_and_flips(self):
        session = GameSession(
            source="test",
            raw_text="",
            players=["linduana", "Bob"],
            messages=[
                Message(player_name="linduana", text="hello", day=1),
                Message(player_name="Linda", text="world", day=1),
            ],
            votes=[Vote(voter_name="linduana", target_name="Bob", day=1)],
            flips=[Flip(player_name="Linda", alignment="town")],
        )
        alias_map = {"linduana": "Linda", "Ailura": "Linda"}
        apply_player_aliases(session, alias_map)

        self.assertEqual(session.players, ["Linda", "Bob"])
        self.assertEqual(len(session.flips), 1)
        self.assertEqual(session.flips[0].player_name, "Linda")
        self.assertEqual(session.flips[0].alignment, "town")
        self.assertEqual({m.player_name for m in session.messages}, {"Linda"})
        self.assertEqual(session.votes[0].voter_name, "Linda")

        tell_results = aggregate_tells(session, [LineCountTell()])
        linda = next(r for r in tell_results if r.player_name == "Linda")
        self.assertEqual(linda.features["line_count"], 2.0)

    def test_alias_persistence_and_load_game(self):
        raw_text = (
            "[00:01] linduana: hello\n"
            "[00:02] Bob: hi\n"
            "Linda's role was town\n"
            "Bob's role was mafia"
        )
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "mafia.db"
            initialize_database(db_path)
            set_player_alias(db_path, "linduana", "Linda")
            game_id = ingest_log(db_path, raw_text, source="test")

            session = load_game(db_path, game_id)
            self.assertIsNotNone(session)
            self.assertEqual({m.player_name for m in session.messages}, {"Linda", "Bob"})
            self.assertEqual({f.player_name for f in session.flips}, {"Linda", "Bob"})

            X, y, _ = build_training_dataset([session], [LineCountTell()], feature_names=["line_count"])
            self.assertEqual(len(X), 2)
            self.assertCountEqual(y, ["town", "mafia"])

    def test_import_aliases_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "mafia.db"
            json_path = Path(tmp) / "aliases.json"
            json_path.write_text(
                json.dumps({"Ailura": "Linda", "linduana": "Linda"}),
                encoding="utf-8",
            )
            count = import_aliases_from_json(db_path, json_path)
            self.assertEqual(count, 2)
            rows = list_player_aliases(db_path)
            self.assertEqual(len(rows), 2)
            self.assertEqual(load_alias_map(db_path)[player_identity_key("Ailura")], "Linda")
