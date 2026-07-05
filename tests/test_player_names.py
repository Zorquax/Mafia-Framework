import unittest

from mafia_framework.io.parser import parse_showdown_log
from mafia_framework.io.player_names import (
    canonical_player_name,
    names_match,
    player_identity_key,
)
from mafia_framework.services.game_log_service import build_game_log, match_players_in_session


class TestPlayerNames(unittest.TestCase):

    def test_identity_key_ignores_decorators_and_case(self):
        self.assertEqual(player_identity_key("Schiavetto ♫♪♫♪"), "schiavetto")
        self.assertEqual(player_identity_key("Schia♫♪♫♪vetto"), "schiavetto")
        self.assertEqual(player_identity_key("SCHIAVETTO"), "schiavetto")
        self.assertEqual(player_identity_key("Schia Vetto"), "schiavetto")
        self.assertTrue(names_match("Schiavetto ♫♪♫♪", "Schia Vetto"))

    def test_parser_merges_name_variants_in_roster(self):
        raw = (
            "[17:11:41] |c:|1779642701|~|**Players (2)**: Schiavetto ♫♪♫♪, Schia Vetto\n"
            "[00:01] Schia♫♪♫♪vetto: hello\n"
            "[00:02] SCHIAVETTO: world\n"
        )
        session = parse_showdown_log(raw)
        self.assertEqual(len(session.players), 1)
        self.assertEqual(session.players[0], "Schiavetto")
        self.assertEqual(len(session.messages), 2)
        self.assertEqual(session.messages[0].player_name, "Schiavetto")
        self.assertEqual(session.messages[1].player_name, "Schiavetto")

    def test_game_log_matches_fuzzy_player_names(self):
        session = parse_showdown_log(
            "[17:11:41] |c:|1779642701|~|**Players (2)**: commanderawesome, aziziller\n"
            "Day 1. The hammer count is set at 9\n"
            "[00:01] commanderawesome: hi\n"
            "[00:02] |c:|1|~|commanderawesome has voted aziziller.\n"
        )
        matched = match_players_in_session(session, ["commanderawesome", "aziziller"])
        self.assertEqual(matched, {"commanderawesome", "aziziller"})
        entries = build_game_log(session, ["commanderawesome", "aziziller"], mode="both")
        self.assertTrue(any(entry.entry_type == "phase" for entry in entries))
        self.assertTrue(any(entry.entry_type == "message" for entry in entries))
        self.assertTrue(any(entry.entry_type == "vote" for entry in entries))

    def test_game_log_includes_elimination_and_reveal(self):
        raw = (
            "[17:11:41] |c:|1779642701|~|**Players (2)**: commanderawesome, aziziller\n"
            "Day 1. The hammer count is set at 9\n"
            "[00:01] commanderawesome: hi\n"
            "aziziller was eliminated!\n"
            "aziziller's role was Vanilla Townie.\n"
        )
        session = parse_showdown_log(raw)
        self.assertEqual(len(session.events), 2)
        self.assertEqual(session.events[0].event_type, "elimination")
        self.assertEqual(session.events[0].player_name, "aziziller")
        self.assertEqual(session.events[1].event_type, "reveal")
        self.assertEqual(session.events[1].player_name, "aziziller")

        # Now test build_game_log
        entries = build_game_log(session, ["aziziller"], mode="both")
        self.assertTrue(any(entry.entry_type == "elimination" for entry in entries))
        self.assertTrue(any(entry.entry_type == "reveal" for entry in entries))
