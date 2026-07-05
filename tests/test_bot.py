import asyncio
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from mafia_framework.bot.client import MafiaBot
from mafia_framework.bot.config import BotConfig
from mafia_framework.bot.tracker import GameTracker
from mafia_framework.bot.strategy import BotStrategy
from mafia_framework.data.models import GameSession, Message, Vote, Flip, LogEvent


class TestBotComponents(unittest.TestCase):

    def test_config_loading_defaults(self):
        # Test loading from a non-existent file
        config = BotConfig.load_from_file("nonexistent_config.toml")
        self.assertEqual(config.showdown.room, "mafia")
        self.assertEqual(config.gameplay.autojoin, True)
        self.assertEqual(config.database.db_path, "data/mafia.db")

    def test_config_loading_custom(self):
        toml_content = """
        [showdown]
        username = "TestBot"
        password = "secretpassword"
        room = "customroom"

        [gameplay]
        autojoin = false
        min_confidence_to_vote = 0.65
        night_idle = false
        update_suspicion_frequency_seconds = 30

        [database]
        db_path = "test_data/test.db"
        """
        with tempfile.TemporaryDirectory() as tmp:
            config_file = Path(tmp) / "config.toml"
            config_file.write_text(toml_content, encoding="utf-8")

            config = BotConfig.load_from_file(config_file)
            self.assertEqual(config.showdown.username, "TestBot")
            self.assertEqual(config.showdown.password, "secretpassword")
            self.assertEqual(config.showdown.room, "customroom")
            self.assertEqual(config.gameplay.autojoin, False)
            self.assertEqual(config.gameplay.min_confidence_to_vote, 0.65)
            self.assertEqual(config.gameplay.night_idle, False)
            self.assertEqual(config.gameplay.update_suspicion_frequency_seconds, 30)
            self.assertEqual(config.database.db_path, "test_data/test.db")

    def test_game_tracker_transitions(self):
        tracker = GameTracker()
        self.assertEqual(tracker.state, "IDLE")

        # 1. Signups start
        event = tracker.process_message("|c:|1779642701|~|A game of Mafia has been started by Host")
        self.assertEqual(tracker.state, "SIGNUPS")
        self.assertEqual(event, "SIGNUPS")

        # 2. Game start / roster announcement
        event = tracker.process_message("|c:|1779642701|~|**Players (3)**: Alice, Bob, Charlie")
        self.assertEqual(tracker.state, "DAY")
        self.assertEqual(event, "STARTED")
        self.assertEqual(tracker.players, ["Alice", "Bob", "Charlie"])

        # 3. Night starts
        event = tracker.process_message("|c:|1779642701|~|Night 1 has begun.")
        self.assertEqual(tracker.state, "NIGHT")
        self.assertEqual(event, "NIGHT")

        # 4. Day starts
        event = tracker.process_message("Day 2. The hammer count is set at 2")
        self.assertEqual(tracker.state, "DAY")
        self.assertEqual(event, "DAY")
        self.assertEqual(tracker.current_day, 2)

        # 5. Game ends
        event = tracker.process_message("The Mafia has won!")
        self.assertEqual(tracker.state, "IDLE")
        self.assertEqual(event, "FINISHED")

    def test_strategy_voting_and_overrides(self):
        # Create a mock session
        session = GameSession(
            source="test",
            raw_text="",
            players=["Alice", "Bob", "BotUser"],
            messages=[
                Message(player_name="Alice", text="hello", day=1),
                Message(player_name="Bob", text="hi", day=1),
            ],
            flips=[
                Flip(player_name="Alice", alignment="town"),
            ],
        )

        strategy = BotStrategy(
            model_path="data/model.pkl",
            model_d1_path="data/model_d1.pkl",
            min_confidence=0.55
        )

        # Test manual override voting target selection
        strategy.set_manual_vote("Bob")
        target, prob = strategy.get_vote_decision(session, bot_username="BotUser", db_path="dummy.db")
        self.assertEqual(target, "Bob")
        self.assertEqual(prob, 1.0)

        # Manual override target not in roster should be ignored
        strategy.set_manual_vote("UnknownPlayer")
        # Since dummy.db model path doesn't exist, it should return None
        target, prob = strategy.get_vote_decision(session, bot_username="BotUser", db_path="dummy.db")
        self.assertIsNone(target)
        self.assertEqual(prob, 0.0)

        # Test resets
        strategy.reset()
        self.assertIsNone(strategy.manual_vote_override)
        self.assertEqual(strategy.suspicion_multipliers, {})

    def test_vote_detection_for_bot_username(self):
        self.assertTrue(MafiaBot._is_vote_for_bot("|c:|123|~|Alice has voted BotUser.", "BotUser"))
        self.assertTrue(MafiaBot._is_vote_for_bot("|c:|123|~|Alice voted for bot-user", "BotUser"))
        self.assertFalse(MafiaBot._is_vote_for_bot("|c:|123|~|Alice has voted Bob.", "BotUser"))

    def test_random_vote_is_skipped_when_strategy_has_prediction(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.strategy = Mock()
        bot.strategy.get_vote_decision.return_value = ("Alice", 0.80)
        bot.config = SimpleNamespace(
            showdown=SimpleNamespace(username="BotUser"),
            database=SimpleNamespace(db_path="dummy.db"),
        )

        session = GameSession(source="test", raw_text="", players=["Alice", "Bob"])
        self.assertFalse(bot._should_do_random_vote(session))

    def test_handle_pm_send_responds_with_random_live_player(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(in_game=True, eliminated=False, players=["Alice", "Bob"], dead_players=set())
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="BotUser"))
        bot.connection = Mock()
        bot.connection.send = AsyncMock()
        bot.strategy = Mock()
        bot.strategy.reset = Mock()
        bot._evaluate_and_vote = AsyncMock()

        with patch("random.choice", return_value="Bob"):
            asyncio.run(bot._handle_pm("Alice", "please send me a random name"))

        bot.connection.send.assert_awaited_once_with("|/pm Alice, Bob")

    def test_handle_pm_claim_uses_revealed_role_with_mafia_goon_special_case(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(in_game=True, eliminated=False, players=["Alice", "Bob"], dead_players=set())
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="BotUser"))
        bot.connection = Mock()
        bot.connection.send = AsyncMock()

        session = GameSession(
            source="test",
            raw_text="",
            players=["Alice", "Bob", "BotUser"],
            events=[LogEvent(player_name="BotUser", event_type="reveal", text="BotUser's role was Mafia Goon")],
        )

        self.assertEqual(bot._get_claim_message(session), "VT 1 to hammer")

        bot.tracker.get_game_session = Mock(return_value=session)
        asyncio.run(bot._handle_pm("Alice", "claim"))

        bot.connection.send.assert_awaited_once_with("|/pm Alice, VT 1 to hammer")

    def test_extract_vote_voter_from_vote_message(self):
        self.assertEqual(MafiaBot._extract_vote_voter("|c:|123|~|Alice has voted BotUser."), "Alice")
        self.assertIsNone(MafiaBot._extract_vote_voter("|c:|123|~|Alice is chatting"))

    def test_question_prompt_ignores_players_after_elimination(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = GameTracker()
        bot.tracker.state = "DAY"
        bot.tracker.players = ["Alice", "Bob"]
        bot.tracker.in_game = True
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="BotUser"))

        bot.tracker.process_message("|c:|123|~|Bob was eliminated!", bot_username="BotUser")
        session = GameSession(source="test", raw_text="", players=["Alice", "Bob"])

        with patch("random.choices", return_value=[1]), patch("random.sample", side_effect=lambda items, k: items[:k]):
            prompt = bot._build_question_prompt(session)

        self.assertIn("Alice", prompt)
        self.assertNotIn("Bob", prompt)

    def test_tracker_marks_bot_eliminated_when_its_name_is_in_elimination_line(self):
        tracker = GameTracker()
        tracker.state = "DAY"
        tracker.players = ["Alice", "BotUser"]
        tracker.in_game = True

        tracker.process_message("|c:|123|~|BotUser was eliminated!", bot_username="BotUser")

        self.assertTrue(tracker.eliminated)
        self.assertNotIn("BotUser", tracker.players)

    def test_tracker_get_game_session_populates_flips_from_reveal_lines(self):
        tracker = GameTracker()
        tracker.accumulated_lines = [
            "|c:|123|~|Alice's role was Mafia Goon.",
            "|c:|123|~|Bob was eliminated!",
        ]

        session = tracker.get_game_session()
        self.assertEqual(len(session.flips), 1)
        self.assertEqual(session.flips[0].player_name, "Alice")
        self.assertEqual(session.flips[0].alignment, "mafia")

    def test_tracker_attributes_chat_messages_to_players(self):
        # Real player chat arrives as |c:|<ts>|<username>|<message>. The
        # normalizer must preserve the username so downstream parsing can
        # attribute the line to a player, instead of discarding it.
        tracker = GameTracker()
        tracker.accumulated_lines = [
            "|c:|1|Alice|I think Bob is scum, he is voting weird",
        ]

        session = tracker.get_game_session()
        self.assertEqual(len(session.messages), 1)
        self.assertEqual(session.messages[0].player_name, "Alice")
        self.assertEqual(session.messages[0].text, "I think Bob is scum, he is voting weird")

    def test_tracker_captures_system_vote_announcements(self):
        tracker = GameTracker()
        tracker.accumulated_lines = [
            "|c:|1|~|Alice has voted Bob.",
        ]

        session = tracker.get_game_session()
        self.assertEqual(len(session.votes), 1)
        self.assertEqual(session.votes[0].voter_name, "Alice")
        self.assertEqual(session.votes[0].target_name, "Bob")

    def test_tracker_uses_cleaned_html_reveal_text_for_flips(self):
        tracker = GameTracker()
        tracker.accumulated_lines = [
            '|raw|<div class="broadcast-blue">thisisbdavi\'s role was Town</div>',
        ]

        session = tracker.get_game_session()
        self.assertEqual(len(session.flips), 1)
        self.assertEqual(session.flips[0].player_name, "thisisbdavi")
        self.assertEqual(session.flips[0].alignment, "town")

    def test_night_action_target_uses_random_non_self_player_for_non_vt_role(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(dead_players=set())
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="BotUser"))
        session = GameSession(
            source="test",
            raw_text="",
            players=["Alice", "Bob", "BotUser"],
            flips=[Flip(player_name="BotUser", alignment="mafia")],
        )

        with patch("random.choice", return_value="Bob"):
            self.assertEqual(bot._choose_night_action_target(session), "Bob")

        session.flips = [Flip(player_name="BotUser", alignment="town")]
        self.assertIsNone(bot._choose_night_action_target(session))

    def test_strategy_skips_self_when_username_is_normalized_differently(self):
        session = GameSession(
            source="test",
            raw_text="",
            players=["Alice", "Bot-User"],
            messages=[Message(player_name="Alice", text="hello", day=1)],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pkl"
            model_path.write_text("", encoding="utf-8")
            strategy = BotStrategy(
                model_path=str(model_path),
                model_d1_path=str(model_path),
                min_confidence=0.55,
            )

            predictions = [
                SimpleNamespace(player_name="Bot-User", probabilities={"mafia": 0.95}),
                SimpleNamespace(player_name="Alice", probabilities={"mafia": 0.80}),
            ]

            with patch("mafia_framework.bot.strategy.predict_session", return_value=predictions):
                target, prob = strategy.get_vote_decision(session, bot_username="BotUser", db_path="dummy.db")

        self.assertEqual(target, "Alice")
        self.assertAlmostEqual(prob, 0.80)
