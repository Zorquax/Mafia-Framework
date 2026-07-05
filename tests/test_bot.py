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
from mafia_framework.services.game_service import UndefinedPlayerRow


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

    def test_strategy_get_town_read_picks_lowest_mafia_probability(self):
        session = GameSession(
            source="test",
            raw_text="",
            players=["Alice", "Bob", "BotUser"],
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
                SimpleNamespace(player_name="Alice", probabilities={"mafia": 0.05}),
                SimpleNamespace(player_name="Bob", probabilities={"mafia": 0.80}),
            ]

            with patch("mafia_framework.bot.strategy.predict_session", return_value=predictions):
                target, prob = strategy.get_town_read(session, bot_username="BotUser", db_path="dummy.db")

        self.assertEqual(target, "Alice")
        self.assertAlmostEqual(prob, 0.95)

    def test_strategy_get_town_read_below_confidence_returns_none(self):
        session = GameSession(
            source="test",
            raw_text="",
            players=["Alice", "Bob", "BotUser"],
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
                SimpleNamespace(player_name="Alice", probabilities={"mafia": 0.50}),
                SimpleNamespace(player_name="Bob", probabilities={"mafia": 0.60}),
            ]

            with patch("mafia_framework.bot.strategy.predict_session", return_value=predictions):
                target, prob = strategy.get_town_read(session, bot_username="BotUser", db_path="dummy.db")

        self.assertIsNone(target)
        self.assertAlmostEqual(prob, 0.50)

    def test_send_chat_message_delays_before_sending(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.connection = Mock()
        bot.connection.room = "mafia"
        bot.connection.send = AsyncMock()

        with patch("mafia_framework.bot.client.asyncio.sleep", new=AsyncMock()) as mock_sleep:
            asyncio.run(bot._send_chat_message("hello"))

        mock_sleep.assert_awaited_once()
        bot.connection.send.assert_awaited_once_with("mafia|hello")

    def test_maybe_remember_chat_line_ignores_own_messages_by_sender(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="zorq_bot"))
        bot._remembered_lines = []

        # Own message, sent as the bot -- must not be remembered even though
        # the text itself doesn't mention the bot's name.
        bot._maybe_remember_chat_line("|c:|123|zorq_bot|I think Bob is scum")
        self.assertEqual(bot._remembered_lines, [])

        # Someone else's message should still be remembered.
        bot._maybe_remember_chat_line("|c:|123|Alice|I think Bob is scum")
        self.assertEqual(bot._remembered_lines, ["I think Bob is scum"])

    def test_maybe_claim_at_v1_triggers_when_bot_reaches_hammer_minus_one(self):
        bot = MafiaBot.__new__(MafiaBot)
        session = GameSession(
            source="test",
            raw_text="",
            players=["Alice", "BotUser"],
            votes=[Vote(voter_name="Alice", target_name="BotUser", day=1, action="vote")],
        )
        bot.tracker = SimpleNamespace(hammer_count=2, get_game_session=Mock(return_value=session))
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="BotUser"))
        bot._claimed_this_day = False
        bot._own_role = "Vanilla Townie"
        bot._send_chat_message = AsyncMock()

        asyncio.run(bot._maybe_claim_at_v1())

        bot._send_chat_message.assert_awaited_once_with("Vanilla Townie 1 to hammer")
        self.assertTrue(bot._claimed_this_day)

    def test_maybe_claim_at_v1_does_nothing_below_hammer_minus_one(self):
        bot = MafiaBot.__new__(MafiaBot)
        session = GameSession(
            source="test",
            raw_text="",
            players=["Alice", "BotUser"],
            votes=[Vote(voter_name="Alice", target_name="BotUser", day=1, action="vote")],
        )
        bot.tracker = SimpleNamespace(hammer_count=3, get_game_session=Mock(return_value=session))
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="BotUser"))
        bot._claimed_this_day = False
        bot._own_role = "Vanilla Townie"
        bot._send_chat_message = AsyncMock()

        asyncio.run(bot._maybe_claim_at_v1())

        bot._send_chat_message.assert_not_awaited()
        self.assertFalse(bot._claimed_this_day)

    def test_evaluate_and_vote_announces_new_town_read(self):
        bot = MafiaBot.__new__(MafiaBot)
        session = GameSession(source="test", raw_text="", players=["Alice", "Bob"])
        bot.tracker = SimpleNamespace(
            state="DAY", eliminated=False, in_game=True,
            get_game_session=Mock(return_value=session),
        )
        bot.strategy = Mock()
        bot.strategy.get_vote_decision = Mock(return_value=(None, 0.0))
        bot.strategy.get_town_read = Mock(return_value=("Alice", 0.90))
        bot.config = SimpleNamespace(
            showdown=SimpleNamespace(username="BotUser"),
            database=SimpleNamespace(db_path="dummy.db"),
            gameplay=SimpleNamespace(town_read_comment_chance=1.0, vote_comment_chance=1.0),
        )
        bot._current_vote_target = None
        bot._current_town_read = None
        bot._send_chat_message = AsyncMock()
        bot.send_room_command = AsyncMock()

        asyncio.run(bot._evaluate_and_vote())

        bot._send_chat_message.assert_awaited_once_with("Alice is town")
        self.assertEqual(bot._current_town_read, "Alice")

    def test_evaluate_and_vote_does_not_repeat_unchanged_town_read(self):
        bot = MafiaBot.__new__(MafiaBot)
        session = GameSession(source="test", raw_text="", players=["Alice", "Bob"])
        bot.tracker = SimpleNamespace(
            state="DAY", eliminated=False, in_game=True,
            get_game_session=Mock(return_value=session),
        )
        bot.strategy = Mock()
        bot.strategy.get_vote_decision = Mock(return_value=(None, 0.0))
        bot.strategy.get_town_read = Mock(return_value=("Alice", 0.90))
        bot.config = SimpleNamespace(
            showdown=SimpleNamespace(username="BotUser"),
            database=SimpleNamespace(db_path="dummy.db"),
            gameplay=SimpleNamespace(town_read_comment_chance=1.0, vote_comment_chance=1.0),
        )
        bot._current_vote_target = None
        bot._current_town_read = "Alice"
        bot._send_chat_message = AsyncMock()
        bot.send_room_command = AsyncMock()

        asyncio.run(bot._evaluate_and_vote())

        bot._send_chat_message.assert_not_awaited()

    def test_cast_vote_with_optional_comment_silent_when_chance_is_zero(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.config = SimpleNamespace(gameplay=SimpleNamespace(vote_comment_chance=0.0))
        bot._send_chat_message = AsyncMock()
        bot.send_room_command = AsyncMock()

        asyncio.run(bot._cast_vote_with_optional_comment("Alice"))

        bot._send_chat_message.assert_not_awaited()
        bot.send_room_command.assert_awaited_once_with("/mafia vote Alice")

    def test_cast_vote_with_optional_comment_speaks_when_chance_is_one(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.config = SimpleNamespace(gameplay=SimpleNamespace(vote_comment_chance=1.0))
        bot._send_chat_message = AsyncMock()
        bot.send_room_command = AsyncMock()

        with patch("mafia_framework.bot.client.asyncio.sleep", new=AsyncMock()):
            asyncio.run(bot._cast_vote_with_optional_comment("Alice"))

        bot._send_chat_message.assert_awaited_once_with("I think Alice is scum")
        bot.send_room_command.assert_awaited_once_with("/mafia vote Alice")

    def test_deadline_events_trigger_reevaluation(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot._random_actions_task = None
        bot._random_vote_task = None
        bot._evaluate_and_vote = AsyncMock()

        for event in ("DEADLINE_3MIN", "DEADLINE_1MIN"):
            bot._evaluate_and_vote.reset_mock()
            asyncio.run(bot._handle_tracker_event(event))
            bot._evaluate_and_vote.assert_awaited_once()

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

    def test_get_claim_message_lies_as_vt_for_listed_roles(self):
        bot = MafiaBot.__new__(MafiaBot)
        for role in ["Werewolf", "Alien", "Cult Leader", "Serial Killer", "Goo"]:
            bot._own_role = role
            self.assertEqual(bot._get_claim_message(), "VT 1 to hammer", msg=f"role={role}")

    def test_get_claim_message_claims_real_role_otherwise(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot._own_role = "Vanilla Townie"
        self.assertEqual(bot._get_claim_message(), "Vanilla Townie 1 to hammer")

        bot._own_role = "Mafia Goon"
        self.assertEqual(bot._get_claim_message(), "Mafia Goon 1 to hammer")

    def test_get_claim_message_none_when_role_unknown(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot._own_role = None
        self.assertIsNone(bot._get_claim_message())

    def test_handle_pm_claim_uses_own_role(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(in_game=True, eliminated=False, players=["Alice", "Bob"], dead_players=set())
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="BotUser"))
        bot.connection = Mock()
        bot.connection.send = AsyncMock()
        bot._own_role = "Cult Leader"

        asyncio.run(bot._handle_pm("Alice", "claim"))

        bot.connection.send.assert_awaited_once_with("|/pm Alice, VT 1 to hammer")

    def test_handle_pm_learns_own_role_from_role_assignment_pm(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(in_game=True, eliminated=False, players=["Alice", "Bob"], dead_players=set())
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="zorq_bot"))
        bot.connection = Mock()
        bot.connection.send = AsyncMock()
        bot._own_role = None

        asyncio.run(bot._handle_pm("Host", "zorq_bot, you are a Vanilla Townie"))

        self.assertEqual(bot._own_role, "Vanilla Townie")
        bot.connection.send.assert_not_awaited()

    def test_handle_pm_ignores_own_echoed_messages(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(in_game=True, eliminated=False, players=["Alice", "Bob"], dead_players=set())
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="zorq_bot"))
        bot.connection = Mock()
        bot.connection.send = AsyncMock()

        asyncio.run(bot._handle_pm("zorq_bot", "please send me a random name"))

        bot.connection.send.assert_not_awaited()

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

    def test_eliminated_flag_stays_latched_across_later_day_markers(self):
        # A later _prune_dead_players() call made without bot_username (as
        # happens on every subsequent day-marker transition) must not flip
        # the eliminated flag back to False.
        tracker = GameTracker()
        tracker.state = "DAY"
        tracker.players = ["Alice", "BotUser"]
        tracker.in_game = True

        tracker.process_message("|c:|1|~|BotUser was eliminated!", bot_username="BotUser")
        self.assertTrue(tracker.eliminated)
        self.assertFalse(tracker.in_game)

        tracker.process_message("Day 2. The hammer count is set at 2", bot_username="BotUser")
        self.assertTrue(tracker.eliminated)
        self.assertFalse(tracker.in_game)
        self.assertEqual(tracker.hammer_count, 2)

    def test_hammer_count_parsed_from_day_marker(self):
        tracker = GameTracker()
        tracker.state = "DAY"
        tracker.players = ["Alice", "Bob"]
        tracker.in_game = True

        tracker.process_message("Day 3. The hammer count is set at 4", bot_username="BotUser")
        self.assertEqual(tracker.hammer_count, 4)

    def test_deadline_warnings_trigger_events_once_each(self):
        tracker = GameTracker()
        tracker.state = "DAY"
        tracker.players = ["Alice", "Bob"]
        tracker.in_game = True

        event = tracker.process_message("**3 minutes left!**", bot_username="BotUser")
        self.assertEqual(event, "DEADLINE_3MIN")
        self.assertEqual(tracker.deadline_warning, "3_minutes")

        # Repeating the same warning must not re-trigger the event.
        event = tracker.process_message("**3 minutes left!**", bot_username="BotUser")
        self.assertIsNone(event)

        event = tracker.process_message("**1 minute left!**", bot_username="BotUser")
        self.assertEqual(event, "DEADLINE_1MIN")
        self.assertEqual(tracker.deadline_warning, "1_minute")

    def test_deadline_warning_resets_on_new_day(self):
        tracker = GameTracker()
        tracker.state = "DAY"
        tracker.players = ["Alice", "Bob"]
        tracker.in_game = True

        tracker.process_message("**1 minute left!**", bot_username="BotUser")
        self.assertEqual(tracker.deadline_warning, "1_minute")

        tracker.process_message("Day 2. The hammer count is set at 3", bot_username="BotUser")
        self.assertIsNone(tracker.deadline_warning)

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

    def test_prompt_for_undefined_roles_assigns_valid_input(self):
        bot = MafiaBot.__new__(MafiaBot)
        rows = [
            UndefinedPlayerRow(game_id=1, display_name=None, player_name="Alice", has_messages=True, is_inferred_town_candidate=True),
            UndefinedPlayerRow(game_id=1, display_name=None, player_name="Bob", has_messages=False, is_inferred_town_candidate=False),
        ]

        with (
            patch("mafia_framework.services.game_service.find_undefined_players", return_value=rows),
            patch("mafia_framework.services.game_service.assign_player_role") as mock_assign,
            patch("builtins.input", side_effect=["town", "mafia"]),
        ):
            asyncio.run(bot._prompt_for_undefined_roles("dummy.db", 1))

        mock_assign.assert_any_call("dummy.db", 1, "Alice", "town")
        mock_assign.assert_any_call("dummy.db", 1, "Bob", "mafia")
        self.assertEqual(mock_assign.call_count, 2)

    def test_prompt_for_undefined_roles_skips_blank_and_invalid_input(self):
        bot = MafiaBot.__new__(MafiaBot)
        rows = [
            UndefinedPlayerRow(game_id=1, display_name=None, player_name="Alice", has_messages=True, is_inferred_town_candidate=False),
            UndefinedPlayerRow(game_id=1, display_name=None, player_name="Bob", has_messages=True, is_inferred_town_candidate=False),
        ]

        with (
            patch("mafia_framework.services.game_service.find_undefined_players", return_value=rows),
            patch("mafia_framework.services.game_service.assign_player_role") as mock_assign,
            patch("builtins.input", side_effect=["", "not_a_role"]),
        ):
            asyncio.run(bot._prompt_for_undefined_roles("dummy.db", 1))

        mock_assign.assert_not_called()

    def test_prompt_for_undefined_roles_noop_when_none_undefined(self):
        bot = MafiaBot.__new__(MafiaBot)

        with (
            patch("mafia_framework.services.game_service.find_undefined_players", return_value=[]),
            patch("builtins.input") as mock_input,
        ):
            asyncio.run(bot._prompt_for_undefined_roles("dummy.db", 1))

        mock_input.assert_not_called()
