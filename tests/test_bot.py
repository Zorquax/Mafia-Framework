import asyncio
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from mafia_framework.bot.client import MafiaBot, RAGEBAIT_LINES
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
        # db_path is resolved to an absolute path (robust to whatever
        # directory the process was launched from), so just check the
        # meaningful suffix rather than the raw config default.
        self.assertTrue(config.database.db_path.replace("\\", "/").endswith("data/mafia.db"))
        self.assertTrue(Path(config.database.db_path).is_absolute())

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
            self.assertTrue(config.database.db_path.replace("\\", "/").endswith("test_data/test.db"))
            self.assertTrue(Path(config.database.db_path).is_absolute())

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

    def test_strategy_voting_without_model_returns_none(self):
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
            model_path="nonexistent_model.pkl",
            model_d1_path="nonexistent_model_d1.pkl",
            min_confidence=0.55
        )

        # Since the model path doesn't exist, no decision can be made.
        target, prob = strategy.get_vote_decision(session, bot_username="BotUser", db_path="dummy.db")
        self.assertIsNone(target)
        self.assertEqual(prob, 0.0)

        strategy.set_suspicion_multiplier("Bob", 2.0)
        strategy.reset()
        self.assertEqual(strategy.suspicion_multipliers, {})

    def test_get_vote_decision_min_confidence_override(self):
        strategy = BotStrategy(model_path="nonexistent_model.pkl", model_d1_path="nonexistent_model_d1.pkl", min_confidence=0.55)
        session = GameSession(source="test", raw_text="", players=["Alice", "Bob", "BotUser"])

        with patch.object(strategy, "_score_players", return_value=[("Bob", 0.65)]):
            # Default threshold (0.55): confident enough to vote.
            target, prob = strategy.get_vote_decision(session, bot_username="BotUser", db_path="dummy.db")
            self.assertEqual(target, "Bob")
            self.assertAlmostEqual(prob, 0.65)

            # A stricter override (e.g. during VoLo) should withhold the
            # vote even though the same underlying score would normally
            # clear the bar.
            target, prob = strategy.get_vote_decision(
                session, bot_username="BotUser", db_path="dummy.db", min_confidence=0.75
            )
            self.assertIsNone(target)
            self.assertAlmostEqual(prob, 0.65)

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

    def test_strategy_get_full_predictions_returns_all_targets_ranked(self):
        session = GameSession(
            source="test",
            raw_text="",
            players=["Alice", "Bob", "Charlie", "BotUser"],
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
                SimpleNamespace(player_name="Alice", probabilities={"mafia": 0.10}),
                SimpleNamespace(player_name="Bob", probabilities={"mafia": 0.75}),
                SimpleNamespace(player_name="Charlie", probabilities={"mafia": 0.40}),
            ]

            with patch("mafia_framework.bot.strategy.predict_session", return_value=predictions):
                results = strategy.get_full_predictions(session, bot_username="BotUser", db_path="dummy.db")

        self.assertEqual([name for name, _ in results], ["Bob", "Charlie", "Alice"])
        self.assertAlmostEqual(results[0][1], 0.75)

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

    def test_maybe_remember_chat_line_ignores_system_announcements(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="zorq_bot"))
        bot._remembered_lines = []

        bot._maybe_remember_chat_line("|c:|123|~|**Players (3)**: Alice, Bob, zorq_bot")
        self.assertEqual(bot._remembered_lines, [])

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

        bot._send_chat_message.assert_awaited_once_with("I HARDCLAIM Vanilla Townie get OFF")
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

    def test_maybe_claim_at_v1_prefers_live_vote_counts_over_chat_derived(self):
        # The chat-derived reconstruction (no votes recorded at all) would
        # say the bot has 0 votes, but the authoritative `/mafia votes` reply
        # says it's already sitting at hammer-minus-one -- that should win.
        bot = MafiaBot.__new__(MafiaBot)
        session = GameSession(source="test", raw_text="", players=["Alice", "BotUser"], votes=[])
        bot.tracker = SimpleNamespace(
            hammer_count=2,
            live_vote_counts={"BotUser": 1},
            get_game_session=Mock(return_value=session),
        )
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="BotUser"))
        bot._claimed_this_day = False
        bot._own_role = "Vanilla Townie"
        bot._send_chat_message = AsyncMock()

        asyncio.run(bot._maybe_claim_at_v1())

        bot._send_chat_message.assert_awaited_once_with("I HARDCLAIM Vanilla Townie get OFF")
        self.assertTrue(bot._claimed_this_day)

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
            gameplay=SimpleNamespace(town_read_comment_chance=1.0, vote_comment_chance=1.0, silent_mode=False),
        )
        bot._current_vote_target = None
        bot._current_town_read = None
        bot._send_chat_message = AsyncMock()
        bot.send_room_command = AsyncMock()

        asyncio.run(bot._evaluate_and_vote())

        bot._send_chat_message.assert_awaited_once_with("Alice is town")
        self.assertEqual(bot._current_town_read, "Alice")

    def test_evaluate_and_vote_raises_confidence_bar_during_volo(self):
        bot = MafiaBot.__new__(MafiaBot)
        session = GameSession(source="test", raw_text="", players=["Alice", "Bob"])
        bot.tracker = SimpleNamespace(
            state="DAY", eliminated=False, in_game=True,
            get_game_session=Mock(return_value=session),
        )
        bot.strategy = Mock()
        bot.strategy.get_vote_decision = Mock(return_value=(None, 0.0))
        bot.strategy.get_town_read = Mock(return_value=(None, 0.0))
        bot.config = SimpleNamespace(
            showdown=SimpleNamespace(username="BotUser"),
            database=SimpleNamespace(db_path="dummy.db"),
            gameplay=SimpleNamespace(town_read_comment_chance=1.0, vote_comment_chance=1.0, volo_min_confidence=0.75, silent_mode=False),
        )
        bot._current_vote_target = None
        bot._current_town_read = None
        bot._send_chat_message = AsyncMock()
        bot.send_room_command = AsyncMock()
        bot._is_volo = Mock(return_value=True)

        asyncio.run(bot._evaluate_and_vote())

        bot.strategy.get_vote_decision.assert_called_once()
        _, kwargs = bot.strategy.get_vote_decision.call_args
        self.assertEqual(kwargs.get("min_confidence"), 0.75)

    def test_evaluate_and_vote_uses_default_confidence_when_not_volo(self):
        bot = MafiaBot.__new__(MafiaBot)
        session = GameSession(source="test", raw_text="", players=["Alice", "Bob"])
        bot.tracker = SimpleNamespace(
            state="DAY", eliminated=False, in_game=True,
            get_game_session=Mock(return_value=session),
        )
        bot.strategy = Mock()
        bot.strategy.get_vote_decision = Mock(return_value=(None, 0.0))
        bot.strategy.get_town_read = Mock(return_value=(None, 0.0))
        bot.config = SimpleNamespace(
            showdown=SimpleNamespace(username="BotUser"),
            database=SimpleNamespace(db_path="dummy.db"),
            gameplay=SimpleNamespace(town_read_comment_chance=1.0, vote_comment_chance=1.0, volo_min_confidence=0.75, silent_mode=False),
        )
        bot._current_vote_target = None
        bot._current_town_read = None
        bot._send_chat_message = AsyncMock()
        bot.send_room_command = AsyncMock()
        bot._is_volo = Mock(return_value=False)

        asyncio.run(bot._evaluate_and_vote())

        _, kwargs = bot.strategy.get_vote_decision.call_args
        self.assertIsNone(kwargs.get("min_confidence"))

    def test_evaluate_and_vote_random_fallback_when_no_confident_target(self):
        bot = MafiaBot.__new__(MafiaBot)
        session = GameSession(source="test", raw_text="", players=["Alice", "Bob", "BotUser"])
        bot.tracker = SimpleNamespace(
            state="DAY", eliminated=False, in_game=True, dead_players=set(),
            get_game_session=Mock(return_value=session),
        )
        bot.strategy = Mock()
        bot.strategy.min_confidence = 0.55
        bot.strategy.get_vote_decision = Mock(return_value=(None, 0.0))
        bot.strategy.get_town_read = Mock(return_value=(None, 0.0))
        bot.strategy.get_full_predictions = Mock(return_value=[("Alice", 0.3), ("Bob", 0.3)])
        bot.config = SimpleNamespace(
            showdown=SimpleNamespace(username="BotUser"),
            database=SimpleNamespace(db_path="dummy.db"),
            gameplay=SimpleNamespace(town_read_comment_chance=1.0, vote_comment_chance=1.0, silent_mode=False),
        )
        bot._current_vote_target = None
        bot._current_town_read = None
        bot._send_chat_message = AsyncMock()
        bot.send_room_command = AsyncMock()

        with patch("random.choice", return_value="Alice"):
            asyncio.run(bot._evaluate_and_vote(allow_random_fallback=True))

        bot.send_room_command.assert_any_call("/mafia vote Alice")
        self.assertEqual(bot._current_vote_target, "Alice")
        bot._send_chat_message.assert_not_awaited()

    def test_evaluate_and_vote_no_random_fallback_by_default(self):
        bot = MafiaBot.__new__(MafiaBot)
        session = GameSession(source="test", raw_text="", players=["Alice", "Bob", "BotUser"])
        bot.tracker = SimpleNamespace(
            state="DAY", eliminated=False, in_game=True, dead_players=set(),
            get_game_session=Mock(return_value=session),
        )
        bot.strategy = Mock()
        bot.strategy.get_vote_decision = Mock(return_value=(None, 0.0))
        bot.strategy.get_town_read = Mock(return_value=(None, 0.0))
        bot.config = SimpleNamespace(
            showdown=SimpleNamespace(username="BotUser"),
            database=SimpleNamespace(db_path="dummy.db"),
            gameplay=SimpleNamespace(town_read_comment_chance=1.0, vote_comment_chance=1.0, silent_mode=False),
        )
        bot._current_vote_target = None
        bot._current_town_read = None
        bot._send_chat_message = AsyncMock()
        bot.send_room_command = AsyncMock()

        asyncio.run(bot._evaluate_and_vote())

        bot.send_room_command.assert_not_awaited()
        self.assertIsNone(bot._current_vote_target)

    def test_pick_random_vote_target_excludes_confident_town_reads(self):
        bot = MafiaBot.__new__(MafiaBot)
        session = GameSession(source="test", raw_text="", players=["Alice", "Bob", "BotUser"])
        bot.tracker = SimpleNamespace(dead_players=set())
        bot.config = SimpleNamespace(
            showdown=SimpleNamespace(username="BotUser"),
            database=SimpleNamespace(db_path="dummy.db"),
        )
        bot.strategy = Mock()
        bot.strategy.min_confidence = 0.55
        # Alice reads confidently town (95% town / 5% mafia); Bob is a toss-up.
        bot.strategy.get_full_predictions = Mock(return_value=[("Alice", 0.05), ("Bob", 0.5)])

        with patch("random.choice", side_effect=lambda pool: pool[0]):
            target = bot._pick_random_vote_target(session)

        self.assertEqual(target, "Bob")

    def test_pick_random_vote_target_falls_back_to_full_pool_if_everyone_reads_town(self):
        bot = MafiaBot.__new__(MafiaBot)
        session = GameSession(source="test", raw_text="", players=["Alice", "Bob", "BotUser"])
        bot.tracker = SimpleNamespace(dead_players=set())
        bot.config = SimpleNamespace(
            showdown=SimpleNamespace(username="BotUser"),
            database=SimpleNamespace(db_path="dummy.db"),
        )
        bot.strategy = Mock()
        bot.strategy.min_confidence = 0.55
        # Both read as confidently town -- degenerate short-game case.
        bot.strategy.get_full_predictions = Mock(return_value=[("Alice", 0.05), ("Bob", 0.02)])

        with patch("random.choice", side_effect=lambda pool: pool[0]):
            target = bot._pick_random_vote_target(session)

        self.assertIn(target, ["Alice", "Bob"])

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
            gameplay=SimpleNamespace(town_read_comment_chance=1.0, vote_comment_chance=1.0, silent_mode=False),
        )
        bot._current_vote_target = None
        bot._current_town_read = "Alice"
        bot._send_chat_message = AsyncMock()
        bot.send_room_command = AsyncMock()

        asyncio.run(bot._evaluate_and_vote())

        bot._send_chat_message.assert_not_awaited()

    def test_cast_vote_with_optional_comment_silent_when_chance_is_zero(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.config = SimpleNamespace(gameplay=SimpleNamespace(vote_comment_chance=0.0, silent_mode=False))
        bot._send_chat_message = AsyncMock()
        bot.send_room_command = AsyncMock()

        asyncio.run(bot._cast_vote_with_optional_comment("Alice"))

        bot._send_chat_message.assert_not_awaited()
        bot.send_room_command.assert_awaited_once_with("/mafia vote Alice")

    def test_cast_vote_with_optional_comment_speaks_when_chance_is_one(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.config = SimpleNamespace(gameplay=SimpleNamespace(vote_comment_chance=1.0, silent_mode=False))
        bot._send_chat_message = AsyncMock()
        bot.send_room_command = AsyncMock()

        with patch("mafia_framework.bot.client.asyncio.sleep", new=AsyncMock()):
            asyncio.run(bot._cast_vote_with_optional_comment("Alice"))

        bot._send_chat_message.assert_awaited_once_with("I think Alice is scum")
        bot.send_room_command.assert_awaited_once_with("/mafia vote Alice")

    def test_cast_vote_with_optional_comment_silent_mode_still_votes_without_narration(self):
        # silent_mode must not stop the vote itself from happening -- only
        # the narration around it.
        bot = MafiaBot.__new__(MafiaBot)
        bot.config = SimpleNamespace(gameplay=SimpleNamespace(vote_comment_chance=1.0, silent_mode=True))
        bot._send_chat_message = AsyncMock()
        bot.send_room_command = AsyncMock()

        asyncio.run(bot._cast_vote_with_optional_comment("Alice"))

        bot._send_chat_message.assert_not_awaited()
        bot.send_room_command.assert_awaited_once_with("/mafia vote Alice")

    def test_random_actions_loop_does_nothing_in_silent_mode(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.config = SimpleNamespace(gameplay=SimpleNamespace(silent_mode=True))
        bot.tracker = SimpleNamespace(state="DAY", in_game=True, eliminated=False)
        bot._send_chat_message = AsyncMock()

        asyncio.run(bot._random_actions_loop())

        bot._send_chat_message.assert_not_awaited()

    def test_random_actions_loop_sends_at_most_one_message_per_cycle(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.config = SimpleNamespace(gameplay=SimpleNamespace(silent_mode=False))
        bot.tracker = SimpleNamespace(state="DAY", in_game=True, eliminated=False)
        bot._remembered_lines = ["some line"]

        async def stop_after_one_message(*args, **kwargs):
            bot.tracker.state = "NIGHT"

        bot._send_chat_message = AsyncMock(side_effect=stop_after_one_message)

        with patch("mafia_framework.bot.client.asyncio.sleep", new=AsyncMock()), \
             patch("random.choices", return_value=["filler"]):
            asyncio.run(bot._random_actions_loop())

        bot._send_chat_message.assert_awaited_once()

    def test_random_actions_loop_none_action_stays_quiet(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.config = SimpleNamespace(gameplay=SimpleNamespace(silent_mode=False))
        tracker = SimpleNamespace(state="DAY", in_game=True, eliminated=False)
        bot.tracker = tracker
        bot._remembered_lines = []
        bot._send_chat_message = AsyncMock()

        sleep_calls = 0

        async def fake_sleep(_):
            nonlocal sleep_calls
            sleep_calls += 1
            if sleep_calls >= 2:
                tracker.state = "NIGHT"

        with patch("mafia_framework.bot.client.asyncio.sleep", new=fake_sleep), \
             patch("random.choices", return_value=["none"]):
            asyncio.run(bot._random_actions_loop())

        bot._send_chat_message.assert_not_awaited()

    def test_evaluate_and_vote_suppresses_town_read_announcement_in_silent_mode(self):
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
            gameplay=SimpleNamespace(town_read_comment_chance=1.0, vote_comment_chance=1.0, silent_mode=True),
        )
        bot._current_vote_target = None
        bot._current_town_read = None
        bot._send_chat_message = AsyncMock()
        bot.send_room_command = AsyncMock()

        asyncio.run(bot._evaluate_and_vote())

        bot._send_chat_message.assert_not_awaited()
        # The read is still tracked internally even though it isn't announced.
        self.assertEqual(bot._current_town_read, "Alice")

    def test_deadline_events_trigger_reevaluation(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot._random_actions_task = None
        bot._random_vote_task = None
        bot._evaluate_and_vote = AsyncMock()

        for event in ("DEADLINE_3MIN", "DEADLINE_1MIN"):
            bot._evaluate_and_vote.reset_mock()
            asyncio.run(bot._handle_tracker_event(event))
            bot._evaluate_and_vote.assert_awaited_once()

    def test_started_event_requests_role_and_original_rolelist(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(state="NIGHT", in_game=True, eliminated=False)
        bot.strategy = Mock()
        bot.strategy.reset = Mock()
        bot.config = SimpleNamespace(gameplay=SimpleNamespace(silent_mode=False))
        bot._random_actions_task = None
        bot.send_room_command = AsyncMock()

        asyncio.run(bot._handle_tracker_event("STARTED"))

        bot.send_room_command.assert_any_call("/mafia role")
        bot.send_room_command.assert_any_call("/mafia originalrolelist")

    def test_votes_update_event_triggers_claim_check(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(state="DAY", in_game=True, eliminated=False)
        bot._maybe_claim_at_v1 = AsyncMock()

        asyncio.run(bot._handle_tracker_event("VOTES_UPDATE"))

        bot._maybe_claim_at_v1.assert_awaited_once()

    def test_votes_update_event_skipped_when_not_in_active_day(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(state="NIGHT", in_game=True, eliminated=False)
        bot._maybe_claim_at_v1 = AsyncMock()

        asyncio.run(bot._handle_tracker_event("VOTES_UPDATE"))

        bot._maybe_claim_at_v1.assert_not_awaited()

    def test_is_plurality_target_true_when_bot_has_the_most_votes(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(live_vote_counts={"BotUser": 3, "Alice": 2})
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="BotUser"))

        self.assertTrue(bot._is_plurality_target())

    def test_is_plurality_target_false_when_someone_else_leads(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(live_vote_counts={"BotUser": 1, "Alice": 3})
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="BotUser"))

        self.assertFalse(bot._is_plurality_target())

    def test_is_plurality_target_false_with_no_live_counts_yet(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(live_vote_counts={})
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="BotUser"))

        self.assertFalse(bot._is_plurality_target())

    def test_maybe_claim_if_plurality_near_deadline_claims_when_leading(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(live_vote_counts={"BotUser": 2, "Alice": 1})
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="BotUser"))
        bot._claimed_this_day = False
        bot._own_role = "Vanilla Townie"
        bot._send_chat_message = AsyncMock()

        asyncio.run(bot._maybe_claim_if_plurality_near_deadline())

        bot._send_chat_message.assert_awaited_once_with("I HARDCLAIM Vanilla Townie get OFF")
        self.assertTrue(bot._claimed_this_day)

    def test_maybe_claim_if_plurality_near_deadline_skips_when_already_claimed(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(live_vote_counts={"BotUser": 2, "Alice": 1})
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="BotUser"))
        bot._claimed_this_day = True
        bot._own_role = "Vanilla Townie"
        bot._send_chat_message = AsyncMock()

        asyncio.run(bot._maybe_claim_if_plurality_near_deadline())

        bot._send_chat_message.assert_not_awaited()

    def test_delayed_plurality_claim_check_checks_every_configured_checkpoint(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(state="DAY", in_game=True, eliminated=False)
        bot.config = SimpleNamespace(gameplay=SimpleNamespace(plurality_claim_check_seconds=[30.0, 20.0, 10.0, 5.0]))
        bot._claimed_this_day = False
        bot.send_room_command = AsyncMock()
        bot._maybe_claim_if_plurality_near_deadline = AsyncMock()

        with patch("mafia_framework.bot.client.asyncio.sleep", new=AsyncMock()):
            asyncio.run(bot._delayed_plurality_claim_check())

        self.assertEqual(bot._maybe_claim_if_plurality_near_deadline.call_count, 4)
        self.assertEqual(bot.send_room_command.call_count, 4)
        bot.send_room_command.assert_any_call("/mafia votes")

    def test_delayed_plurality_claim_check_stops_once_claimed(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(state="DAY", in_game=True, eliminated=False)
        bot.config = SimpleNamespace(gameplay=SimpleNamespace(plurality_claim_check_seconds=[30.0, 20.0, 10.0, 5.0]))
        bot._claimed_this_day = False
        bot.send_room_command = AsyncMock()

        async def claim_side_effect():
            bot._claimed_this_day = True

        bot._maybe_claim_if_plurality_near_deadline = AsyncMock(side_effect=claim_side_effect)

        with patch("mafia_framework.bot.client.asyncio.sleep", new=AsyncMock()):
            asyncio.run(bot._delayed_plurality_claim_check())

        self.assertEqual(bot._maybe_claim_if_plurality_near_deadline.call_count, 1)

    def test_deadline_1min_schedules_plurality_claim_check(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot._random_actions_task = None
        bot._evaluate_and_vote = AsyncMock()
        bot._delayed_plurality_claim_check = AsyncMock()

        with patch("mafia_framework.bot.client.asyncio.create_task") as mock_create_task:
            asyncio.run(bot._handle_tracker_event("DEADLINE_1MIN"))
            mock_create_task.assert_called_once()

    def test_deadline_3min_does_not_schedule_plurality_claim_check(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot._random_actions_task = None
        bot._evaluate_and_vote = AsyncMock()

        with patch("mafia_framework.bot.client.asyncio.create_task") as mock_create_task:
            asyncio.run(bot._handle_tracker_event("DEADLINE_3MIN"))
            mock_create_task.assert_not_called()

    def test_count_mafia_roles_counts_tokens_containing_mafia(self):
        self.assertEqual(MafiaBot._count_mafia_roles(["mafia", "ic", "vt"]), 1)
        self.assertEqual(MafiaBot._count_mafia_roles(["mafia roleblocker", "mafia goon", "vt", "cop"]), 2)
        self.assertEqual(MafiaBot._count_mafia_roles(["vt", "cop", "doctor"]), 0)

    def test_is_volo_false_without_rolelist_data(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(original_role_tokens=[], dead_players=set())
        session = GameSession(source="test", raw_text="", players=["Alice", "Bob"])

        self.assertFalse(bot._is_volo(session))

    def test_is_volo_true_when_mafia_near_parity(self):
        bot = MafiaBot.__new__(MafiaBot)
        # 1 mafia, 3 roles total; one town death already confirmed (not
        # mafia), so of the 2 alive, 1 is mafia -- that's parity.
        bot.tracker = SimpleNamespace(
            original_role_tokens=["mafia", "vt", "ic"],
            dead_players={"Ic"},
        )
        session = GameSession(
            source="test",
            raw_text="",
            players=["Alice", "Bob"],
            flips=[Flip(player_name="Ic", alignment="town")],
        )

        self.assertTrue(bot._is_volo(session))

    def test_is_volo_false_when_mafia_far_from_parity(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(original_role_tokens=["mafia", "vt", "vt", "vt", "vt"], dead_players=set())
        session = GameSession(source="test", raw_text="", players=["Alice", "Bob", "Carl", "Dave", "Eve"])

        self.assertFalse(bot._is_volo(session))

    def test_vote_detection_for_bot_username(self):
        self.assertTrue(MafiaBot._is_vote_for_bot("|c:|123|~|Alice has voted BotUser.", "BotUser"))
        self.assertTrue(MafiaBot._is_vote_for_bot("|c:|123|~|Alice voted for bot-user", "BotUser"))
        self.assertFalse(MafiaBot._is_vote_for_bot("|c:|123|~|Alice has voted Bob.", "BotUser"))

    def test_extract_vote_voter_name(self):
        self.assertEqual(MafiaBot._extract_vote_voter_name("|c:|123|~|Alice has voted BotUser."), "Alice")
        self.assertIsNone(MafiaBot._extract_vote_voter_name("|c:|123|~|Alice is just chatting"))

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
        # Any Mafia-aligned role (not just a specific named one) plus a set
        # of dangerous-to-reveal non-town roles from other alignments.
        for role in [
            "Werewolf", "Alien", "Cult Leader", "Serial Killer", "Goo",
            "Mafia Goon", "Mafia Roleblocker", "Mafia Boss",
        ]:
            bot._own_role = role
            self.assertEqual(bot._get_claim_message(), "Vanilla Townie", msg=f"role={role}")

    def test_get_claim_message_claims_real_role_otherwise(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot._own_role = "Vanilla Townie"
        self.assertEqual(bot._get_claim_message(), "Vanilla Townie")

        bot._own_role = "Doctor"
        self.assertEqual(bot._get_claim_message(), "Doctor")

    def test_get_claim_message_none_when_role_unknown(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot._own_role = None
        self.assertIsNone(bot._get_claim_message())

    def test_finished_event_sends_ragebait_after_gg(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot._random_actions_task = None
        bot.tracker = SimpleNamespace(players=["Alice", "BotUser"], dead_players={"Bob"}, reset=Mock())
        bot.strategy = Mock()
        bot.config = SimpleNamespace(
            gameplay=SimpleNamespace(silent_mode=False),
            showdown=SimpleNamespace(username="BotUser"),
        )
        bot.send_room_command = AsyncMock()
        bot._send_chat_message = AsyncMock()
        bot._save_game_to_db = AsyncMock()

        asyncio.run(bot._handle_tracker_event("FINISHED"))

        bot.send_room_command.assert_awaited_once_with("gg")
        bot._send_chat_message.assert_awaited_once()
        sent_text = bot._send_chat_message.call_args[0][0]
        matching_line = next(line for line in RAGEBAIT_LINES if sent_text.startswith(line))
        self.assertEqual(sent_text, f"{matching_line} Alice Bob")

    def test_finished_event_skips_ragebait_in_silent_mode(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot._random_actions_task = None
        bot.tracker = Mock()
        bot.strategy = Mock()
        bot.config = SimpleNamespace(gameplay=SimpleNamespace(silent_mode=True))
        bot.send_room_command = AsyncMock()
        bot._send_chat_message = AsyncMock()
        bot._save_game_to_db = AsyncMock()

        asyncio.run(bot._handle_tracker_event("FINISHED"))

        bot.send_room_command.assert_awaited_once_with("gg")
        bot._send_chat_message.assert_not_awaited()

    def test_handle_pm_claim_uses_own_role(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(
            in_game=True, eliminated=False, players=["Alice", "Bob"], dead_players=set(),
            hammer_count=None, get_game_session=Mock(return_value=None),
        )
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="BotUser"))
        bot.connection = Mock()
        bot.connection.send = AsyncMock()
        bot._own_role = "Cult Leader"

        asyncio.run(bot._handle_pm("Alice", ".claim"))

        bot.connection.send.assert_awaited_once_with("|/pm Alice, Vanilla Townie")

    def test_handle_pm_claim_appends_1_to_hammer_when_actually_at_v1(self):
        bot = MafiaBot.__new__(MafiaBot)
        session = GameSession(
            source="test",
            raw_text="",
            players=["Alice", "BotUser"],
            votes=[Vote(voter_name="Alice", target_name="BotUser", day=1, action="vote")],
        )
        bot.tracker = SimpleNamespace(
            in_game=True, eliminated=False, players=["Alice", "BotUser"], dead_players=set(),
            hammer_count=2, get_game_session=Mock(return_value=session),
        )
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="BotUser"))
        bot.connection = Mock()
        bot.connection.send = AsyncMock()
        bot._own_role = "Vanilla Townie"

        asyncio.run(bot._handle_pm("Alice", ".claim"))

        bot.connection.send.assert_awaited_once_with("|/pm Alice, Vanilla Townie 1 to hammer")

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

    def test_parse_own_role_box_from_mafia_role_response(self):
        # Real captured response to sending "/mafia role" in a live game.
        line = '|c|~|/raw <div class="infobox">Your role is: Mafia Goon</div>'
        self.assertEqual(MafiaBot._parse_own_role_box(line), "Mafia Goon")

    def test_parse_own_role_box_returns_none_for_unrelated_line(self):
        self.assertIsNone(MafiaBot._parse_own_role_box("|c:|123|Alice|hello everyone"))

    def test_handle_pm_ignores_own_echoed_messages(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(in_game=True, eliminated=False, players=["Alice", "Bob"], dead_players=set())
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="zorq_bot"))
        bot.connection = Mock()
        bot.connection.send = AsyncMock()

        asyncio.run(bot._handle_pm("zorq_bot", "please send me a random name"))

        bot.connection.send.assert_not_awaited()

    def test_format_reads_message(self):
        self.assertEqual(
            MafiaBot._format_reads_message([("Bob", 0.753), ("Alice", 0.10)]),
            "Bob 75% | Alice 10%",
        )
        self.assertEqual(MafiaBot._format_reads_message([]), "no reads available")

    def test_handle_pm_reads_command_returns_full_ranked_list(self):
        bot = MafiaBot.__new__(MafiaBot)
        session = GameSession(source="test", raw_text="", players=["Alice", "Bob"])
        bot.tracker = SimpleNamespace(
            in_game=True, eliminated=False, players=["Alice", "Bob"], dead_players=set(),
            get_game_session=Mock(return_value=session),
        )
        bot.config = SimpleNamespace(
            showdown=SimpleNamespace(username="BotUser"),
            database=SimpleNamespace(db_path="dummy.db"),
        )
        bot.strategy = Mock()
        bot.strategy.get_full_predictions = Mock(return_value=[("Bob", 0.82), ("Alice", 0.20)])
        bot.connection = Mock()
        bot.connection.send = AsyncMock()

        asyncio.run(bot._handle_pm("Host", ".reads"))

        bot.connection.send.assert_awaited_once_with("|/pm Host, Bob 82% | Alice 20%")

    def test_handle_pm_vote_command_casts_a_direct_vote(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(in_game=True, eliminated=False, players=["Alice", "Bob"], dead_players=set())
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="BotUser"))
        bot.connection = Mock()
        bot.connection.room = "mafia"
        bot.connection.send = AsyncMock()
        bot._current_vote_target = None

        asyncio.run(bot._handle_pm("Host", ".vote Alice"))

        bot.connection.send.assert_any_call("mafia|/mafia vote Alice")
        bot.connection.send.assert_any_call("|/pm Host, Voted Alice.")
        self.assertEqual(bot._current_vote_target, "Alice")

    def test_handle_pm_vote_rejects_nonexistent_player(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(in_game=True, eliminated=False, players=["Alice", "Bob"], dead_players=set())
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="BotUser"))
        bot.connection = Mock()
        bot.connection.room = "mafia"
        bot.connection.send = AsyncMock()
        bot._current_vote_target = None

        # Typo: "pomegrenato" vs the real player "Pomegranato".
        bot.tracker.players = ["Pomegranato", "lordsnackquaza"]
        asyncio.run(bot._handle_pm("Host", ".vote pomegrenato"))

        bot.connection.send.assert_awaited_once_with("|/pm Host, pomegrenato is not a real player.")
        self.assertIsNone(bot._current_vote_target)

    def test_delayed_first_evaluation_waits_configured_seconds(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(state="DAY", eliminated=False)
        bot.config = SimpleNamespace(gameplay=SimpleNamespace(first_evaluation_delay_seconds=60.0))
        bot._evaluate_and_vote = AsyncMock()

        with patch("mafia_framework.bot.client.asyncio.sleep", new=AsyncMock()) as mock_sleep:
            asyncio.run(bot._delayed_first_evaluation())

        mock_sleep.assert_awaited_once_with(60.0)
        bot._evaluate_and_vote.assert_awaited_once()

    def test_delayed_first_evaluation_skips_if_day_already_over(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(state="NIGHT", eliminated=False)
        bot.config = SimpleNamespace(gameplay=SimpleNamespace(first_evaluation_delay_seconds=60.0))
        bot._evaluate_and_vote = AsyncMock()

        with patch("mafia_framework.bot.client.asyncio.sleep", new=AsyncMock()):
            asyncio.run(bot._delayed_first_evaluation())

        bot._evaluate_and_vote.assert_not_awaited()

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

    def test_question_prompt_rallies_players_onto_current_vote_target(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = GameTracker()
        bot.tracker.state = "DAY"
        bot.tracker.players = ["Alice", "Bob", "Carl"]
        bot.tracker.in_game = True
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="BotUser"))
        bot._current_vote_target = "Carl"
        session = GameSession(source="test", raw_text="", players=["Alice", "Bob", "Carl"])

        with patch("random.random", return_value=0.0), patch("random.choice", side_effect=lambda pool: pool[0]):
            prompt = bot._build_question_prompt(session)

        self.assertEqual(prompt, "Alice vote Carl with me")

    def test_question_prompt_rally_excludes_the_vote_target_itself(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = GameTracker()
        bot.tracker.state = "DAY"
        bot.tracker.players = ["Alice", "Carl"]
        bot.tracker.in_game = True
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="BotUser"))
        bot._current_vote_target = "Carl"
        session = GameSession(source="test", raw_text="", players=["Alice", "Carl"])

        with patch("random.random", return_value=0.0), patch("random.choice", side_effect=lambda pool: pool[0]):
            prompt = bot._build_question_prompt(session)

        self.assertEqual(prompt, "Alice vote Carl with me")
        self.assertNotIn("Carl vote Carl", prompt)

    def test_question_prompt_skips_rally_without_a_current_vote_target(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = GameTracker()
        bot.tracker.state = "DAY"
        bot.tracker.players = ["Alice", "Bob"]
        bot.tracker.in_game = True
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="BotUser"))
        bot._current_vote_target = None
        session = GameSession(source="test", raw_text="", players=["Alice", "Bob"])

        with patch("random.random", return_value=0.0), patch("random.choices", return_value=[1]), patch("random.sample", side_effect=lambda items, k: items[:k]):
            prompt = bot._build_question_prompt(session)

        self.assertNotIn("with me", prompt)

    def test_started_event_clears_remembered_lines(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(state="NIGHT", in_game=False, eliminated=False)
        bot.strategy = Mock()
        bot.strategy.reset = Mock()
        bot.config = SimpleNamespace(gameplay=SimpleNamespace(silent_mode=False))
        bot._random_actions_task = None
        bot._remembered_lines = ["something said before the game started"]

        asyncio.run(bot._handle_tracker_event("STARTED"))

        self.assertEqual(bot._remembered_lines, [])

    def test_save_game_to_db_handles_no_stdin_gracefully(self):
        # Running headless/backgrounded means there's no terminal to answer
        # the save-confirmation prompt -- input() raises EOFError, which
        # must not escape and crash the message-processing loop.
        bot = MafiaBot.__new__(MafiaBot)
        session = GameSession(source="test", raw_text="some raw log text", players=["Alice", "Bob"])
        bot.tracker = SimpleNamespace(get_game_session=Mock(return_value=session))

        with patch("builtins.input", side_effect=EOFError):
            asyncio.run(bot._save_game_to_db())

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

    def test_reset_clears_raw_history_so_old_game_eliminations_dont_leak(self):
        # _prune_dead_players re-scans the entire raw_text_history on every
        # call. If reset() doesn't clear it, a stale elimination line from a
        # previous completed game (including one eliminating the bot itself)
        # would get re-detected as soon as the next game's first day marker
        # fires, silently marking the bot as already-eliminated in a brand
        # new game it's actually alive in.
        tracker = GameTracker()
        tracker.state = "DAY"
        tracker.players = ["Alice", "BotUser"]
        tracker.in_game = True

        tracker.process_message("|c:|1|~|BotUser was eliminated!", bot_username="BotUser")
        self.assertTrue(tracker.eliminated)

        tracker.reset()
        self.assertEqual(tracker.raw_text_history, [])

        # A brand new game starts and the bot is alive and playing.
        tracker.state = "DAY"
        tracker.players = ["Carl", "BotUser"]
        tracker.in_game = True
        tracker.process_message("Day 2. The hammer count is set at 2", bot_username="BotUser")

        self.assertFalse(tracker.eliminated)
        self.assertTrue(tracker.in_game)
        self.assertIn("BotUser", tracker.players)

    def test_duplicate_roster_broadcast_does_not_refire_started(self):
        # Seen live: the room re-broadcasts the exact same "**Players (N)**:"
        # line twice around game start (once at roster lock, once at the
        # actual start). The second one must not look like a brand new game.
        tracker = GameTracker()
        roster_line = "|c:|123|~|**Players (2)**: Alice, BotUser"

        first_event = tracker.process_message(roster_line, bot_username="BotUser")
        self.assertEqual(first_event, "STARTED")

        second_event = tracker.process_message(roster_line, bot_username="BotUser")
        self.assertIsNone(second_event)
        self.assertEqual(tracker.players, ["Alice", "BotUser"])
        self.assertTrue(tracker.in_game)

    def test_spectate_broadcast_does_not_override_confirmed_in_game(self):
        # Seen live: this "in progress / become a substitute / spectate"
        # broadcast is sent generically to anyone joining/refreshing the
        # room while a game is active -- including actual participants --
        # and it arrived (in the backlog replay) AFTER the roster line that
        # had already confirmed the bot was playing. It silently flipped
        # in_game back to False, breaking every subsequent vote reaction,
        # claim, and night action for a game the bot was still actually in.
        tracker = GameTracker()
        roster_line = "|c:|123|~|**Players (3)**: Alice, Bob, BotUser"
        spectate_line = (
            '|c:|124|~|/uhtml mafia,<div class="broadcast-blue">'
            '<p style="font-weight: bold">A game of Mafia is in progress.</p>'
            '<p><button class="button" name="send" value="/msgroom mafia,/mafia sub in">Become a substitute</button> '
            '<button class="button" name="send" value="/join view-mafia-mafia">Spectate the game</button></p></div>'
        )

        event = tracker.process_message(roster_line, bot_username="BotUser")
        self.assertEqual(event, "STARTED")
        self.assertTrue(tracker.in_game)

        second_event = tracker.process_message(spectate_line, bot_username="BotUser")
        self.assertIsNone(second_event)
        self.assertTrue(tracker.in_game)

    def test_spectate_broadcast_still_applies_when_genuinely_not_playing(self):
        tracker = GameTracker()
        tracker.state = "DAY"
        tracker.in_game = False
        spectate_line = (
            '|c:|124|~|/uhtml mafia,<div class="broadcast-blue">'
            '<p style="font-weight: bold">A game of Mafia is in progress.</p>'
            '<p><button class="button" name="send" value="/msgroom mafia,/mafia sub in">Become a substitute</button> '
            '<button class="button" name="send" value="/join view-mafia-mafia">Spectate the game</button></p></div>'
        )

        event = tracker.process_message(spectate_line, bot_username="BotUser")

        self.assertIsNone(event)
        self.assertFalse(tracker.in_game)

    def test_hammer_count_parsed_from_day_marker(self):
        tracker = GameTracker()
        tracker.state = "DAY"
        tracker.players = ["Alice", "Bob"]
        tracker.in_game = True

        tracker.process_message("Day 3. The hammer count is set at 4", bot_username="BotUser")
        self.assertEqual(tracker.hammer_count, 4)

    def test_parses_mafia_votes_response(self):
        tracker = GameTracker()
        tracker.state = "DAY"
        tracker.players = ["A Flowers Dream", "Brady1014", "mist"]
        tracker.in_game = True

        line = (
            '|c:|123|~|/raw <div class="infobox">Votes (Hammer: 2)<br />'
            "2* A Flowers Dream (Brady1014, mist)<br />"
            "1 Brady1014 (A Flowers Dream)</div>"
        )
        event = tracker.process_message(line, bot_username="mist")

        self.assertEqual(event, "VOTES_UPDATE")
        self.assertEqual(tracker.hammer_count, 2)
        self.assertEqual(
            tracker.live_vote_counts,
            {"A Flowers Dream": 2, "Brady1014": 1},
        )

    def test_votes_response_from_player_chat_is_ignored(self):
        # Only a genuine system ("~") message can update the live tally --
        # a player pasting the same text shouldn't be trusted.
        tracker = GameTracker()
        tracker.state = "DAY"
        tracker.players = ["Alice", "Bob"]
        tracker.in_game = True

        line = '|c:|123|Alice|Votes (Hammer: 2)<br />2* Bob (Alice)'
        event = tracker.process_message(line, bot_username="BotUser")

        self.assertIsNone(event)
        self.assertIsNone(tracker.hammer_count)
        self.assertEqual(tracker.live_vote_counts, {})

    def test_parses_mafia_votes_response_excludes_no_vote_bucket(self):
        # The real reply also lists a "No Vote" row for idling players --
        # seen live as e.g. "2 No Vote (Alice, Bob)" -- which isn't a real
        # target and shouldn't be treated as one.
        tracker = GameTracker()
        tracker.state = "DAY"
        tracker.players = ["Carl", "Alice", "Bob"]
        tracker.in_game = True

        line = (
            '|c:|123|~|/raw <div class="infobox">Votes (Hammer: 2)<br />'
            "1 Carl (Carl)<br />"
            "2 No Vote (Alice, Bob)</div>"
        )
        tracker.process_message(line, bot_username="Carl")

        self.assertEqual(tracker.live_vote_counts, {"Carl": 1})

    def test_parses_original_rolelist_response(self):
        tracker = GameTracker()
        tracker.state = "DAY"
        tracker.in_game = True

        event = tracker.process_message(
            "|c:|123|~|Original Rolelist: mafia, ic, vt", bot_username="BotUser"
        )

        self.assertIsNone(event)
        self.assertEqual(tracker.original_role_tokens, ["mafia", "ic", "vt"])

    def test_original_rolelist_from_player_chat_is_ignored(self):
        tracker = GameTracker()
        tracker.state = "DAY"
        tracker.in_game = True

        tracker.process_message(
            "|c:|123|Alice|Original Rolelist: mafia, ic, vt", bot_username="BotUser"
        )

        self.assertEqual(tracker.original_role_tokens, [])

    def test_player_chat_cannot_fake_phase_transitions(self):
        # Several phase-detection regexes have no anchor requiring them to
        # start at a line boundary, so without a sender check, a player
        # simply typing something that resembles a system announcement could
        # falsely flip the tracker's state.
        tracker = GameTracker()
        tracker.state = "DAY"
        tracker.players = ["Alice", "Bob"]
        tracker.in_game = True

        cases = [
            "|c:|123|Alice|lol night 2 has begun already",
            "|c:|124|Bob|the town has won this fr no cap",
            "|c:|125|Alice|**bold statement**",
            "|c:|126|Bob|Day 5. The hammer count is set at 3",
        ]
        for line in cases:
            event = tracker.process_message(line, bot_username="BotUser")
            self.assertIsNone(event, msg=f"player chat should never produce an event: {line!r}")
            self.assertEqual(tracker.state, "DAY", msg=f"player chat should never change state: {line!r}")

        # A genuine system-authored message must still work.
        event = tracker.process_message("|c:|127|~|Night 2 has begun.", bot_username="BotUser")
        self.assertEqual(event, "NIGHT")
        self.assertEqual(tracker.state, "NIGHT")

    def test_sub_replaces_player_in_roster_from_combined_message(self):
        # Real example: both sentences arrive bundled in one message.
        tracker = GameTracker()
        tracker.state = "DAY"
        tracker.players = ["Blue flare fusion", "Alice", "Bob"]
        tracker.in_game = True

        event = tracker.process_message(
            "|c:|1|~|Blue flare fusion has been subbed out. mist has joined the game.",
            bot_username="BotUser",
        )
        self.assertIsNone(event)
        self.assertEqual(tracker.players, ["mist", "Alice", "Bob"])

    def test_sub_replaces_player_across_separate_messages(self):
        tracker = GameTracker()
        tracker.state = "DAY"
        tracker.players = ["Charlie", "Dave"]
        tracker.in_game = True

        tracker.process_message("|c:|1|~|Charlie has been subbed out.", bot_username="BotUser")
        tracker.process_message("|c:|2|~|Eve has joined the game.", bot_username="BotUser")

        self.assertEqual(tracker.players, ["Eve", "Dave"])

    def test_sub_out_marks_bot_no_longer_in_game(self):
        tracker = GameTracker()
        tracker.state = "DAY"
        tracker.players = ["BotUser", "Alice"]
        tracker.in_game = True

        tracker.process_message(
            "|c:|1|~|BotUser has been subbed out. NewPlayer has joined the game.",
            bot_username="BotUser",
        )

        self.assertEqual(tracker.players, ["NewPlayer", "Alice"])
        self.assertFalse(tracker.in_game)

    def test_sub_in_marks_bot_now_in_game(self):
        tracker = GameTracker()
        tracker.state = "DAY"
        tracker.players = ["Alice", "Bob"]
        tracker.in_game = False

        tracker.process_message(
            "|c:|1|~|Alice has been subbed out. BotUser has joined the game.",
            bot_username="BotUser",
        )

        self.assertEqual(tracker.players, ["BotUser", "Bob"])
        self.assertTrue(tracker.in_game)

    def test_sub_ignores_player_chat_mentioning_subs(self):
        tracker = GameTracker()
        tracker.state = "DAY"
        tracker.players = ["Alice", "Bob"]
        tracker.in_game = True

        tracker.process_message(
            "|c:|1|Alice|lol Bob has been subbed out. troll has joined the game.",
            bot_username="BotUser",
        )

        self.assertEqual(tracker.players, ["Alice", "Bob"])

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
