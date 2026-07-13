import asyncio
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from mafia_framework.bot.client import MafiaBot, RAGEBAIT_LINES, CLANKER_OFFENDED_LINES
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

        # 2a. Roster lock -- just a roster update, not the real game start
        # (roles haven't been distributed yet at this point).
        event = tracker.process_message("|c:|1779642701|~|**Players (3)**: Alice, Bob, Charlie")
        self.assertEqual(tracker.state, "SIGNUPS")
        self.assertIsNone(event)
        self.assertEqual(tracker.players, ["Alice", "Bob", "Charlie"])

        # 2b. The explicit "game is starting" announcement is what actually
        # begins the game -- this is the rolling/role-distribution period,
        # not Day 1 yet, so state goes to NIGHT (no chatting) until the
        # real Day 1 marker arrives.
        event = tracker.process_message(
            '|c:|1779642701|~|The game of Mafia is starting!'
        )
        self.assertEqual(tracker.state, "NIGHT")
        self.assertEqual(event, "STARTED")

        # 3. Night starts -- state is already NIGHT from the STARTED
        # transition above, so this explicit announcement is a duplicate
        # of the same night and correctly produces no new event (see the
        # NIGHT_START_RE dedup guard).
        event = tracker.process_message("|c:|1779642701|~|Night 1 has begun.")
        self.assertEqual(tracker.state, "NIGHT")
        self.assertIsNone(event)

        # 4. Day starts
        event = tracker.process_message("Day 2. The hammer count is set at 2")
        self.assertEqual(tracker.state, "DAY")
        self.assertEqual(event, "DAY")
        self.assertEqual(tracker.current_day, 2)

        # 5. Game ends
        event = tracker.process_message("The Mafia has won!")
        self.assertEqual(tracker.state, "IDLE")
        self.assertEqual(event, "FINISHED")

    def test_night_detected_for_submit_action_or_idle_phrasing(self):
        # Confirmed live: some hosts phrase the night marker as "Night 2.
        # Submit whether you are using an action or idle..." instead of
        # "Night X has begun" -- this used to never register as NIGHT.
        tracker = GameTracker()
        tracker.state = "DAY"

        event = tracker.process_message(
            '|raw|<div class="broadcast-blue">Night 2. Submit whether you are using an '
            "action or idle. If you are using an action, DM your action to the host.</div>"
        )

        self.assertEqual(tracker.state, "NIGHT")
        self.assertEqual(event, "NIGHT")

    def test_night_detected_for_its_night_in_the_game_phrasing(self):
        tracker = GameTracker()
        tracker.state = "DAY"

        event = tracker.process_message(
            "|notify|It's night in the game of Mafia! Send in an action or idle."
        )

        self.assertEqual(tracker.state, "NIGHT")
        self.assertEqual(event, "NIGHT")

    def test_night_dedup_guard_prevents_duplicate_night_event(self):
        # Confirmed live: hosts send both "It's night in the game of
        # Mafia!" and "Night N. Submit whether..." for the *same* night --
        # without a dedup guard each one independently re-fired NIGHT,
        # causing the idle/kill action to be sent twice for one night.
        tracker = GameTracker()
        tracker.state = "DAY"

        first_event = tracker.process_message(
            "|notify|It's night in the game of Mafia! Send in an action or idle."
        )
        second_event = tracker.process_message(
            '|raw|<div class="broadcast-blue">Night 2. Submit whether you are using an '
            "action or idle. If you are using an action, DM your action to the host.</div>"
        )

        self.assertEqual(first_event, "NIGHT")
        self.assertIsNone(second_event)
        self.assertEqual(tracker.state, "NIGHT")

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

    def test_is_modexe_theme_true(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(theme="Modexe")
        self.assertTrue(bot._is_modexe_theme())

    def test_is_modexe_theme_true_for_modified_execution_name(self):
        # "Modexe" is a community nickname; the real on-the-record theme
        # name in the catalog is "Modified Execution".
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(theme="Modified Execution")
        self.assertTrue(bot._is_modexe_theme())

    def test_is_modexe_theme_true_for_cult_exe(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(theme="Cult.exe")
        self.assertTrue(bot._is_modexe_theme())

    def test_is_modexe_theme_true_for_mime_exe(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(theme="mime.exe")
        self.assertTrue(bot._is_modexe_theme())

    def test_is_modexe_theme_false_for_other_themes(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(theme="CCTV")
        self.assertFalse(bot._is_modexe_theme())

    def test_is_modexe_theme_false_when_no_theme_known(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(theme=None)
        self.assertFalse(bot._is_modexe_theme())

    def test_evaluate_and_vote_votes_town_read_when_modexe(self):
        # Voting = handing someone a gun in Modexe, so the bot should vote
        # its most-trusted town read instead of its top suspect.
        bot = MafiaBot.__new__(MafiaBot)
        session = GameSession(source="test", raw_text="", players=["Alice", "Bob"])
        bot.tracker = SimpleNamespace(
            state="DAY", eliminated=False, in_game=True, theme="Modexe",
            get_game_session=Mock(return_value=session),
        )
        bot.strategy = Mock()
        bot.strategy.get_vote_decision = Mock(return_value=(None, 0.0))
        bot.strategy.get_town_read = Mock(return_value=("Alice", 0.95))
        bot.config = SimpleNamespace(
            showdown=SimpleNamespace(username="BotUser"),
            database=SimpleNamespace(db_path="dummy.db"),
            gameplay=SimpleNamespace(town_read_comment_chance=1.0, vote_comment_chance=1.0, silent_mode=False),
        )
        bot._current_vote_target = None
        bot._current_town_read = None
        bot._send_chat_message = AsyncMock()
        bot.send_room_command = AsyncMock()

        with patch("mafia_framework.bot.client.asyncio.sleep", new=AsyncMock()):
            asyncio.run(bot._evaluate_and_vote())

        bot.strategy.get_vote_decision.assert_not_called()
        bot.send_room_command.assert_any_call("/mafia vote Alice")
        self.assertEqual(bot._current_vote_target, "Alice")

    def test_pick_random_vote_target_inverted_excludes_confident_scum_reads(self):
        bot = MafiaBot.__new__(MafiaBot)
        session = GameSession(source="test", raw_text="", players=["Alice", "Bob", "BotUser"])
        bot.tracker = SimpleNamespace(dead_players=set())
        bot.config = SimpleNamespace(
            showdown=SimpleNamespace(username="BotUser"),
            database=SimpleNamespace(db_path="dummy.db"),
        )
        bot.strategy = Mock()
        bot.strategy.min_confidence = 0.55
        # Alice reads confidently scum (90% mafia); Bob is a toss-up.
        bot.strategy.get_full_predictions = Mock(return_value=[("Alice", 0.9), ("Bob", 0.5)])

        with patch("random.choice", side_effect=lambda pool: pool[0]):
            target = bot._pick_random_vote_target(session, invert=True)

        self.assertEqual(target, "Bob")

    def test_is_popcorn_theme_true(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(theme="Popcorn")
        self.assertTrue(bot._is_popcorn_theme())

    def test_is_popcorn_theme_false_for_other_themes(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(theme="CCTV")
        self.assertFalse(bot._is_popcorn_theme())

    def test_check_gun_pickup_sets_flag_on_role_reveal(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(theme="Popcorn")
        bot._own_role = "Vanilla Townie"
        bot._has_gun = False
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="zorqbot"))

        line = '|raw|<div class="broadcast-blue">zorqbot\'s role was <span style="font-weight:bold;color:#060">Vanilla Townie</span>.</div>'
        bot._check_gun_pickup(line)

        self.assertTrue(bot._has_gun)

    def test_check_gun_pickup_sets_flag_on_plain_has_gun_announcement_from_host(self):
        # Confirmed live: the host's announcement is plain text, not bolded.
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(theme="Popcorn", host="ghostlyplanets")
        bot._own_role = "Vanilla Townie"
        bot._has_gun = False
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="zorqbot"))

        bot._check_gun_pickup("|c:|1783690899|+ghostlyplanets|zorqbot has gun")

        self.assertTrue(bot._has_gun)

    def test_check_gun_pickup_ignores_plain_has_gun_from_non_host(self):
        # A regular player saying the exact same words (joke/guess) shouldn't
        # be trusted -- only the host's word counts when it's not bolded.
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(theme="Popcorn", host="ghostlyplanets")
        bot._own_role = "Vanilla Townie"
        bot._has_gun = False
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="zorqbot"))

        bot._check_gun_pickup("|c:|1783690899|Lunarmob|zorqbot has gun")

        self.assertFalse(bot._has_gun)

    def test_check_gun_pickup_sets_flag_on_bolded_has_gun(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(theme="Popcorn")
        bot._own_role = "Vanilla Townie"
        bot._has_gun = False
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="zorqbot"))

        bot._check_gun_pickup("|c:|123|~|**zorqbot has gun**")

        self.assertTrue(bot._has_gun)

    def test_check_gun_pickup_ignores_unrelated_gun_chatter(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(theme="Popcorn")
        bot._own_role = "Vanilla Townie"
        bot._has_gun = False
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="zorqbot"))

        bot._check_gun_pickup("|c:|1783690854|+ghostlyplanets|who has gun bro")

        self.assertFalse(bot._has_gun)

    def test_check_gun_pickup_ignores_other_players_role_reveal(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(theme="Popcorn")
        bot._own_role = "Vanilla Townie"
        bot._has_gun = False
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="zorqbot"))

        line = '|raw|<div class="broadcast-blue">SomeoneElse\'s role was <span style="font-weight:bold;color:#060">Vanilla Townie</span>.</div>'
        bot._check_gun_pickup(line)

        self.assertFalse(bot._has_gun)

    def test_check_gun_pickup_noop_when_not_popcorn(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(theme="CCTV")
        bot._own_role = "Vanilla Townie"
        bot._has_gun = False
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="zorqbot"))

        line = '|raw|<div class="broadcast-blue">zorqbot\'s role was <span style="font-weight:bold;color:#060">Vanilla Townie</span>.</div>'
        bot._check_gun_pickup(line)

        self.assertFalse(bot._has_gun)

    def test_check_gun_pickup_noop_when_not_vanilla_townie(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(theme="Popcorn")
        bot._own_role = "Cop"
        bot._has_gun = False
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="zorqbot"))

        line = '|raw|<div class="broadcast-blue">zorqbot\'s role was <span style="font-weight:bold;color:#060">Vanilla Townie</span>.</div>'
        bot._check_gun_pickup(line)

        self.assertFalse(bot._has_gun)

    def test_evaluate_and_vote_shoots_instead_of_voting_when_gun_held(self):
        bot = MafiaBot.__new__(MafiaBot)
        session = GameSession(source="test", raw_text="", players=["Alice", "Bob"])
        bot.tracker = SimpleNamespace(
            state="DAY", eliminated=False, in_game=True, theme="Popcorn",
            get_game_session=Mock(return_value=session),
        )
        bot.strategy = Mock()
        bot.strategy.get_vote_decision = Mock(return_value=("Alice", 0.9))
        bot.strategy.get_town_read = Mock(return_value=(None, 0.0))
        bot.config = SimpleNamespace(
            showdown=SimpleNamespace(username="BotUser"),
            database=SimpleNamespace(db_path="dummy.db"),
            gameplay=SimpleNamespace(town_read_comment_chance=1.0, vote_comment_chance=1.0, silent_mode=False),
        )
        bot._current_vote_target = None
        bot._current_town_read = None
        bot._has_gun = True
        bot._send_chat_message = AsyncMock()
        bot.send_room_command = AsyncMock()

        asyncio.run(bot._evaluate_and_vote())

        bot._send_chat_message.assert_any_call("**shoot Alice**")
        bot.send_room_command.assert_not_called()
        self.assertEqual(bot._current_vote_target, "Alice")

    def test_choose_idea_pick_prefers_town_option(self):
        options = [
            ("mafiaoneshotstrongman", "Mafia One-Shot Strongman", "Mafia"),
            ("day2suicidalbulletproofpurplegoo", "Day 2 Suicidal Bulletproof Purple Goo", "Town"),
        ]
        self.assertEqual(
            MafiaBot._choose_idea_pick(options), "day2suicidalbulletproofpurplegoo"
        )

    def test_choose_idea_pick_falls_back_to_non_mafia_when_no_town(self):
        options = [
            ("mafiasecretagent", "Mafia Secret Agent", "Mafia"),
            ("traitorcelebrity", "Traitor Celebrity", None),
        ]
        self.assertEqual(MafiaBot._choose_idea_pick(options), "traitorcelebrity")

    def test_choose_idea_pick_still_picks_something_when_all_options_mafia(self):
        # Even the lowest-priority tier still gets picked (the first
        # listed, since it's a tie) rather than giving up entirely.
        options = [
            ("mafiasecretagent", "Mafia Secret Agent", "Mafia"),
            ("mafiaoneshotstrongman", "Mafia One-Shot Strongman", "Mafia"),
        ]
        self.assertEqual(MafiaBot._choose_idea_pick(options), "mafiasecretagent")

    def test_choose_idea_pick_prefers_third_party_over_serial_killer(self):
        options = [
            ("solo_survivor", "Survivor", None),
            ("solo_serial_killer", "Serial Killer", None),
        ]
        self.assertEqual(MafiaBot._choose_idea_pick(options), "solo_survivor")

    def test_choose_idea_pick_prefers_serial_killer_over_group_scum(self):
        options = [
            ("mafia_goon", "Mafia Goon", "Mafia"),
            ("solo_serial_killer", "Serial Killer", None),
        ]
        self.assertEqual(MafiaBot._choose_idea_pick(options), "solo_serial_killer")

    def test_choose_idea_pick_prefers_town_over_third_party(self):
        options = [
            ("solo_survivor", "Survivor", None),
            ("vt", "Vanilla Townie", "Town"),
        ]
        self.assertEqual(MafiaBot._choose_idea_pick(options), "vt")

    def test_choose_idea_pick_treats_werewolf_alien_cult_goo_as_group_scum(self):
        for role_name, alignment in [
            ("Werewolf Roleblocker", "Werewolf"),
            ("Alien Contrary", "Alien"),
            ("Cult Leader", None),
            ("Day 2 Suicidal Cult One-Shot Goomaker", None),
            ("Replicant Roleblocker", None),
        ]:
            options = [
                ("scum_role", role_name, alignment),
                ("solo_survivor", "Survivor", None),
            ]
            self.assertEqual(
                MafiaBot._choose_idea_pick(options), "solo_survivor", msg=f"role={role_name}"
            )

    def test_classify_idea_option_named_third_party_roles(self):
        for role_name in [
            "Judas", "Saulus", "Survivor", "1-Shot Townie", "Underdog", "Wild Card",
            "Wild Card GI",  # a suffixed variant seen in real IDEA role lists
        ]:
            self.assertEqual(
                MafiaBot._classify_idea_option(role_name, None), 2, msg=f"role={role_name}"
            )

    def test_maybe_pick_idea_role_sends_ideapick_command_for_town_option(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = GameTracker()
        bot._idea_last_candidates = None
        bot.send_room_command = AsyncMock()

        line = (
            '|pagehtml|<div class="pad broadcast-blue"><h3>Host: Overkill_Tuna</h3>'
            '<p><b>IDEA information:</b><br /><b>role:</b> '
            '<button class="button disabled" style="color:#575757;">clear</button>'
            '<button class="button" name="send" value="/msgroom mafia,/mafia ideapick role, day2suicidalbulletproofpurplegoo">Day 2 Suicidal Bulletproof Purple Goo</button>'
            '<button class="button" name="send" value="/msgroom mafia,/mafia ideapick role, mafiaoneshotstrongman">Mafia One-Shot Strongman</button><br /></p>'
            '<p><details><summary class="button"><b>Role details:</b></summary>'
            '<p><details><summary>Day 2 Suicidal Bulletproof Purple Goo</summary><table><tr><td><ul>'
            '<li>You are aligned with the <span style="color:#060;">Town</span>. You win...</li></ul></td></tr></table></details>'
            '<details><summary>Mafia One-Shot Strongman</summary><table><tr><td><ul>'
            '<li>You are aligned with the <span style="color:#F00;">Mafia</span>. You win...</li></ul></td></tr></table></details>'
            '</p></details></p></div>'
        )

        asyncio.run(bot._maybe_pick_idea_role(line))

        bot.send_room_command.assert_called_once_with("/mafia ideapick role, day2suicidalbulletproofpurplegoo")
        self.assertEqual(
            bot._idea_last_candidates,
            frozenset({"day2suicidalbulletproofpurplegoo", "mafiaoneshotstrongman"}),
        )

    def test_maybe_pick_idea_role_skips_when_same_round_already_acted_on(self):
        # Post-pick panel state: the chosen option is now disabled/dropped
        # out, only the other original option ("bar") still shows as
        # clickable -- a subset of the round already acted on, not new.
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = GameTracker()
        bot._idea_last_candidates = frozenset({"foo", "bar"})
        bot.send_room_command = AsyncMock()

        line = (
            '|pagehtml|<div><p><b>IDEA information:</b><br /><b>role:</b> '
            '<button class="button" name="send" value="/msgroom mafia,/mafia ideapick role, bar">Bar</button></p></div>'
        )

        asyncio.run(bot._maybe_pick_idea_role(line))

        bot.send_room_command.assert_not_called()

    def test_maybe_pick_idea_role_reacts_to_a_new_round_with_different_options(self):
        # A game can run through more than one IDEA round -- a fresh set
        # of options (not overlapping with the last round acted on)
        # should be picked again, not permanently ignored.
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = GameTracker()
        bot._idea_last_candidates = frozenset({"foo", "bar"})
        bot.send_room_command = AsyncMock()

        line = (
            '|pagehtml|<div><p><b>IDEA information:</b><br /><b>role:</b> '
            '<button class="button" name="send" value="/msgroom mafia,/mafia ideapick role, baz">Baz</button>'
            '<button class="button" name="send" value="/msgroom mafia,/mafia ideapick role, qux">Qux</button></p></div>'
        )

        asyncio.run(bot._maybe_pick_idea_role(line))

        bot.send_room_command.assert_called_once_with("/mafia ideapick role, baz")
        self.assertEqual(bot._idea_last_candidates, frozenset({"baz", "qux"}))

    def test_maybe_react_to_clanker_reacts_and_shifts_vote_during_day(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(state="DAY", in_game=True, eliminated=False)
        bot.config = SimpleNamespace(
            gameplay=SimpleNamespace(troll_mode=True, silent_mode=False),
            showdown=SimpleNamespace(username="zorqbot"),
        )
        bot._current_vote_target = None
        bot._send_chat_message = AsyncMock()
        bot.send_room_command = AsyncMock()

        with patch("mafia_framework.bot.client.asyncio.sleep", new=AsyncMock()):
            asyncio.run(bot._maybe_react_to_clanker("|c:|123|Lunarmob|ur just a clanker lol"))

        bot._send_chat_message.assert_called_once()
        self.assertIn(bot._send_chat_message.call_args[0][0], CLANKER_OFFENDED_LINES)
        bot.send_room_command.assert_called_once_with("/mafia vote Lunarmob")
        self.assertEqual(bot._current_vote_target, "Lunarmob")

    def test_maybe_react_to_clanker_noop_when_troll_mode_off(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(state="DAY", in_game=True, eliminated=False)
        bot.config = SimpleNamespace(
            gameplay=SimpleNamespace(troll_mode=False, silent_mode=False),
            showdown=SimpleNamespace(username="zorqbot"),
        )
        bot._send_chat_message = AsyncMock()
        bot.send_room_command = AsyncMock()

        asyncio.run(bot._maybe_react_to_clanker("|c:|123|Lunarmob|ur just a clanker lol"))

        bot._send_chat_message.assert_not_called()
        bot.send_room_command.assert_not_called()

    def test_maybe_react_to_clanker_noop_when_silent_mode_on(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(state="DAY", in_game=True, eliminated=False)
        bot.config = SimpleNamespace(
            gameplay=SimpleNamespace(troll_mode=True, silent_mode=True),
            showdown=SimpleNamespace(username="zorqbot"),
        )
        bot._send_chat_message = AsyncMock()
        bot.send_room_command = AsyncMock()

        asyncio.run(bot._maybe_react_to_clanker("|c:|123|Lunarmob|ur just a clanker lol"))

        bot._send_chat_message.assert_not_called()
        bot.send_room_command.assert_not_called()

    def test_maybe_react_to_clanker_ignores_own_echoed_message(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(state="DAY", in_game=True, eliminated=False)
        bot.config = SimpleNamespace(
            gameplay=SimpleNamespace(troll_mode=True, silent_mode=False),
            showdown=SimpleNamespace(username="zorqbot"),
        )
        bot._send_chat_message = AsyncMock()
        bot.send_room_command = AsyncMock()

        asyncio.run(bot._maybe_react_to_clanker("|c:|123|zorqbot|clanker"))

        bot._send_chat_message.assert_not_called()
        bot.send_room_command.assert_not_called()

    def test_maybe_react_to_clanker_no_vote_shift_outside_day(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(state="NIGHT", in_game=True, eliminated=False)
        bot.config = SimpleNamespace(
            gameplay=SimpleNamespace(troll_mode=True, silent_mode=False),
            showdown=SimpleNamespace(username="zorqbot"),
        )
        bot._current_vote_target = None
        bot._send_chat_message = AsyncMock()
        bot.send_room_command = AsyncMock()

        with patch("mafia_framework.bot.client.asyncio.sleep", new=AsyncMock()):
            asyncio.run(bot._maybe_react_to_clanker("|c:|123|Lunarmob|clanker"))

        bot._send_chat_message.assert_called_once()
        bot.send_room_command.assert_not_called()

    def test_maybe_react_to_clanker_ignores_unrelated_message(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(state="DAY", in_game=True, eliminated=False)
        bot.config = SimpleNamespace(
            gameplay=SimpleNamespace(troll_mode=True, silent_mode=False),
            showdown=SimpleNamespace(username="zorqbot"),
        )
        bot._send_chat_message = AsyncMock()
        bot.send_room_command = AsyncMock()

        asyncio.run(bot._maybe_react_to_clanker("|c:|123|Lunarmob|hello everyone"))

        bot._send_chat_message.assert_not_called()
        bot.send_room_command.assert_not_called()

    def test_evaluate_and_vote_skips_when_already_in_progress(self):
        # Confirmed live: model inference plus optional comment delays can
        # take several seconds, and _evaluate_and_vote gets triggered from
        # more than one place (day start, deadline warnings) -- a second
        # call arriving before the first finishes must not re-run the
        # whole evaluation (that used to send the same read/vote twice).
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(state="DAY", eliminated=False, in_game=True)
        bot.tracker.get_game_session = Mock(return_value=GameSession(source="test", raw_text="", players=["Alice"]))
        bot._evaluating_vote = True
        bot._evaluate_and_vote_impl = AsyncMock()

        asyncio.run(bot._evaluate_and_vote())

        bot._evaluate_and_vote_impl.assert_not_awaited()

    def test_evaluate_and_vote_clears_guard_after_running(self):
        bot = MafiaBot.__new__(MafiaBot)
        session = GameSession(source="test", raw_text="", players=["Alice"])
        bot.tracker = SimpleNamespace(
            state="DAY", eliminated=False, in_game=True, get_game_session=Mock(return_value=session)
        )
        bot._evaluating_vote = False
        bot._evaluate_and_vote_impl = AsyncMock()

        asyncio.run(bot._evaluate_and_vote())

        bot._evaluate_and_vote_impl.assert_awaited_once_with(False, session)
        self.assertFalse(bot._evaluating_vote)

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
        bot.config = SimpleNamespace(
            gameplay=SimpleNamespace(silent_mode=False, random_action_interval_seconds=[180.0, 300.0])
        )
        bot.tracker = SimpleNamespace(state="DAY", in_game=True, eliminated=False)
        bot._remembered_lines = ["some line"]

        async def stop_after_one_message(*args, **kwargs):
            bot.tracker.state = "NIGHT"

        bot._send_chat_message = AsyncMock(side_effect=stop_after_one_message)

        with patch("mafia_framework.bot.client.asyncio.sleep", new=AsyncMock()), \
             patch("random.choices", return_value=["reaction"]):
            asyncio.run(bot._random_actions_loop())

        bot._send_chat_message.assert_awaited_once()

    def test_random_actions_loop_none_action_stays_quiet(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.config = SimpleNamespace(
            gameplay=SimpleNamespace(silent_mode=False, random_action_interval_seconds=[180.0, 300.0])
        )
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

    def test_delayed_autojoin_waits_configured_seconds_then_joins(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.config = SimpleNamespace(gameplay=SimpleNamespace(autojoin_delay_seconds=5.0))
        bot.send_room_command = AsyncMock()

        with patch("mafia_framework.bot.client.asyncio.sleep", new=AsyncMock()) as mock_sleep:
            asyncio.run(bot._delayed_autojoin())

        mock_sleep.assert_awaited_once_with(5.0)
        bot.send_room_command.assert_awaited_once_with("/mafia join")

    def test_delayed_autojoin_skips_sleep_when_delay_is_zero(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.config = SimpleNamespace(gameplay=SimpleNamespace(autojoin_delay_seconds=0.0))
        bot.send_room_command = AsyncMock()

        with patch("mafia_framework.bot.client.asyncio.sleep", new=AsyncMock()) as mock_sleep:
            asyncio.run(bot._delayed_autojoin())

        mock_sleep.assert_not_awaited()
        bot.send_room_command.assert_awaited_once_with("/mafia join")

    def test_signups_event_schedules_delayed_autojoin(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.config = SimpleNamespace(gameplay=SimpleNamespace(autojoin=True))
        bot._delayed_autojoin = AsyncMock()

        with patch("mafia_framework.bot.client.asyncio.create_task") as mock_create_task:
            asyncio.run(bot._handle_tracker_event("SIGNUPS"))
            mock_create_task.assert_called_once()

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
        bot._maybe_defend_town_plurality = AsyncMock()
        bot._maybe_quickhammer = AsyncMock()

        asyncio.run(bot._handle_tracker_event("VOTES_UPDATE"))

        bot._maybe_claim_at_v1.assert_awaited_once()
        bot._maybe_defend_town_plurality.assert_awaited_once()
        bot._maybe_quickhammer.assert_awaited_once()

    def test_votes_update_event_skipped_when_not_in_active_day(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(state="NIGHT", in_game=True, eliminated=False)
        bot._maybe_claim_at_v1 = AsyncMock()
        bot._maybe_defend_town_plurality = AsyncMock()
        bot._maybe_quickhammer = AsyncMock()

        asyncio.run(bot._handle_tracker_event("VOTES_UPDATE"))

        bot._maybe_claim_at_v1.assert_not_awaited()
        bot._maybe_defend_town_plurality.assert_not_awaited()
        bot._maybe_quickhammer.assert_not_awaited()

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

    def test_find_quickhammer_target_returns_sole_v1_player(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(hammer_count=3, live_vote_counts={"Alice": 2, "Bob": 1}, partners=[])
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="BotUser"))

        self.assertEqual(bot._find_quickhammer_target(), "Alice")

    def test_find_quickhammer_target_excludes_partners(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(hammer_count=3, live_vote_counts={"Alice": 2}, partners=["Alice"])
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="BotUser"))

        self.assertIsNone(bot._find_quickhammer_target())

    def test_find_quickhammer_target_excludes_self(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(hammer_count=3, live_vote_counts={"BotUser": 2}, partners=[])
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="BotUser"))

        self.assertIsNone(bot._find_quickhammer_target())

    def test_find_quickhammer_target_none_when_nobody_at_v1(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(hammer_count=3, live_vote_counts={"Alice": 1}, partners=[])
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="BotUser"))

        self.assertIsNone(bot._find_quickhammer_target())

    def test_find_quickhammer_target_none_when_multiple_at_v1(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(hammer_count=3, live_vote_counts={"Alice": 2, "Carl": 2}, partners=[])
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="BotUser"))

        self.assertIsNone(bot._find_quickhammer_target())

    def test_maybe_quickhammer_instantly_hammers_in_volo(self):
        bot = MafiaBot.__new__(MafiaBot)
        session = GameSession(source="test", raw_text="", players=["Alice", "Bob", "BotUser"])
        bot.tracker = SimpleNamespace(
            hammer_count=3, live_vote_counts={"Alice": 2}, partners=[],
            original_role_tokens=["mafia", "vt", "vt"], dead_players=set(),
            get_game_session=Mock(return_value=session),
        )
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="BotUser"))
        bot._own_role = "Mafia Goon"
        bot._current_vote_target = None
        bot.send_room_command = AsyncMock()

        asyncio.run(bot._maybe_quickhammer())

        bot.send_room_command.assert_awaited_once_with("/mafia vote Alice")
        self.assertEqual(bot._current_vote_target, "Alice")

    def test_maybe_quickhammer_skipped_when_not_volo(self):
        bot = MafiaBot.__new__(MafiaBot)
        session = GameSession(
            source="test", raw_text="", players=["Alice", "Bob", "Carl", "Dave", "Eve", "BotUser"]
        )
        bot.tracker = SimpleNamespace(
            hammer_count=4, live_vote_counts={"Alice": 3}, partners=[],
            original_role_tokens=["mafia", "vt", "vt", "vt", "vt", "vt"], dead_players=set(),
            get_game_session=Mock(return_value=session),
        )
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="BotUser"))
        bot._own_role = "Mafia Goon"
        bot._current_vote_target = None
        bot.send_room_command = AsyncMock()

        asyncio.run(bot._maybe_quickhammer())

        bot.send_room_command.assert_not_awaited()

    def test_maybe_quickhammer_skipped_when_not_mafia(self):
        bot = MafiaBot.__new__(MafiaBot)
        session = GameSession(source="test", raw_text="", players=["Alice", "Bob", "BotUser"])
        bot.tracker = SimpleNamespace(
            hammer_count=3, live_vote_counts={"Alice": 2}, partners=[],
            original_role_tokens=["mafia", "vt", "vt"], dead_players=set(),
            get_game_session=Mock(return_value=session),
        )
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="BotUser"))
        bot._own_role = "Vanilla Townie"
        bot._current_vote_target = None
        bot.send_room_command = AsyncMock()

        asyncio.run(bot._maybe_quickhammer())

        bot.send_room_command.assert_not_awaited()

    def test_maybe_quickhammer_does_not_repeat_same_target(self):
        bot = MafiaBot.__new__(MafiaBot)
        session = GameSession(source="test", raw_text="", players=["Alice", "Bob", "BotUser"])
        bot.tracker = SimpleNamespace(
            hammer_count=3, live_vote_counts={"Alice": 2}, partners=[],
            original_role_tokens=["mafia", "vt", "vt"], dead_players=set(),
            get_game_session=Mock(return_value=session),
        )
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="BotUser"))
        bot._own_role = "Mafia Goon"
        bot._current_vote_target = "Alice"
        bot.send_room_command = AsyncMock()

        asyncio.run(bot._maybe_quickhammer())

        bot.send_room_command.assert_not_awaited()

    def test_get_plurality_leader_returns_sole_leader(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(live_vote_counts={"Alice": 3, "Bob": 1})
        self.assertEqual(bot._get_plurality_leader(), "Alice")

    def test_get_plurality_leader_none_on_tie(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(live_vote_counts={"Alice": 2, "Bob": 2})
        self.assertIsNone(bot._get_plurality_leader())

    def test_get_plurality_leader_none_when_no_votes(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(live_vote_counts={})
        self.assertIsNone(bot._get_plurality_leader())

    def test_maybe_defend_town_plurality_reacts_when_very_confident(self):
        bot = MafiaBot.__new__(MafiaBot)
        session = GameSession(source="test", raw_text="", players=["Alice", "Bob"])
        bot.tracker = SimpleNamespace(
            theme="CCTV", live_vote_counts={"Alice": 3, "Bob": 1},
            get_game_session=Mock(return_value=session),
        )
        bot.strategy = Mock()
        bot.strategy.get_town_read = Mock(return_value=("Alice", 0.9))
        bot.config = SimpleNamespace(
            gameplay=SimpleNamespace(silent_mode=False, plurality_defense_min_confidence=0.85),
            showdown=SimpleNamespace(username="BotUser"),
            database=SimpleNamespace(db_path="dummy.db"),
        )
        bot._defended_plurality_target = None
        bot._send_chat_message = AsyncMock()

        asyncio.run(bot._maybe_defend_town_plurality())

        bot._send_chat_message.assert_awaited_once()
        sent_text = bot._send_chat_message.call_args[0][0]
        self.assertIn("Alice", sent_text)
        self.assertEqual(bot._defended_plurality_target, "Alice")

    def test_maybe_defend_town_plurality_skips_below_confidence_bar(self):
        bot = MafiaBot.__new__(MafiaBot)
        session = GameSession(source="test", raw_text="", players=["Alice", "Bob"])
        bot.tracker = SimpleNamespace(
            theme="CCTV", live_vote_counts={"Alice": 3, "Bob": 1},
            get_game_session=Mock(return_value=session),
        )
        bot.strategy = Mock()
        bot.strategy.get_town_read = Mock(return_value=("Alice", 0.6))
        bot.config = SimpleNamespace(
            gameplay=SimpleNamespace(silent_mode=False, plurality_defense_min_confidence=0.85),
            showdown=SimpleNamespace(username="BotUser"),
            database=SimpleNamespace(db_path="dummy.db"),
        )
        bot._defended_plurality_target = None
        bot._send_chat_message = AsyncMock()

        asyncio.run(bot._maybe_defend_town_plurality())

        bot._send_chat_message.assert_not_awaited()

    def test_maybe_defend_town_plurality_skips_when_leader_not_the_town_read(self):
        bot = MafiaBot.__new__(MafiaBot)
        session = GameSession(source="test", raw_text="", players=["Alice", "Bob"])
        bot.tracker = SimpleNamespace(
            theme="CCTV", live_vote_counts={"Alice": 3, "Bob": 1},
            get_game_session=Mock(return_value=session),
        )
        bot.strategy = Mock()
        bot.strategy.get_town_read = Mock(return_value=("Bob", 0.95))
        bot.config = SimpleNamespace(
            gameplay=SimpleNamespace(silent_mode=False, plurality_defense_min_confidence=0.85),
            showdown=SimpleNamespace(username="BotUser"),
            database=SimpleNamespace(db_path="dummy.db"),
        )
        bot._defended_plurality_target = None
        bot._send_chat_message = AsyncMock()

        asyncio.run(bot._maybe_defend_town_plurality())

        bot._send_chat_message.assert_not_awaited()

    def test_maybe_defend_town_plurality_skips_when_modexe(self):
        bot = MafiaBot.__new__(MafiaBot)
        session = GameSession(source="test", raw_text="", players=["Alice", "Bob"])
        bot.tracker = SimpleNamespace(
            theme="Modexe", live_vote_counts={"Alice": 3, "Bob": 1},
            get_game_session=Mock(return_value=session),
        )
        bot.strategy = Mock()
        bot.strategy.get_town_read = Mock(return_value=("Alice", 0.95))
        bot.config = SimpleNamespace(
            gameplay=SimpleNamespace(silent_mode=False, plurality_defense_min_confidence=0.85),
            showdown=SimpleNamespace(username="BotUser"),
            database=SimpleNamespace(db_path="dummy.db"),
        )
        bot._defended_plurality_target = None
        bot._send_chat_message = AsyncMock()

        asyncio.run(bot._maybe_defend_town_plurality())

        bot._send_chat_message.assert_not_awaited()

    def test_maybe_defend_town_plurality_only_reacts_once_per_leader(self):
        bot = MafiaBot.__new__(MafiaBot)
        session = GameSession(source="test", raw_text="", players=["Alice", "Bob"])
        bot.tracker = SimpleNamespace(
            theme="CCTV", live_vote_counts={"Alice": 3, "Bob": 1},
            get_game_session=Mock(return_value=session),
        )
        bot.strategy = Mock()
        bot.strategy.get_town_read = Mock(return_value=("Alice", 0.95))
        bot.config = SimpleNamespace(
            gameplay=SimpleNamespace(silent_mode=False, plurality_defense_min_confidence=0.85),
            showdown=SimpleNamespace(username="BotUser"),
            database=SimpleNamespace(db_path="dummy.db"),
        )
        bot._defended_plurality_target = "Alice"
        bot._send_chat_message = AsyncMock()

        asyncio.run(bot._maybe_defend_town_plurality())

        bot._send_chat_message.assert_not_awaited()

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

    def test_is_mylo_false_without_rolelist_data(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(original_role_tokens=[], dead_players=set())
        session = GameSession(source="test", raw_text="", players=["Alice", "Bob"])

        self.assertFalse(bot._is_mylo(session))

    def test_is_mylo_true_at_threshold_one_mafia_four_players(self):
        # (mafia_alive * 2) + 2 = 4 -- exactly 4 players left with 1 mafia.
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(original_role_tokens=["mafia", "vt", "vt", "vt"], dead_players=set())
        session = GameSession(source="test", raw_text="", players=["Alice", "Bob", "Carl", "Dave"])

        self.assertTrue(bot._is_mylo(session))

    def test_is_mylo_true_at_threshold_two_mafia_six_players(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(
            original_role_tokens=["mafia", "mafia", "vt", "vt", "vt", "vt"], dead_players=set()
        )
        session = GameSession(
            source="test", raw_text="", players=["Alice", "Bob", "Carl", "Dave", "Eve", "Frank"]
        )

        self.assertTrue(bot._is_mylo(session))

    def test_is_mylo_false_above_threshold(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(
            original_role_tokens=["mafia", "vt", "vt", "vt", "vt"], dead_players=set()
        )
        session = GameSession(source="test", raw_text="", players=["Alice", "Bob", "Carl", "Dave", "Eve"])

        self.assertFalse(bot._is_mylo(session))

    def test_evaluate_and_vote_insta_novotes_in_mylo(self):
        bot = MafiaBot.__new__(MafiaBot)
        session = GameSession(source="test", raw_text="", players=["Alice", "Bob", "Carl", "Dave"])
        bot.tracker = SimpleNamespace(
            state="DAY", eliminated=False, in_game=True, theme="CCTV",
            original_role_tokens=["mafia", "vt", "vt", "vt"], dead_players=set(),
            get_game_session=Mock(return_value=session),
        )
        bot.strategy = Mock()
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

        bot.strategy.get_vote_decision.assert_not_called()
        bot.send_room_command.assert_any_call("/mafia vote novote")
        self.assertEqual(bot._current_vote_target, "novote")

    def test_evaluate_and_vote_mylo_does_not_repeat_novote_command(self):
        bot = MafiaBot.__new__(MafiaBot)
        session = GameSession(source="test", raw_text="", players=["Alice", "Bob", "Carl", "Dave"])
        bot.tracker = SimpleNamespace(
            state="DAY", eliminated=False, in_game=True, theme="CCTV",
            original_role_tokens=["mafia", "vt", "vt", "vt"], dead_players=set(),
            get_game_session=Mock(return_value=session),
        )
        bot.strategy = Mock()
        bot.strategy.get_town_read = Mock(return_value=(None, 0.0))
        bot.config = SimpleNamespace(
            showdown=SimpleNamespace(username="BotUser"),
            database=SimpleNamespace(db_path="dummy.db"),
            gameplay=SimpleNamespace(town_read_comment_chance=1.0, vote_comment_chance=1.0, silent_mode=False),
        )
        bot._current_vote_target = "novote"
        bot._current_town_read = None
        bot._send_chat_message = AsyncMock()
        bot.send_room_command = AsyncMock()

        asyncio.run(bot._evaluate_and_vote())

        bot.send_room_command.assert_not_called()

    def test_evaluate_and_vote_mylo_skipped_when_modexe(self):
        bot = MafiaBot.__new__(MafiaBot)
        session = GameSession(source="test", raw_text="", players=["Alice", "Bob", "Carl", "Dave"])
        bot.tracker = SimpleNamespace(
            state="DAY", eliminated=False, in_game=True, theme="Modexe",
            original_role_tokens=["mafia", "vt", "vt", "vt"], dead_players=set(),
            get_game_session=Mock(return_value=session),
        )
        bot.strategy = Mock()
        bot.strategy.get_vote_decision = Mock(return_value=(None, 0.0))
        bot.strategy.get_town_read = Mock(return_value=("Alice", 0.95))
        bot.config = SimpleNamespace(
            showdown=SimpleNamespace(username="BotUser"),
            database=SimpleNamespace(db_path="dummy.db"),
            gameplay=SimpleNamespace(town_read_comment_chance=1.0, vote_comment_chance=1.0, silent_mode=False),
        )
        bot._current_vote_target = None
        bot._current_town_read = None
        bot._send_chat_message = AsyncMock()
        bot.send_room_command = AsyncMock()

        with patch("mafia_framework.bot.client.asyncio.sleep", new=AsyncMock()):
            asyncio.run(bot._evaluate_and_vote())

        bot.send_room_command.assert_any_call("/mafia vote Alice")

    def test_delayed_vote_reaction_sends_a_catchphrase(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot._send_chat_message = AsyncMock()

        with patch("mafia_framework.bot.client.asyncio.sleep", new=AsyncMock()), \
             patch("random.uniform", return_value=3.0), \
             patch("random.choice", side_effect=lambda pool: pool[0]):
            asyncio.run(bot._delayed_vote_reaction("|c:|123|~|Alice has voted BotUser."))

        bot._send_chat_message.assert_awaited_once_with("why me")

    def test_vote_detection_for_bot_username(self):
        self.assertTrue(MafiaBot._is_vote_for_bot("|c:|123|~|Alice has voted BotUser.", "BotUser"))
        self.assertTrue(MafiaBot._is_vote_for_bot("|c:|123|~|Alice voted for bot-user", "BotUser"))
        self.assertFalse(MafiaBot._is_vote_for_bot("|c:|123|~|Alice has voted Bob.", "BotUser"))

    def test_extract_vote_voter_name(self):
        self.assertEqual(MafiaBot._extract_vote_voter_name("|c:|123|~|Alice has voted BotUser."), "Alice")
        self.assertIsNone(MafiaBot._extract_vote_voter_name("|c:|123|~|Alice is just chatting"))

    def test_extract_me_action_bare_me(self):
        # Sender names show up in ALL CAPS in the raw line for a real /me
        # action (a Showdown rendering quirk) -- the command text itself
        # is still lowercase "/me".
        self.assertEqual(
            MafiaBot._extract_me_action("|c:|123| TRIMMERZ|/me", "BotUser"), ("/me", "")
        )

    def test_extract_me_action_with_text(self):
        self.assertEqual(
            MafiaBot._extract_me_action("|c:|123|Alice|/me dies dramatically", "BotUser"),
            ("/me", "dies dramatically"),
        )

    def test_extract_me_action_case_insensitive_command(self):
        # The exact typed case (caps here) is preserved in the returned
        # prefix, so the mirrored reply matches it instead of always
        # capitalizing regardless of what triggered it.
        self.assertEqual(
            MafiaBot._extract_me_action("|c:|123|Alice|/ME dies dramatically", "BotUser"),
            ("/ME", "dies dramatically"),
        )

    def test_extract_me_action_ignores_own_messages(self):
        self.assertIsNone(MafiaBot._extract_me_action("|c:|123|BotUser|/me dies", "BotUser"))

    def test_extract_me_action_none_for_normal_chat(self):
        self.assertIsNone(MafiaBot._extract_me_action("|c:|123|Alice|hello there", "BotUser"))

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

    def test_handle_pm_ignores_friend_request_system_notification(self):
        # Confirmed live: PS's friend-request UI card arrives via the same
        # |pm| format as a real message, and its HTML contains a literal
        # name="send" attribute -- that used to misfire the "send me a
        # random name" easter egg and PM back a random player's name.
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(in_game=True, eliminated=False, players=["Alice", "Bob"], dead_players=set())
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="BotUser"))
        bot.connection = Mock()
        bot.connection.send = AsyncMock()

        msg = (
            '/uhtml sent-chineseshaq,<button class="button" name="send" '
            'value="/friends accept chineseshaq">Accept</button> | '
            '<button class="button" name="send" value="/friends reject chineseshaq">Deny</button>'
        )
        asyncio.run(bot._handle_pm("ChineseShaq", msg))

        bot.connection.send.assert_not_awaited()

    def test_handle_pm_send_random_name_requires_random_and_name_too(self):
        # Narrowed from a bare "send" substring check -- a real unrelated
        # message like this shouldn't trigger the random-name easter egg.
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(in_game=True, eliminated=False, players=["Alice", "Bob"], dead_players=set())
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="BotUser"))
        bot.connection = Mock()
        bot.connection.send = AsyncMock()

        asyncio.run(bot._handle_pm("Alice", "can someone send the theme list"))

        bot.connection.send.assert_not_awaited()

    def test_get_claim_message_lies_as_vt_for_listed_roles(self):
        bot = MafiaBot.__new__(MafiaBot)
        # Any Mafia-aligned role (not just a specific named one) plus a set
        # of dangerous-to-reveal non-town roles from other alignments.
        for role in [
            "Werewolf", "Alien", "Cult Leader", "Serial Killer", "Goo",
            "Mafia Goon", "Mafia Roleblocker", "Mafia Boss",
            "Solo Condemner", "Solo Traitor Lover Vigilante One-Shot Strongman", "Condemner",
            "Replicant Roleblocker",
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

    def test_finished_event_schedules_game_end_chat_with_ragebait_when_enough_players(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot._random_actions_task = None
        bot.tracker = SimpleNamespace(
            players=["Alice", "BotUser", "C", "D", "E", "F", "G"],
            dead_players={"Bob"},
            reset=Mock(),
        )
        bot.strategy = Mock()
        bot.config = SimpleNamespace(
            gameplay=SimpleNamespace(silent_mode=False),
            showdown=SimpleNamespace(username="BotUser"),
        )
        bot._delayed_game_end_chat = AsyncMock()
        bot._save_game_to_db = AsyncMock()

        with patch("mafia_framework.bot.client.asyncio.create_task") as mock_create_task:
            asyncio.run(bot._handle_tracker_event("FINISHED"))
            mock_create_task.assert_called_once()

        bot._delayed_game_end_chat.assert_called_once()
        ragebait_message = bot._delayed_game_end_chat.call_args[0][0]
        matching_line = next(line for line in RAGEBAIT_LINES if ragebait_message.startswith(line))
        self.assertEqual(ragebait_message, f"{matching_line} Alice C D E F G Bob")

    def test_finished_event_skips_ragebait_when_below_player_threshold(self):
        # 8-player minimum for trash talk -- this game only had 3 (Alice,
        # BotUser, Bob), well below it, even though silent_mode is off.
        bot = MafiaBot.__new__(MafiaBot)
        bot._random_actions_task = None
        bot.tracker = SimpleNamespace(players=["Alice", "BotUser"], dead_players={"Bob"}, reset=Mock())
        bot.strategy = Mock()
        bot.config = SimpleNamespace(
            gameplay=SimpleNamespace(silent_mode=False),
            showdown=SimpleNamespace(username="BotUser"),
        )
        bot._delayed_game_end_chat = AsyncMock()
        bot._save_game_to_db = AsyncMock()

        with patch("mafia_framework.bot.client.asyncio.create_task"):
            asyncio.run(bot._handle_tracker_event("FINISHED"))

        bot._delayed_game_end_chat.assert_called_once_with(None)

    def test_finished_event_skips_ragebait_in_silent_mode(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot._random_actions_task = None
        bot.tracker = SimpleNamespace(
            players=["Alice", "BotUser", "C", "D", "E", "F", "G", "H"], dead_players=set(), reset=Mock()
        )
        bot.strategy = Mock()
        bot.config = SimpleNamespace(
            gameplay=SimpleNamespace(silent_mode=True),
            showdown=SimpleNamespace(username="BotUser"),
        )
        bot._delayed_game_end_chat = AsyncMock()
        bot._save_game_to_db = AsyncMock()

        with patch("mafia_framework.bot.client.asyncio.create_task"):
            asyncio.run(bot._handle_tracker_event("FINISHED"))

        bot._delayed_game_end_chat.assert_called_once_with(None)

    def test_delayed_game_end_chat_waits_then_sends_gg_and_ragebait(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.config = SimpleNamespace(gameplay=SimpleNamespace(autojoin_delay_seconds=5.0))
        bot.send_room_command = AsyncMock()
        bot._send_chat_message = AsyncMock()

        with patch("mafia_framework.bot.client.asyncio.sleep", new=AsyncMock()) as mock_sleep:
            asyncio.run(bot._delayed_game_end_chat("gg easy Alice Bob"))

        mock_sleep.assert_awaited_once_with(5.0)
        bot.send_room_command.assert_awaited_once_with("gg")
        bot._send_chat_message.assert_awaited_once_with("gg easy Alice Bob")

    def test_delayed_game_end_chat_skips_ragebait_when_none(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.config = SimpleNamespace(gameplay=SimpleNamespace(autojoin_delay_seconds=0.0))
        bot.send_room_command = AsyncMock()
        bot._send_chat_message = AsyncMock()

        asyncio.run(bot._delayed_game_end_chat(None))

        bot.send_room_command.assert_awaited_once_with("gg")
        bot._send_chat_message.assert_not_awaited()

    def test_handle_pm_speak_says_the_given_line_in_room(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(in_game=True, eliminated=False, players=["Alice"], dead_players=set())
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="BotUser"))
        bot.connection = Mock()
        bot.connection.send = AsyncMock()
        bot._send_chat_message = AsyncMock()

        asyncio.run(bot._handle_pm("Alice", ".speak hello everyone how's it going"))

        bot._send_chat_message.assert_awaited_once_with("hello everyone how's it going")
        bot.connection.send.assert_awaited_once_with("|/pm Alice, said: hello everyone how's it going")

    def test_handle_pm_speak_does_nothing_without_a_line(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(in_game=True, eliminated=False, players=["Alice"], dead_players=set())
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="BotUser"))
        bot.connection = Mock()
        bot.connection.send = AsyncMock()
        bot._send_chat_message = AsyncMock()

        asyncio.run(bot._handle_pm("Alice", ".speak"))

        bot._send_chat_message.assert_not_awaited()
        bot.connection.send.assert_not_awaited()

    def test_handle_pm_speak_refuses_a_line_starting_with_slash(self):
        # Confirmed live: Showdown treats a room message starting with "/"
        # as a slash command, not chat text -- send_room_command can't
        # tell the difference, so this must be blocked before it ever
        # reaches _send_chat_message, or any PM sender could make the bot
        # execute arbitrary commands (a player got it to send a real /pm
        # this way before this fix).
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(in_game=True, eliminated=False, players=["Alice"], dead_players=set())
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="BotUser"))
        bot.connection = Mock()
        bot.connection.send = AsyncMock()
        bot._send_chat_message = AsyncMock()

        asyncio.run(bot._handle_pm("Alice", ".speak /pm someone, do something"))

        bot._send_chat_message.assert_not_awaited()
        bot.connection.send.assert_awaited_once()
        sent_pm = bot.connection.send.call_args[0][0]
        self.assertIn("can't start a spoken line with '/'", sent_pm)

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
        bot.send_room_command = AsyncMock()

        with patch("mafia_framework.bot.client.asyncio.sleep", new=AsyncMock()):
            asyncio.run(bot._handle_pm("Host", ".reads"))

        bot.send_room_command.assert_awaited_once_with("/mafia votes")
        bot.connection.send.assert_awaited_once_with("|/pm Host, Bob 82% | Alice 20%")

    def test_handle_pm_reads_command_excludes_players_no_longer_in_live_votes(self):
        # A player added mid-game and later removed/left through wording we
        # don't have a regex for can linger in session.players -- cross-
        # checking against the live vote-roster filters them back out.
        bot = MafiaBot.__new__(MafiaBot)
        session = GameSession(source="test", raw_text="", players=["Alice", "Bob", "Departed"])
        bot.tracker = SimpleNamespace(
            in_game=True, eliminated=False, players=["Alice", "Bob", "Departed"], dead_players=set(),
            get_game_session=Mock(return_value=session),
            live_vote_counts={"Alice": 0, "Bob": 0},
        )
        bot.config = SimpleNamespace(
            showdown=SimpleNamespace(username="BotUser"),
            database=SimpleNamespace(db_path="dummy.db"),
        )
        bot.strategy = Mock()
        bot.strategy.get_full_predictions = Mock(
            return_value=[("Bob", 0.82), ("Departed", 0.55), ("Alice", 0.20)]
        )
        bot.connection = Mock()
        bot.connection.send = AsyncMock()
        bot.send_room_command = AsyncMock()

        with patch("mafia_framework.bot.client.asyncio.sleep", new=AsyncMock()):
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

    def test_handle_pm_vote_command_handles_multi_word_player_names(self):
        # Real names can contain spaces (e.g. "I give u pile alt") -- only
        # grabbing the first whitespace-delimited token after ".vote" would
        # try to match just "I" and fail to find a real player.
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(
            in_game=True, eliminated=False, players=["I give u pile alt", "Bob"], dead_players=set()
        )
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="BotUser"))
        bot.connection = Mock()
        bot.connection.room = "mafia"
        bot.connection.send = AsyncMock()
        bot._current_vote_target = None

        asyncio.run(bot._handle_pm("Host", ".vote I give u pile alt"))

        bot.connection.send.assert_any_call("mafia|/mafia vote I give u pile alt")
        bot.connection.send.assert_any_call("|/pm Host, Voted I give u pile alt.")
        self.assertEqual(bot._current_vote_target, "I give u pile alt")

    def test_handle_pm_vote_novote_is_always_a_valid_option(self):
        # "No Vote" is a real room option (its own button in the votes
        # list), not a player -- it must never be rejected as "not a real
        # player" just because it isn't in the roster.
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(in_game=True, eliminated=False, players=["Alice", "Bob"], dead_players=set())
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="BotUser"))
        bot.connection = Mock()
        bot.connection.room = "mafia"
        bot.connection.send = AsyncMock()
        bot._current_vote_target = "Alice"

        asyncio.run(bot._handle_pm("Host", ".vote novote"))

        bot.connection.send.assert_any_call("mafia|/mafia vote novote")
        bot.connection.send.assert_any_call("|/pm Host, Voted No Vote.")
        self.assertIsNone(bot._current_vote_target)

    def test_handle_pm_vote_no_vote_with_space_is_also_valid(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(in_game=True, eliminated=False, players=["Alice", "Bob"], dead_players=set())
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="BotUser"))
        bot.connection = Mock()
        bot.connection.room = "mafia"
        bot.connection.send = AsyncMock()
        bot._current_vote_target = None

        asyncio.run(bot._handle_pm("Host", ".vote No Vote"))

        bot.connection.send.assert_any_call("mafia|/mafia vote novote")
        bot.connection.send.assert_any_call("|/pm Host, Voted No Vote.")

    def test_handle_pm_unvote_command_clears_current_vote(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(in_game=True, eliminated=False, players=["Alice", "Bob"], dead_players=set())
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="BotUser"))
        bot.connection = Mock()
        bot.connection.room = "mafia"
        bot.connection.send = AsyncMock()
        bot._current_vote_target = "Alice"

        asyncio.run(bot._handle_pm("Host", ".unvote"))

        bot.connection.send.assert_any_call("mafia|/mafia unvote")
        bot.connection.send.assert_any_call("|/pm Host, Unvoted.")
        self.assertIsNone(bot._current_vote_target)

    def test_handle_pm_unvote_command_when_not_voting_anyone(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(in_game=True, eliminated=False, players=["Alice", "Bob"], dead_players=set())
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="BotUser"))
        bot.connection = Mock()
        bot.connection.room = "mafia"
        bot.connection.send = AsyncMock()
        bot._current_vote_target = None

        asyncio.run(bot._handle_pm("Host", ".unvote"))

        bot.connection.send.assert_awaited_once_with("|/pm Host, not voting anyone")

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

        # Pin the template choice to one that doesn't happen to contain
        # "with me" itself (e.g. the normal "Vote with me plz" template) --
        # otherwise this assertion would be a flaky false-positive on the
        # real random.choice rather than actually verifying the rally
        # branch (gated on current_vote_target) was never reached.
        with patch("random.random", return_value=0.0), patch("random.choices", return_value=[1]), \
             patch("random.sample", side_effect=lambda items, k: items[:k]), \
             patch("random.choice", side_effect=lambda seq: next(t for t in seq if "give me reads" in t)):
            prompt = bot._build_question_prompt(session)

        self.assertEqual(prompt, "Alice give me reads")

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

    def test_save_game_to_db_discards_when_auto_save_disabled_and_no_stdin(self):
        # Running headless/backgrounded means there's no terminal to answer
        # the save-confirmation prompt -- input() raises EOFError, which
        # must not escape and crash the message-processing loop. With
        # auto_save_games off, this preserves the old discard behavior.
        bot = MafiaBot.__new__(MafiaBot)
        session = GameSession(source="test", raw_text="some raw log text", players=["Alice", "Bob"])
        bot.tracker = SimpleNamespace(get_game_session=Mock(return_value=session))
        bot.config = SimpleNamespace(gameplay=SimpleNamespace(auto_save_games=False))

        with patch("builtins.input", side_effect=EOFError):
            asyncio.run(bot._save_game_to_db())

    def test_save_game_to_db_auto_saves_when_no_stdin(self):
        # The whole point of auto_save_games: an unattended live-test run
        # has no one to answer the prompt, but the game should still get
        # saved rather than silently discarded every time.
        bot = MafiaBot.__new__(MafiaBot)
        session = GameSession(
            source="test", raw_text="some raw log text", players=["Alice", "Bob"], flips=[]
        )
        bot.tracker = SimpleNamespace(get_game_session=Mock(return_value=session))
        bot.config = SimpleNamespace(
            gameplay=SimpleNamespace(auto_save_games=True),
            database=SimpleNamespace(db_path="dummy.db"),
        )

        with patch("builtins.input", side_effect=EOFError), \
             patch("mafia_framework.bot.client.ingest_log", return_value=42) as mock_ingest, \
             patch.object(bot, "_prompt_for_undefined_roles", new=AsyncMock()) as mock_prompt:
            asyncio.run(bot._save_game_to_db())

        mock_ingest.assert_called_once_with(db_path="dummy.db", raw_text="some raw log text", source="live_bot")
        mock_prompt.assert_awaited_once_with("dummy.db", 42)

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
        # A "**Players (N)**:" roster snapshot alone is just a roster
        # update now -- it never fires STARTED by itself (only the
        # explicit "game is starting" announcement does), so repeating it
        # is naturally harmless.
        tracker = GameTracker()
        tracker.state = "SIGNUPS"
        roster_line = "|c:|123|~|**Players (2)**: Alice, BotUser"

        first_event = tracker.process_message(roster_line, bot_username="BotUser")
        self.assertIsNone(first_event)

        second_event = tracker.process_message(roster_line, bot_username="BotUser")
        self.assertIsNone(second_event)
        self.assertEqual(tracker.players, ["Alice", "BotUser"])

        start_event = tracker.process_message(
            "|raw|<div class=\"broadcast-blue\">The game of Mafia is starting!</div>", bot_username="BotUser"
        )
        self.assertEqual(start_event, "STARTED")
        self.assertTrue(tracker.in_game)

        # A repeat of the explicit start announcement (state is no longer
        # SIGNUPS) must not look like a brand new game either.
        repeat_start_event = tracker.process_message(
            "|raw|<div class=\"broadcast-blue\">The game of Mafia is starting!</div>", bot_username="BotUser"
        )
        self.assertIsNone(repeat_start_event)

    def test_duplicate_game_end_message_does_not_refire_finished(self):
        # Seen live: the room can send more than one "game has ended"-shaped
        # message for the same finish (e.g. a win announcement followed by
        # a separate wrap-up message), which used to re-fire FINISHED each
        # time -- causing "gg" and the ragebait line to send multiple times
        # for a single game end.
        tracker = GameTracker()
        tracker.state = "DAY"
        tracker.players = ["Alice", "BotUser"]
        tracker.in_game = True

        first_event = tracker.process_message(
            "|c:|123|~|The game of Mafia has ended.", bot_username="BotUser"
        )
        self.assertEqual(first_event, "FINISHED")

        second_event = tracker.process_message(
            "|c:|124|~|The game of Mafia has ended.", bot_username="BotUser"
        )
        self.assertIsNone(second_event)

    def test_duplicate_day_marker_does_not_refire_day(self):
        # Seen live: DAY_MARKER_RE also matches generic decorative
        # separators (---, ***, etc.) with no requirement to actually say
        # "day"/"hammer" -- an unrelated system message (e.g. a reveal
        # announcement with a divider) accidentally matched this and
        # re-fired an already-active Day 1, cascading into duplicate vote
        # re-evaluations and random-actions tasks.
        tracker = GameTracker()
        tracker.state = "NIGHT"
        tracker.players = ["Alice", "Bob"]
        tracker.in_game = True

        first_event = tracker.process_message(
            "Day 1. The hammer count is set at 2", bot_username="BotUser"
        )
        self.assertEqual(first_event, "DAY")
        self.assertEqual(tracker.current_day, 1)

        second_event = tracker.process_message("***", bot_username="BotUser")
        self.assertIsNone(second_event)
        self.assertEqual(tracker.current_day, 1)

    def test_genuine_new_day_still_fires_after_a_duplicate(self):
        tracker = GameTracker()
        tracker.state = "NIGHT"
        tracker.players = ["Alice", "Bob"]
        tracker.in_game = True

        tracker.process_message("Day 1. The hammer count is set at 2", bot_username="BotUser")
        tracker.state = "NIGHT"  # a real night phase happens in between

        event = tracker.process_message(
            "Day 2. The hammer count is set at 3", bot_username="BotUser"
        )
        self.assertEqual(event, "DAY")
        self.assertEqual(tracker.current_day, 2)

    def test_spectate_broadcast_does_not_override_confirmed_in_game(self):
        # Seen live: this "in progress / become a substitute / spectate"
        # broadcast is sent generically to anyone joining/refreshing the
        # room while a game is active -- including actual participants --
        # and it arrived (in the backlog replay) AFTER the roster line that
        # had already confirmed the bot was playing. It silently flipped
        # in_game back to False, breaking every subsequent vote reaction,
        # claim, and night action for a game the bot was still actually in.
        tracker = GameTracker()
        tracker.state = "SIGNUPS"
        roster_line = "|c:|123|~|**Players (3)**: Alice, Bob, BotUser"
        spectate_line = (
            '|c:|124|~|/uhtml mafia,<div class="broadcast-blue">'
            '<p style="font-weight: bold">A game of Mafia is in progress.</p>'
            '<p><button class="button" name="send" value="/msgroom mafia,/mafia sub in">Become a substitute</button> '
            '<button class="button" name="send" value="/join view-mafia-mafia">Spectate the game</button></p></div>'
        )

        tracker.process_message(roster_line, bot_username="BotUser")
        event = tracker.process_message(
            "|raw|<div class=\"broadcast-blue\">The game of Mafia is starting!</div>", bot_username="BotUser"
        )
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

    def test_roster_alone_does_not_start_the_game(self):
        # Reported live: the bot was voting/announcing town reads before
        # "The game of Mafia is starting!" ever appeared, because a
        # "Players (N):" roster snapshot alone used to be treated as the
        # game start. Only the explicit announcement may do that.
        tracker = GameTracker()
        tracker.state = "SIGNUPS"

        event = tracker.process_message(
            "|c:|123|~|**Players (3)**: Alice, Bob, BotUser", bot_username="BotUser"
        )

        self.assertIsNone(event)
        self.assertEqual(tracker.state, "SIGNUPS")
        self.assertFalse(tracker.in_game)
        self.assertEqual(tracker.players, ["Alice", "Bob", "BotUser"])

    def test_get_game_session_reflects_a_sub_not_the_stale_original_roster(self):
        # Reported live: after "A Flowers Dream has been subbed out.
        # Werewolf has joined the game." (the real combined broadcast text
        # for a sub), tracker.players correctly updated, but session.players
        # -- what predictions/reads actually use -- kept the original,
        # departed player's name. That's because get_game_session re-parses
        # the accumulated raw text, which still contains the original
        # "Players (N): ..." roster line, and used to only fall back to
        # tracker.players when that re-parse came up empty (which it never
        # did, since the stale roster line is always still in there).
        tracker = GameTracker()
        tracker.bot_username = "BotUser"
        tracker.state = "SIGNUPS"
        tracker.process_message(
            "|c:|1|~|**Players (3)**: A Flowers Dream, Aziziller, BotUser", bot_username="BotUser"
        )
        tracker.process_message(
            "|c:|2|~|The game of Mafia is starting!", bot_username="BotUser"
        )

        tracker.process_message(
            '|raw|<div class="broadcast-blue">A Flowers Dream has been subbed out. '
            "Werewolf has joined the game.</div>",
            bot_username="BotUser",
        )

        self.assertEqual(tracker.players, ["Werewolf", "Aziziller", "BotUser"])
        session = tracker.get_game_session()
        self.assertEqual(session.players, ["Werewolf", "Aziziller", "BotUser"])

    def test_roster_lock_does_not_immediately_count_as_day(self):
        # Seen live: roster lock is followed by a rolling/role-distribution
        # period ("Night 0") before Day 1 actually starts. Treating the
        # roster announcement as already-DAY made the bot start chatting
        # (filler/reactions/questions, all gated on state == "DAY") before
        # the game itself had really started.
        tracker = GameTracker()
        tracker.state = "SIGNUPS"

        tracker.process_message(
            "|c:|123|~|**Players (3)**: Alice, Bob, BotUser", bot_username="BotUser"
        )
        event = tracker.process_message(
            "|raw|<div class=\"broadcast-blue\">The game of Mafia is starting!</div>", bot_username="BotUser"
        )
        self.assertEqual(event, "STARTED")
        self.assertNotEqual(tracker.state, "DAY")

        day_event = tracker.process_message(
            "Day 1. The hammer count is set at 2", bot_username="BotUser"
        )
        self.assertEqual(day_event, "DAY")
        self.assertEqual(tracker.state, "DAY")

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

    def test_parse_theme_if_present_extracts_theme_name(self):
        tracker = GameTracker()
        line = (
            '|pagehtml|<div class="pad broadcast-blue">'
            '<p style="font-weight:bold;">Players (4): Alice, Bob</p><hr/>'
            '<p><span style="font-weight:bold;">Theme</span>: Modexe</p>'
            '<p>Some theme description here</p></div>'
        )

        found = tracker.parse_theme_if_present(line)

        self.assertTrue(found)
        self.assertEqual(tracker.theme, "Modexe")

    def test_parse_theme_if_present_false_when_absent(self):
        tracker = GameTracker()
        self.assertFalse(tracker.parse_theme_if_present("|c:|123|~|Day 1. The hammer count is set at 2"))
        self.assertIsNone(tracker.theme)

    def test_parse_host_if_present_extracts_host_name(self):
        tracker = GameTracker()
        line = (
            '|pagehtml|<div class="pad broadcast-blue">'
            '<h1 style="text-align:center;">Mafia</h1><h3>Host: ghostlyplanets</h3>'
            '<p style="font-weight:bold;">Players (4): Alice, Bob</p></div>'
        )

        found = tracker.parse_host_if_present(line)

        self.assertTrue(found)
        self.assertEqual(tracker.host, "ghostlyplanets")

    def test_parse_host_if_present_false_when_absent(self):
        tracker = GameTracker()
        self.assertFalse(tracker.parse_host_if_present("|c:|123|~|Day 1. The hammer count is set at 2"))
        self.assertIsNone(tracker.host)

    def test_parse_partners_if_present_extracts_single_partner(self):
        tracker = GameTracker()
        line = '<p><span style="font-weight:bold">Partners</span>: Trimmerz</p>'

        found = tracker.parse_partners_if_present(line)

        self.assertTrue(found)
        self.assertEqual(tracker.partners, ["Trimmerz"])

    def test_parse_partners_if_present_extracts_multiple_partners(self):
        tracker = GameTracker()
        line = '<p><span style="font-weight:bold">Partners</span>: Trimmerz, Alice</p>'

        tracker.parse_partners_if_present(line)

        self.assertEqual(tracker.partners, ["Trimmerz", "Alice"])

    def test_parse_partners_if_present_false_when_absent(self):
        tracker = GameTracker()
        self.assertFalse(tracker.parse_partners_if_present("|c:|123|~|Day 1. The hammer count is set at 2"))
        self.assertEqual(tracker.partners, [])

    def test_parse_idea_options_if_present_extracts_options_and_alignment(self):
        # Trimmed real capture from a live "/mafia votes" panel during an
        # IDEA module -- "clear" is disabled (no pick made yet), and the two
        # real role options each have their alignment spelled out below.
        line = (
            '|pagehtml|<div class="pad broadcast-blue"><h3>Host: Overkill_Tuna</h3>'
            '<p><b>IDEA information:</b><br /><b>role:</b> '
            '<button class="button disabled" style="color:#575757;">clear</button>'
            '<button class="button" name="send" value="/msgroom mafia,/mafia ideapick role, day2suicidalbulletproofpurplegoo">Day 2 Suicidal Bulletproof Purple Goo</button>'
            '<button class="button" name="send" value="/msgroom mafia,/mafia ideapick role, mafiaoneshotstrongman">Mafia One-Shot Strongman</button><br /></p>'
            '<p><details><summary class="button"><b>Role details:</b></summary>'
            '<p><details><summary>Day 2 Suicidal Bulletproof Purple Goo</summary><table><tr><td><ul>'
            '<li>You are aligned with the <span style="color:#060;">Town</span>. You win...</li></ul></td></tr></table></details>'
            '<details><summary>Mafia One-Shot Strongman</summary><table><tr><td><ul>'
            '<li>You are aligned with the <span style="color:#F00;">Mafia</span>. You win...</li></ul></td></tr></table></details>'
            '</p></details></p></div>'
        )

        options = GameTracker().parse_idea_options_if_present(line)

        self.assertEqual(
            options,
            [
                ("day2suicidalbulletproofpurplegoo", "Day 2 Suicidal Bulletproof Purple Goo", "Town"),
                ("mafiaoneshotstrongman", "Mafia One-Shot Strongman", "Mafia"),
            ],
        )

    def test_parse_idea_options_if_present_empty_when_absent(self):
        tracker = GameTracker()
        self.assertEqual(tracker.parse_idea_options_if_present("|c:|123|~|Day 1 has begun."), [])

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

    def test_night_action_target_uses_random_non_self_player_for_mafia_role(self):
        # Determined from the live-known role (learned via /mafia role),
        # not flips -- those only exist after the bot has already died.
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(dead_players=set())
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="BotUser"))
        bot._own_role = "Mafia Goon"
        session = GameSession(source="test", raw_text="", players=["Alice", "Bob", "BotUser"])

        with patch("random.choice", return_value="Bob"):
            self.assertEqual(bot._choose_night_action_target(session), "Bob")

    def test_night_action_target_excludes_mafia_partners(self):
        # Confirmed live bug: a random kill could target the bot's own
        # Mafia partner since only self was excluded from the pool.
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(dead_players=set(), partners=["Bob"])
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="BotUser"))
        bot._own_role = "Mafia Goon"
        session = GameSession(source="test", raw_text="", players=["Alice", "Bob", "BotUser"])

        with patch("random.choice", side_effect=lambda pool: pool[0]):
            result = bot._choose_night_action_target(session)

        self.assertEqual(result, "Alice")

    def test_night_action_target_none_for_town_role(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(dead_players=set())
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="BotUser"))
        bot._own_role = "Vanilla Townie"
        session = GameSession(source="test", raw_text="", players=["Alice", "Bob", "BotUser"])

        self.assertIsNone(bot._choose_night_action_target(session))

    def test_night_action_target_none_when_role_unknown(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot.tracker = SimpleNamespace(dead_players=set())
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="BotUser"))
        bot._own_role = None
        session = GameSession(source="test", raw_text="", players=["Alice", "Bob", "BotUser"])

        self.assertIsNone(bot._choose_night_action_target(session))

    def test_choose_role_action_doctor_picks_from_town_reads(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot._own_role = "Doctor"
        bot._current_vote_target = None
        bot.tracker = SimpleNamespace(dead_players=set())
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="BotUser"), database=SimpleNamespace(db_path="dummy.db"))
        bot.strategy = Mock()
        bot.strategy.min_confidence = 0.55
        bot.strategy.get_full_predictions = Mock(return_value=[("Alice", 0.05), ("Bob", 0.5)])
        session = GameSession(source="test", raw_text="", players=["Alice", "Bob", "BotUser"])

        with patch("random.choice", side_effect=lambda pool: pool[0]):
            result = bot._choose_role_action(session)

        self.assertEqual(result, ("Doc", "Alice"))

    def test_choose_role_action_doctor_falls_back_to_full_pool(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot._own_role = "Doctor"
        bot._current_vote_target = None
        bot.tracker = SimpleNamespace(dead_players=set())
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="BotUser"), database=SimpleNamespace(db_path="dummy.db"))
        bot.strategy = Mock()
        bot.strategy.min_confidence = 0.55
        bot.strategy.get_full_predictions = Mock(return_value=[("Alice", 0.5), ("Bob", 0.5)])
        session = GameSession(source="test", raw_text="", players=["Alice", "Bob", "BotUser"])

        with patch("random.choice", side_effect=lambda pool: pool[0]):
            result = bot._choose_role_action(session)

        self.assertEqual(result, ("Doc", "Alice"))

    def test_choose_role_action_cop_picks_from_scum_reads(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot._own_role = "Cop"
        bot._current_vote_target = None
        bot.tracker = SimpleNamespace(dead_players=set())
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="BotUser"), database=SimpleNamespace(db_path="dummy.db"))
        bot.strategy = Mock()
        bot.strategy.min_confidence = 0.55
        bot.strategy.get_full_predictions = Mock(return_value=[("Alice", 0.9), ("Bob", 0.5)])
        session = GameSession(source="test", raw_text="", players=["Alice", "Bob", "BotUser"])

        with patch("random.choice", side_effect=lambda pool: pool[0]):
            result = bot._choose_role_action(session)

        self.assertEqual(result, ("Cop", "Alice"))

    def test_choose_role_action_cop_does_not_match_similarly_named_role(self):
        # "Cop-Of-All-Trades" is a mechanically different real role from
        # this ruleset -- an exact match avoids misfiring on it.
        bot = MafiaBot.__new__(MafiaBot)
        bot._own_role = "Cop-Of-All-Trades"
        bot._current_vote_target = None
        bot.tracker = SimpleNamespace(dead_players=set())
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="BotUser"), database=SimpleNamespace(db_path="dummy.db"))
        session = GameSession(source="test", raw_text="", players=["Alice", "Bob", "BotUser"])

        self.assertIsNone(bot._choose_role_action(session))

    def test_choose_role_action_pretty_lady_uses_days_vote_target_if_alive(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot._own_role = "Pretty Lady"
        bot._current_vote_target = "Bob"
        bot.tracker = SimpleNamespace(dead_players=set())
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="BotUser"), database=SimpleNamespace(db_path="dummy.db"))
        session = GameSession(source="test", raw_text="", players=["Alice", "Bob", "BotUser"])

        result = bot._choose_role_action(session)

        self.assertEqual(result, ("Pretty Lady", "Bob"))

    def test_choose_role_action_pretty_lady_falls_back_to_scum_reads_if_vote_target_died(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot._own_role = "Pretty Lady"
        bot._current_vote_target = "Bob"
        bot.tracker = SimpleNamespace(dead_players={"Bob"})
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="BotUser"), database=SimpleNamespace(db_path="dummy.db"))
        bot.strategy = Mock()
        bot.strategy.min_confidence = 0.55
        bot.strategy.get_full_predictions = Mock(return_value=[("Alice", 0.9), ("Carl", 0.2)])
        session = GameSession(source="test", raw_text="", players=["Alice", "Carl", "BotUser"])

        with patch("random.choice", side_effect=lambda pool: pool[0]):
            result = bot._choose_role_action(session)

        self.assertEqual(result, ("Pretty Lady", "Alice"))

    def test_choose_role_action_jailkeeper_falls_back_to_fully_random(self):
        # Unlike Pretty Lady, Jailkeeper's fallback is the full alive pool,
        # not specifically the scum list.
        bot = MafiaBot.__new__(MafiaBot)
        bot._own_role = "Jailkeeper"
        bot._current_vote_target = None
        bot.tracker = SimpleNamespace(dead_players=set())
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="BotUser"), database=SimpleNamespace(db_path="dummy.db"))
        session = GameSession(source="test", raw_text="", players=["Alice", "Bob", "BotUser"])

        with patch("random.choice", side_effect=lambda pool: pool[0]):
            result = bot._choose_role_action(session)

        self.assertEqual(result, ("Jailkeeper", "Alice"))

    def test_choose_role_action_vigilante_picks_highest_scumread(self):
        # Unlike Cop, Vigilante always goes for the single top-suspicion
        # read (argmax), not a random pick among everyone above the
        # confidence bar.
        bot = MafiaBot.__new__(MafiaBot)
        bot._own_role = "Vigilante"
        bot._current_vote_target = None
        bot.tracker = SimpleNamespace(dead_players=set())
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="BotUser"), database=SimpleNamespace(db_path="dummy.db"))
        bot.strategy = Mock()
        bot.strategy.min_confidence = 0.55
        bot.strategy.get_full_predictions = Mock(return_value=[("Alice", 0.6), ("Bob", 0.9), ("Carol", 0.4)])
        session = GameSession(source="test", raw_text="", players=["Alice", "Bob", "Carol", "BotUser"])

        result = bot._choose_role_action(session)

        self.assertEqual(result, ("Vigilante", "Bob"))

    def test_choose_role_action_vigilante_falls_back_to_random_when_no_predictions(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot._own_role = "Vigilante"
        bot._current_vote_target = None
        bot.tracker = SimpleNamespace(dead_players=set())
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="BotUser"), database=SimpleNamespace(db_path="dummy.db"))
        bot.strategy = Mock()
        bot.strategy.min_confidence = 0.55
        bot.strategy.get_full_predictions = Mock(return_value=[])
        session = GameSession(source="test", raw_text="", players=["Alice", "Bob", "BotUser"])

        with patch("random.choice", side_effect=lambda pool: pool[0]):
            result = bot._choose_role_action(session)

        self.assertEqual(result, ("Vigilante", "Alice"))

    def test_choose_role_action_none_for_mafia_aligned_vigilante_hybrid(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot._own_role = "Mafia Vigilante"
        session = GameSession(source="test", raw_text="", players=["Alice", "Bob", "BotUser"])

        self.assertIsNone(bot._choose_role_action(session))

    def test_choose_role_action_none_for_mafia_aligned_hybrid_role(self):
        # A hybrid like "Mafia Doctor" should defer to the Mafia kill logic
        # instead of the plain Doctor behavior.
        bot = MafiaBot.__new__(MafiaBot)
        bot._own_role = "Mafia Doctor"
        session = GameSession(source="test", raw_text="", players=["Alice", "Bob", "BotUser"])

        self.assertIsNone(bot._choose_role_action(session))

    def test_choose_role_action_none_for_plain_town_role(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot._own_role = "Vanilla Townie"
        session = GameSession(source="test", raw_text="", players=["Alice", "Bob", "BotUser"])

        self.assertIsNone(bot._choose_role_action(session))

    def test_night_event_sends_role_action_before_falling_back_to_kill(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot._random_actions_task = None
        bot._own_role = "Doctor"
        bot._current_vote_target = None
        bot.config = SimpleNamespace(
            showdown=SimpleNamespace(username="BotUser"),
            gameplay=SimpleNamespace(night_idle=True),
            database=SimpleNamespace(db_path="dummy.db"),
        )
        bot.strategy = Mock()
        bot.strategy.min_confidence = 0.55
        bot.strategy.get_full_predictions = Mock(return_value=[("Alice", 0.05), ("Bob", 0.5)])
        session = GameSession(source="test", raw_text="", players=["Alice", "Bob", "BotUser"])
        bot.tracker = SimpleNamespace(in_game=True, dead_players=set(), get_game_session=Mock(return_value=session))
        bot.send_room_command = AsyncMock()

        with patch("random.choice", side_effect=lambda pool: pool[0]):
            asyncio.run(bot._handle_tracker_event("NIGHT"))

        bot.send_room_command.assert_awaited_once_with("/mafia action Doc Alice")

    def test_night_event_sends_kill_action_for_mafia_role(self):
        bot = MafiaBot.__new__(MafiaBot)
        bot._random_actions_task = None
        bot._own_role = "Mafia Goon"
        bot.config = SimpleNamespace(showdown=SimpleNamespace(username="BotUser"), gameplay=SimpleNamespace(night_idle=True))
        session = GameSession(source="test", raw_text="", players=["Alice", "Bob", "BotUser"])
        bot.tracker = SimpleNamespace(in_game=True, dead_players=set(), get_game_session=Mock(return_value=session))
        bot.send_room_command = AsyncMock()

        with patch("random.choice", return_value="Bob"):
            asyncio.run(bot._handle_tracker_event("NIGHT"))

        bot.send_room_command.assert_awaited_once_with("/mafia action kill Bob")

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

    def test_prompt_for_undefined_roles_handles_no_stdin_gracefully(self):
        # The game itself is already saved by the time this runs -- an
        # EOFError here (no interactive stdin) must not look like the
        # whole save failed, just leave remaining players unassigned.
        bot = MafiaBot.__new__(MafiaBot)
        rows = [
            UndefinedPlayerRow(game_id=1, display_name=None, player_name="Alice", has_messages=True, is_inferred_town_candidate=False),
            UndefinedPlayerRow(game_id=1, display_name=None, player_name="Bob", has_messages=True, is_inferred_town_candidate=False),
        ]

        with (
            patch("mafia_framework.services.game_service.find_undefined_players", return_value=rows),
            patch("mafia_framework.services.game_service.assign_player_role") as mock_assign,
            patch("builtins.input", side_effect=EOFError),
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
