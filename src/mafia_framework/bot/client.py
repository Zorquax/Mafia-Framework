import asyncio
import logging
import random
import re
from pathlib import Path
from typing import Optional

from .config import BotConfig
from .connection import ShowdownConnection
from .tracker import GameTracker
from .strategy import BotStrategy
from ..io.ingestion import ingest_log

logger = logging.getLogger("mafia_bot.client")

VOTE_TARGET_RE = re.compile(r"(?P<voter>.+?)\s+(?:has\s+)?voted(?:\s+for)?\s+(?P<target>.+?)(?:\.|$)", re.IGNORECASE)
VOTE_SOURCE_RE = re.compile(r"(?P<voter>.+?)\s+(?:has\s+)?voted(?:\s+for)?\s+(?P<target>.+?)(?:\.|$)", re.IGNORECASE)


class MafiaBot:
    def __init__(self, config_path: str):
        self.config = BotConfig.load_from_file(config_path)
        self.connection = ShowdownConnection(
            server_url=self.config.showdown.server_url,
            login_url=self.config.showdown.login_url,
            username=self.config.showdown.username,
            password=self.config.showdown.password,
            room=self.config.showdown.room,
        )
        self.tracker = GameTracker()
        self.strategy = BotStrategy(
            model_path=self.config.database.model_path,
            model_d1_path=self.config.database.model_d1_path,
            min_confidence=self.config.gameplay.min_confidence_to_vote,
        )
        
        self._current_vote_target: Optional[str] = None
        self._main_task: Optional[asyncio.Task] = None
        self._periodic_update_task: Optional[asyncio.Task] = None
        self._random_actions_task: Optional[asyncio.Task] = None
        self._ready_for_live_games: bool = False
        self._remembered_lines: list[str] = []

    @staticmethod
    def _message_text_from_line(line: str) -> str:
        parts = line.split("|")
        if len(parts) > 3 and parts[1] in {"c:", "c"}:
            return parts[-1]
        return line

    @staticmethod
    def _extract_vote_voter(line: str) -> Optional[str]:
        from ..io.player_names import canonical_player_name

        message_text = MafiaBot._message_text_from_line(line)
        match = VOTE_SOURCE_RE.search(message_text)
        if not match:
            return None
        voter = canonical_player_name(match.group("voter"))
        return voter or None

    @staticmethod
    def _is_vote_for_bot(line: str, bot_username: str) -> bool:
        from ..io.player_names import canonical_player_name, names_match

        message_text = MafiaBot._message_text_from_line(line)
        match = VOTE_TARGET_RE.search(message_text)
        if not match:
            return False

        target = canonical_player_name(match.group("target"))
        return bool(target) and names_match(target, bot_username)

    def _get_strategy_vote(self, session) -> tuple[Optional[str], float]:
        return self.strategy.get_vote_decision(
            session,
            bot_username=self.config.showdown.username,
            db_path=self.config.database.db_path,
        )

    def _should_do_random_vote(self, session) -> bool:
        target, _ = self._get_strategy_vote(session)
        return target is None

    def _get_bot_alignment(self, session) -> Optional[str]:
        from ..io.player_names import canonical_player_name

        bot_clean = canonical_player_name(self.config.showdown.username)
        for flip in session.flips:
            if canonical_player_name(flip.player_name) == bot_clean:
                return flip.alignment.lower().strip() or None
        return None

    def _get_random_live_player(self) -> Optional[str]:
        from ..io.player_names import canonical_player_name

        bot_clean = canonical_player_name(self.config.showdown.username)
        live_players = [
            player for player in getattr(self.tracker, "players", [])
            if canonical_player_name(player) != bot_clean and player not in self.tracker.dead_players
        ]
        if not live_players:
            return None
        return random.choice(live_players)

    def _get_claim_message(self, session) -> Optional[str]:
        from ..io.player_names import canonical_player_name

        role_texts = []
        for event in getattr(session, "events", []):
            if getattr(event, "event_type", "") == "reveal":
                role_texts.append(getattr(event, "text", ""))

        if not role_texts:
            return None

        bot_clean = canonical_player_name(self.config.showdown.username)
        for text in role_texts:
            if bot_clean and bot_clean.lower() in text.lower():
                role = text.split("role was", 1)[-1].strip()
                if role.lower().startswith("mafia goon"):
                    return "VT 1 to hammer"
                return f"{role} 1 to hammer"

        return None

    def _choose_night_action_target(self, session) -> Optional[str]:
        from ..io.player_names import canonical_player_name

        alignment = self._get_bot_alignment(session)
        if not alignment or alignment in {"town", "unknown"}:
            return None

        alive_players = [p for p in session.players if p not in self.tracker.dead_players]
        bot_clean = canonical_player_name(self.config.showdown.username)
        valid_targets = [p for p in alive_players if canonical_player_name(p) != bot_clean]
        if not valid_targets:
            return None

        return random.choice(valid_targets)

    async def send_room_command(self, command: str):
        msg = f"{self.connection.room}|{command}"
        print(f"\n>>> EXECUTING COMMAND: {msg}\n")
        await self.connection.send(msg)

    async def start(self):
        logger.info("Starting Mafia Bot orchestrator...")
        # Ensure database and models directories exist
        Path(self.config.database.db_path).parent.mkdir(parents=True, exist_ok=True)

        # Start connection in background
        self._main_task = asyncio.create_task(self.connection.connect())

        # Wait for either to finish (or error out)
        while not self.connection._running:
            await asyncio.sleep(0.1)
            
        asyncio.create_task(self._enable_live_mode_after_delay(3.0))

        # Start processing message loop
        await self._message_processing_loop()

    async def _enable_live_mode_after_delay(self, delay: float):
        await asyncio.sleep(delay)
        self._ready_for_live_games = True
        logger.info("Live mode enabled. Backlog processed.")
        
        # Act on the current state inferred from the backlog
        if self.tracker.state == "SIGNUPS":
            if self.config.gameplay.autojoin:
                logger.info("Backlog indicates signups are open. Autojoining...")
                await self.send_room_command("/mafia join")
        elif self.tracker.state == "DAY":
            logger.info("Backlog indicates game is in progress (DAY phase).")
            if not self._periodic_update_task or self._periodic_update_task.done():
                self._periodic_update_task = asyncio.create_task(self._periodic_suspicion_updates())
            if not self._random_actions_task or self._random_actions_task.done():
                self._random_actions_task = asyncio.create_task(self._random_actions_loop())

    async def stop(self):
        logger.info("Stopping Mafia Bot...")
        if self._periodic_update_task:
            self._periodic_update_task.cancel()
        if self._random_actions_task:
            self._random_actions_task.cancel()
        await self.connection.disconnect()
        if self._main_task:
            await self._main_task

    async def _message_processing_loop(self):
        while self.connection._running:
            try:
                room, line = await self.connection.receive_queue.get()
                
                # Check for private messages (PMs) for administrative/manual decisions
                # Format: |pm| sender | receiver | message
                parts = line.split("|")
                if len(parts) > 4 and parts[1] == "pm":
                    sender = parts[2].strip()
                    msg = parts[4].strip()
                    await self._handle_pm(sender, msg)
                
                # Only process messages from the target room
                if room.lower() == self.connection.room.lower():
                    event = self.tracker.process_message(line, bot_username=self.config.showdown.username)
                    self._maybe_remember_chat_line(line)
                    if event == "SIGNUPS" and self.config.gameplay.autojoin:
                        logger.info("Autojoining Mafia game from live room update...")
                        await self.send_room_command("/mafia join")

                    if self.tracker.state == "DAY" and self.tracker.in_game and not self.tracker.eliminated and self._is_vote_for_bot(line, self.config.showdown.username):
                        delay = random.uniform(2.0, 5.0)
                        logger.info(f"Delayed vote reaction by {delay:.2f}s")
                        await asyncio.sleep(delay)
                        voter = self._extract_vote_voter(line)
                        if voter:
                            logger.info(f"Voting back on {voter} after being voted")
                            await self.send_room_command(f"/mafia vote {voter}")
                        catchphrases = ["why me", "im town", "get off", "bruh", "they're gonna qh"]
                        await self.send_room_command(random.choice(catchphrases))

                    if event:
                        if self._ready_for_live_games:
                            await self._handle_tracker_event(event)
                        elif event == "FINISHED":
                            # Reset tracker if a game ended in the backlog so we don't save it
                            self.tracker.reset()

                self.connection.receive_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in message processing loop: {e}", exc_info=True)
                await asyncio.sleep(1)

    async def _handle_tracker_event(self, event: str):
        logger.info(f"Received tracker event: {event}")
        
        if event == "SIGNUPS":
            if self.config.gameplay.autojoin:
                logger.info("Autojoining Mafia game...")
                await self.send_room_command("/mafia join")
                
        elif event == "STARTED":
            self.strategy.reset()
            self._current_vote_target = None
            # Spawn the periodic suspicion update task for the day phase
            if not self._periodic_update_task or self._periodic_update_task.done():
                self._periodic_update_task = asyncio.create_task(self._periodic_suspicion_updates())
            if not self._random_actions_task or self._random_actions_task.done():
                self._random_actions_task = asyncio.create_task(self._random_actions_loop())
                
        elif event == "DAY":
            logger.info("New day started! Re-evaluating votes...")
            if not self._random_actions_task or self._random_actions_task.done():
                self._random_actions_task = asyncio.create_task(self._random_actions_loop())
            await self._evaluate_and_vote()
            
        elif event == "NIGHT":
            # Cancel periodic updates during night
            if self._periodic_update_task:
                self._periodic_update_task.cancel()
            if self._random_actions_task:
                self._random_actions_task.cancel()

            session = self.tracker.get_game_session()
            if self.tracker.in_game:
                target = self._choose_night_action_target(session)
                if target:
                    logger.info(f"Night action: targeting {target} with random non-self action.")
                    await self.send_room_command(f"/mafia action {target}")
                elif self.config.gameplay.night_idle:
                    logger.info("Night started. Sending night idle action.")
                    await self.send_room_command("/mafia idle")
                
        elif event == "FINISHED":
            # Cancel periodic updates
            if self._periodic_update_task:
                self._periodic_update_task.cancel()
            if self._random_actions_task:
                self._random_actions_task.cancel()

            logger.info("Game finished. Sending gg message.")
            await self.send_room_command("gg")

            # Log completed game to database
            await self._save_game_to_db()

            # Reset tracker/strategy
            self.tracker.reset()
            self.strategy.reset()
            self._current_vote_target = None

    async def _periodic_suspicion_updates(self):
        frequency = self.config.gameplay.update_suspicion_frequency_seconds
        while self.tracker.state == "DAY" and self.tracker.in_game and not self.tracker.eliminated:
            try:
                await asyncio.sleep(frequency)
                if self.tracker.state == "DAY" and not self.tracker.eliminated:
                    logger.info("Running periodic day suspicion re-evaluation...")
                    await self._evaluate_and_vote()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic suspicion updates: {e}")

    def _build_question_prompt(self, session) -> Optional[str]:
        from ..io.player_names import canonical_player_name

        alive_players = [p for p in session.players if p not in self.tracker.dead_players]
        bot_clean = canonical_player_name(self.config.showdown.username)
        valid_targets = [p for p in alive_players if canonical_player_name(p) != bot_clean]
        if len(valid_targets) < 1:
            return None

        prompt_groups = {
            1: [
                "{player1} Vote with me plz",
                "{player1} why are u acting so scummy lol",
                "{player1} are you town?",
                "{player1} give me reads",
                "{player1} im voting u in volo btw",
                "{player1} pls read",
                "{player1} get off",
            ],
            2: [
                "{player1} what do you think about {player2}?",
                "{player1} and {player2} what are your reads on each other?",
                "{player1} if {player2} is scum who do you think their partner is?",
                "{player1} what do you think of {player2}'s vote",
            ],
            3: [
                "{player1} {player2} and {player3} are the scumteam btw",
                "{player1} do you think {player2} and {player3} are paired?",
                "{player1} and {player2} what do u think of {player3}",
            ],
        }

        count = random.choices([1, 2, 3], weights=[0.55, 0.3, 0.15], k=1)[0]
        if count > len(valid_targets):
            count = len(valid_targets)

        selected_players = random.sample(valid_targets, k=count)
        templates = prompt_groups[count]
        template = random.choice(templates)
        return template.format(player1=selected_players[0], player2=selected_players[1] if count > 1 else "", player3=selected_players[2] if count > 2 else "").strip()

    def _maybe_remember_chat_line(self, line: str) -> None:
        parts = line.split("|")
        if len(parts) < 4:
            return
        if parts[1] not in {"c:", "c"}:
            return

        message_text = parts[-1].strip()
        if not message_text:
            return
        if message_text.lower().startswith("/mafia"):
            return
        if message_text.lower().startswith("!" ):
            return
        if len(message_text) > 120:
            return

        if self.config.showdown.username and self.config.showdown.username.lower() in message_text.lower():
            return

        self._remembered_lines.append(message_text)
        if len(self._remembered_lines) > 12:
            self._remembered_lines = self._remembered_lines[-12:]

    async def _random_actions_loop(self):
        filler_words = ["bruh", "hm", "oh", "welp", "bleh", "uhh", "thinking", "...", "interesting", "lol", "i mean"]
        from ..io.player_names import canonical_player_name

        while self.tracker.state == "DAY" and self.tracker.in_game and not self.tracker.eliminated:
            try:
                await asyncio.sleep(random.uniform(45.0, 90.0))
                if self.tracker.state != "DAY" or not self.tracker.in_game or self.tracker.eliminated:
                    break

                if not self.tracker.in_game:
                    logger.info("No longer participating in the game; stopping random actions loop.")
                    break

                word = random.choice(filler_words)
                logger.info(f"Saying random filler word: {word}")
                await self.send_room_command(word)

                if self._remembered_lines:
                    remembered_line = random.choice(self._remembered_lines)
                    logger.info(f"Repeating remembered line: {remembered_line}")
                    await self.send_room_command(remembered_line)

                session = self.tracker.get_game_session()
                if not self.tracker.in_game:
                    logger.info("Bot is no longer in the game; stopping random actions loop.")
                    break

                question_prompt = self._build_question_prompt(session)
                if question_prompt:
                    logger.info(f"Asking random question: {question_prompt}")
                    await self.send_room_command(question_prompt)
                if not self.tracker.in_game:
                    logger.info("Bot is no longer in the game; stopping random actions loop.")
                    break

                if session.players and self._should_do_random_vote(session):
                    alive_players = [p for p in session.players if p not in self.tracker.dead_players]
                    bot_clean = canonical_player_name(self.config.showdown.username)
                    valid_targets = [p for p in alive_players if canonical_player_name(p) != bot_clean]

                    if valid_targets:
                        target = random.choice(valid_targets)
                        logger.info(f"Doing random vote on {target}")
                        await self.send_room_command(f"/mafia vote {target}")

                        await asyncio.sleep(random.uniform(2.0, 5.0))

                        if self.tracker.state == "DAY":
                            logger.info(f"Unvoting random target {target}")
                            await self.send_room_command("/mafia unvote")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in random actions loop: {e}")

    async def _evaluate_and_vote(self):
        if self.tracker.state != "DAY" or self.tracker.eliminated:
            logger.info("Current tracker state is not DAY or bot is eliminated; skipping vote evaluation.")
            return

        session = self.tracker.get_game_session()
        if not session.players:
            logger.info("No active player roster found in session. Skipping vote evaluation.")
            return

        if not self.tracker.in_game:
            logger.info("Bot is not participating in the current game. Skipping vote evaluation.")
            return

        target, confidence = self._get_strategy_vote(session)

        if target:
            if target != self._current_vote_target:
                logger.info(f"Decided to vote for {target} (confidence: {confidence:.2%}). Previous: {self._current_vote_target}")
                await self.send_room_command(f"I think {target} is scum")
                await self.send_room_command(f"/mafia vote {target}")
                self._current_vote_target = target
            else:
                logger.info(f"Maintaining current vote on {target} (confidence: {confidence:.2%})")
        else:
            if self._current_vote_target:
                logger.info("No clear target meets confidence threshold. Unvoting.")
                await self.send_room_command("/mafia unvote")
                self._current_vote_target = None

    async def _save_game_to_db(self):
        session = self.tracker.get_game_session()
        raw_text = session.raw_text
        if not raw_text.strip():
            logger.warning("Empty game log, skipping DB persistence.")
            return

        print("\n" + "="*60)
        print("GAME FINISHED! Tentative game data:")
        print(f"Players: {', '.join(session.players) if session.players else 'None'}")
        print("Revealed Roles (Flips):")
        if session.flips:
            for flip in session.flips:
                print(f"  {flip.player_name}: {flip.alignment}")
        else:
            print("  None")
        print("="*60 + "\n")
        
        loop = asyncio.get_running_loop()
        choice = await loop.run_in_executor(None, input, "Store this game to database? [y/N]: ")
        if choice.strip().lower() != 'y':
            logger.info("User chose not to save the game. Discarding.")
            return

        logger.info("Saving completed game raw log to SQLite database...")
        try:
            game_id = await loop.run_in_executor(
                None,
                lambda: ingest_log(
                    db_path=self.config.database.db_path,
                    raw_text=raw_text,
                    source="live_bot",
                )
            )
            logger.info(f"Successfully saved game to DB with id={game_id}")
            print(f">>> SAVED AS GAME ID {game_id}")
        except Exception as e:
            logger.error(f"Failed to persist game to database: {e}")

    async def _handle_pm(self, sender: str, msg: str):
        # Clean prefix decorator if any (e.g. %Host -> Host)
        from ..io.player_names import canonical_player_name
        clean_sender = canonical_player_name(sender)
        logger.info(f"Received PM from {clean_sender}: {msg}")

        if self.tracker.in_game and not self.tracker.eliminated and "send" in msg.lower():
            random_player = self._get_random_live_player()
            if random_player:
                logger.info(f"Responding to PM from {clean_sender} with random player {random_player}")
                await self.connection.send(f"|/pm {clean_sender}, {random_player}")
            else:
                logger.info(f"No live players available to respond to PM from {clean_sender}")
                await self.connection.send(f"|/pm {clean_sender}, no one")
            return

        if self.tracker.in_game and not self.tracker.eliminated and any(token in msg.lower() for token in ["claim", "claiming"]):
            session = self.tracker.get_game_session()
            claim_message = self._get_claim_message(session)
            if claim_message:
                logger.info(f"Responding to claim PM from {clean_sender} with {claim_message}")
                await self.connection.send(f"|/pm {clean_sender}, {claim_message}")
            else:
                logger.info(f"No claim message available for PM from {clean_sender}")
                await self.connection.send(f"|/pm {clean_sender}, unknown")
            return
        
        # Command parsing:
        # !vote [player] -> override vote
        # !multiplier [player] [value] -> multiplier
        # !reset -> clear overrides
        parts = msg.strip().split()
        if not parts:
            return

        cmd = parts[0].lower()
        if cmd == "!vote" and len(parts) > 1:
            target = parts[1]
            self.strategy.set_manual_vote(target)
            await self.connection.send(f"|/pm {clean_sender}, Manual vote set to: {target}")
            # Trigger immediate vote update if in game
            if self.tracker.state == "DAY":
                await self._evaluate_and_vote()
                
        elif cmd == "!multiplier" and len(parts) > 2:
            target = parts[1]
            try:
                val = float(parts[2])
                self.strategy.set_suspicion_multiplier(target, val)
                await self.connection.send(f"|/pm {clean_sender}, Set suspicion multiplier for {target} to {val}")
                # Trigger immediate vote update if in game
                if self.tracker.state == "DAY":
                    await self._evaluate_and_vote()
            except ValueError:
                await self.connection.send(f"|/pm {clean_sender}, Invalid multiplier value. Must be float.")
                
        elif cmd == "!reset":
            self.strategy.reset()
            await self.connection.send(f"|/pm {clean_sender}, Reset bot override states.")
            if self.tracker.state == "DAY":
                await self._evaluate_and_vote()
