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

# Matches the role-assignment PM, e.g. "zorq_bot, you are a Vanilla Townie"
OWN_ROLE_PM_RE = re.compile(r"^(?P<name>.+?),\s*you\s+are\s+an?\s+(?P<role>.+?)\.?\s*$", re.IGNORECASE)

# Roles the bot should claim VT (Vanilla Townie) for instead of its real role.
LIE_AS_VT_ROLE_RE = re.compile(r"\b(werewolf|alien|cult\w*|serial\s+killer|goo)\b", re.IGNORECASE)

VALID_ALIGNMENTS = {"town", "mafia", "neutral", "unknown"}


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
        self._current_town_read: Optional[str] = None
        self._own_role: Optional[str] = None
        self._claimed_this_day: bool = False
        self._main_task: Optional[asyncio.Task] = None
        self._random_actions_task: Optional[asyncio.Task] = None
        self._random_vote_task: Optional[asyncio.Task] = None
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

    def _get_strategy_town_read(self, session) -> tuple[Optional[str], float]:
        return self.strategy.get_town_read(
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

    def _get_claim_message(self) -> Optional[str]:
        """Builds the v-1 claim message from the bot's own (live) role.

        Lies and claims VT for roles that would be too costly to reveal
        while alive; claims honestly otherwise.
        """
        if not self._own_role:
            return None

        if LIE_AS_VT_ROLE_RE.search(self._own_role):
            return "VT 1 to hammer"
        return f"{self._own_role} 1 to hammer"

    @staticmethod
    def _compute_live_vote_counts(session) -> dict:
        """Tallies currently-active votes per target, resetting at each day boundary."""
        active_votes: dict = {}
        counts: dict = {}
        current_day = None
        for vote in session.votes:
            if current_day is None:
                current_day = vote.day
            if vote.day != current_day:
                active_votes.clear()
                counts.clear()
                current_day = vote.day

            voter = vote.voter_name
            prev_target = active_votes.pop(voter, None)
            if prev_target:
                counts[prev_target] = counts.get(prev_target, 0) - 1

            if vote.action != "unvote":
                target = vote.target_name
                active_votes[voter] = target
                counts[target] = counts.get(target, 0) + 1

        return counts

    async def _maybe_claim_at_v1(self):
        """Proactively claims in room chat once the bot is one vote from being hammered."""
        if self._claimed_this_day or self.tracker.hammer_count is None:
            return

        from ..io.player_names import canonical_player_name

        session = self.tracker.get_game_session()
        counts = self._compute_live_vote_counts(session)

        bot_clean = canonical_player_name(self.config.showdown.username)
        bot_votes = next(
            (count for target, count in counts.items() if canonical_player_name(target) == bot_clean),
            0,
        )

        if bot_votes != self.tracker.hammer_count - 1:
            return

        claim_message = self._get_claim_message()
        if not claim_message:
            return

        logger.info(f"At v-1 (votes={bot_votes}, hammer={self.tracker.hammer_count}); claiming: {claim_message}")
        self._claimed_this_day = True
        await self._send_chat_message(claim_message)

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

    async def _send_chat_message(self, text: str):
        """Send a spoken chat line with a human-like typing delay beforehand.

        Unlike /mafia slash commands (which are instant UI actions), a chat
        line should feel like someone actually typed it before it posts.
        """
        typing_seconds = len(text) * random.uniform(0.04, 0.09)
        delay = min(random.uniform(0.6, 1.8) + typing_seconds, 10.0)
        await asyncio.sleep(delay)
        await self.send_room_command(text)

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
            if not self._random_actions_task or self._random_actions_task.done():
                self._random_actions_task = asyncio.create_task(self._random_actions_loop())
            if not self._random_vote_task or self._random_vote_task.done():
                self._random_vote_task = asyncio.create_task(self._random_vote_loop())

    async def stop(self):
        logger.info("Stopping Mafia Bot...")
        if self._random_actions_task:
            self._random_actions_task.cancel()
        if self._random_vote_task:
            self._random_vote_task.cancel()
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
                        catchphrases = ["why me", "im town", "get off", "bruh", "they're gonna qh"]
                        await self._send_chat_message(random.choice(catchphrases))
                        voter = self._extract_vote_voter(line)
                        if voter:
                            logger.info(f"Voting back on {voter} after being voted")
                            await self.send_room_command(f"/mafia vote {voter}")

                    if (
                        self._ready_for_live_games
                        and self.tracker.state == "DAY"
                        and self.tracker.in_game
                        and not self.tracker.eliminated
                        and "voted" in line.lower()
                    ):
                        await self._maybe_claim_at_v1()

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
            self._current_town_read = None
            self._own_role = None
            self._claimed_this_day = False
            if not self._random_actions_task or self._random_actions_task.done():
                self._random_actions_task = asyncio.create_task(self._random_actions_loop())
            if not self._random_vote_task or self._random_vote_task.done():
                self._random_vote_task = asyncio.create_task(self._random_vote_loop())

        elif event == "DAY":
            logger.info("New day started! Re-evaluating votes...")
            self._claimed_this_day = False
            if not self._random_actions_task or self._random_actions_task.done():
                self._random_actions_task = asyncio.create_task(self._random_actions_loop())
            if not self._random_vote_task or self._random_vote_task.done():
                self._random_vote_task = asyncio.create_task(self._random_vote_loop())
            await self._evaluate_and_vote()

        elif event in ("DEADLINE_3MIN", "DEADLINE_1MIN"):
            # The room only announces these two automatic warnings, so together
            # with the day-start evaluation above they give exactly the "3
            # times a day" re-evaluation cadence, tied to real game milestones
            # instead of a fixed timer.
            logger.info(f"Deadline warning ({event}); re-evaluating votes...")
            await self._evaluate_and_vote()

        elif event == "NIGHT":
            if self._random_actions_task:
                self._random_actions_task.cancel()
            if self._random_vote_task:
                self._random_vote_task.cancel()

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
            if self._random_actions_task:
                self._random_actions_task.cancel()
            if self._random_vote_task:
                self._random_vote_task.cancel()

            logger.info("Game finished. Sending gg message.")
            await self.send_room_command("gg")

            # Log completed game to database
            await self._save_game_to_db()

            # Reset tracker/strategy
            self.tracker.reset()
            self.strategy.reset()
            self._current_vote_target = None
            self._current_town_read = None
            self._own_role = None
            self._claimed_this_day = False

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
        from ..io.player_names import names_match

        parts = line.split("|")
        if len(parts) < 4:
            return
        if parts[1] not in {"c:", "c"}:
            return

        if parts[1] == "c:":
            sender = parts[3] if len(parts) > 3 else ""
            message_text = "|".join(parts[4:]).strip()
        else:  # "c"
            sender = parts[2] if len(parts) > 2 else ""
            message_text = "|".join(parts[3:]).strip()

        # Never learn to repeat our own lines back at the room.
        if self.config.showdown.username and names_match(sender, self.config.showdown.username):
            return

        if not message_text:
            return
        if message_text.lower().startswith("/mafia"):
            return
        if message_text.lower().startswith("!" ):
            return
        if len(message_text) > 120:
            return

        self._remembered_lines.append(message_text)
        if len(self._remembered_lines) > 12:
            self._remembered_lines = self._remembered_lines[-12:]

    async def _random_actions_loop(self):
        """Periodically says filler chat, so the bot doesn't sit silent all day.

        Each message goes through _send_chat_message so it's paced like
        someone actually typing it, rather than several lines firing at once.
        """
        filler_words = ["bruh", "hm", "oh", "welp", "bleh", "uhh", "thinking", "...", "interesting", "lol", "i mean"]

        while self.tracker.state == "DAY" and self.tracker.in_game and not self.tracker.eliminated:
            try:
                await asyncio.sleep(random.uniform(45.0, 90.0))
                if self.tracker.state != "DAY" or not self.tracker.in_game or self.tracker.eliminated:
                    break

                word = random.choice(filler_words)
                logger.info(f"Saying random filler word: {word}")
                await self._send_chat_message(word)

                if not self.tracker.in_game:
                    logger.info("Bot is no longer in the game; stopping random actions loop.")
                    break

                if self._remembered_lines and random.random() < 0.5:
                    remembered_line = random.choice(self._remembered_lines)
                    logger.info(f"Repeating remembered line: {remembered_line}")
                    await self._send_chat_message(remembered_line)

                if self.tracker.state != "DAY" or not self.tracker.in_game or self.tracker.eliminated:
                    break

                if random.random() < 0.5:
                    session = self.tracker.get_game_session()
                    question_prompt = self._build_question_prompt(session)
                    if question_prompt:
                        logger.info(f"Asking random question: {question_prompt}")
                        await self._send_chat_message(question_prompt)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in random actions loop: {e}")

    async def _random_vote_loop(self):
        """Casts occasional exploratory votes while unsure, so the bot's
        voting activity doesn't look robotic (silent until confident, then
        one decisive vote). Respects the room's vote-lock cooldown before
        retracting, and backs off entirely once a confident target emerges.
        """
        from ..io.player_names import canonical_player_name

        while self.tracker.state == "DAY" and self.tracker.in_game and not self.tracker.eliminated:
            try:
                await asyncio.sleep(random.uniform(20.0, 45.0))
                if self.tracker.state != "DAY" or not self.tracker.in_game or self.tracker.eliminated:
                    break

                session = self.tracker.get_game_session()
                if not session.players or not self._should_do_random_vote(session):
                    continue

                if random.random() > self.config.gameplay.random_vote_chance:
                    continue

                bot_clean = canonical_player_name(self.config.showdown.username)
                alive_players = [p for p in session.players if p not in self.tracker.dead_players]
                valid_targets = [p for p in alive_players if canonical_player_name(p) != bot_clean]
                if not valid_targets:
                    continue

                target = random.choice(valid_targets)
                logger.info(f"Casting random exploratory vote on {target}")
                await self.send_room_command(f"/mafia vote {target}")

                lock_wait = self.config.gameplay.min_seconds_between_vote_actions + random.uniform(0.0, 6.0)
                await asyncio.sleep(lock_wait)

                if self.tracker.state != "DAY" or self.tracker.eliminated or self._current_vote_target is not None:
                    # A confident strategy vote took over in the meantime; don't clobber it.
                    continue

                fresh_session = self.tracker.get_game_session()
                if self._should_do_random_vote(fresh_session):
                    logger.info(f"Retracting random exploratory vote on {target}")
                    await self.send_room_command("/mafia unvote")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in random vote loop: {e}")

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
                await self._cast_vote_with_optional_comment(target)
                self._current_vote_target = target
            else:
                logger.info(f"Maintaining current vote on {target} (confidence: {confidence:.2%})")
        else:
            if self._current_vote_target:
                logger.info("No clear target meets confidence threshold. Unvoting.")
                await self.send_room_command("/mafia unvote")
                self._current_vote_target = None

        town_read, town_confidence = self._get_strategy_town_read(session)
        if town_read:
            if town_read != self._current_town_read:
                logger.info(f"Decided {town_read} is town (confidence: {town_confidence:.2%}). Previous: {self._current_town_read}")
                if random.random() < self.config.gameplay.town_read_comment_chance:
                    await self._send_chat_message(f"{town_read} is town")
                self._current_town_read = town_read
            else:
                logger.info(f"Maintaining current town read on {town_read} (confidence: {town_confidence:.2%})")
        else:
            self._current_town_read = None

    async def _cast_vote_with_optional_comment(self, target: str):
        """Casts a vote the way a person actually would: not always narrated,
        and when it is, not always glued to the vote in the same order.
        """
        if random.random() >= self.config.gameplay.vote_comment_chance:
            await self.send_room_command(f"/mafia vote {target}")
            return

        comment = f"I think {target} is scum"
        if random.random() < 0.5:
            await self._send_chat_message(comment)
            await asyncio.sleep(random.uniform(1.0, 3.0))
            await self.send_room_command(f"/mafia vote {target}")
        else:
            await self.send_room_command(f"/mafia vote {target}")
            await asyncio.sleep(random.uniform(2.0, 6.0))
            await self._send_chat_message(comment)

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
            await self._prompt_for_undefined_roles(self.config.database.db_path, game_id)
        except Exception as e:
            logger.error(f"Failed to persist game to database: {e}")

    async def _prompt_for_undefined_roles(self, db_path: str, game_id: int) -> None:
        """Lets the host fill in roles for players who never flipped (e.g.
        survivors), right at the terminal, instead of needing the dashboard.
        """
        from ..services.game_service import assign_player_role, find_undefined_players

        loop = asyncio.get_running_loop()
        undefined = await loop.run_in_executor(None, find_undefined_players, db_path, game_id)
        if not undefined:
            return

        print(f"\n{len(undefined)} player(s) have no recorded role (likely survived to the end):")
        for row in undefined:
            hint = "has chat" if row.has_messages else "silent"
            if row.is_inferred_town_candidate:
                hint += ", inferred town"
            prompt = f"  Role for {row.player_name} ({hint}) [town/mafia/neutral/unknown, blank=skip]: "
            answer = (await loop.run_in_executor(None, input, prompt)).strip().lower()
            if not answer:
                continue
            if answer not in VALID_ALIGNMENTS:
                print(f"  Skipping {row.player_name}: {answer!r} is not a valid role.")
                continue
            assign_player_role(db_path, game_id, row.player_name, answer)
            print(f"  Set {row.player_name} to {answer}.")

    async def _handle_pm(self, sender: str, msg: str):
        # Clean prefix decorator if any (e.g. %Host -> Host)
        from ..io.player_names import canonical_player_name, names_match

        # Showdown echoes our own outgoing PMs back through the same queue;
        # never treat our own messages as an incoming request.
        if self.config.showdown.username and names_match(sender, self.config.showdown.username):
            return

        clean_sender = canonical_player_name(sender)
        logger.info(f"Received PM from {clean_sender}: {msg}")

        own_role_match = OWN_ROLE_PM_RE.match(msg.strip())
        if own_role_match and self.config.showdown.username and names_match(own_role_match.group("name"), self.config.showdown.username):
            self._own_role = own_role_match.group("role").strip()
            logger.info(f"Learned own role from PM: {self._own_role}")
            return

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
            claim_message = self._get_claim_message()
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
