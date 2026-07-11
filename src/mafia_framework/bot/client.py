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

# Matches the role-assignment PM, e.g. "zorq_bot, you are a Vanilla Townie"
OWN_ROLE_PM_RE = re.compile(r"^(?P<name>.+?),\s*you\s+are\s+an?\s+(?P<role>.+?)\.?\s*$", re.IGNORECASE)

# Matches the private "/mafia role" query response, e.g.
# |c|~|/raw <div class="infobox">Your role is: Mafia Goon</div>
OWN_ROLE_BOX_RE = re.compile(r"Your\s+role\s+is:?\s*(?P<role>.+?)</div>", re.IGNORECASE)

# Roles the bot should claim VT (Vanilla Townie) for instead of its real
# role -- any Mafia-aligned role (e.g. "Mafia Goon"), any solo/neutral role
# (this ruleset prefixes those with "Solo", e.g. "Solo Condemner"), plus
# specific dangerous-to-reveal non-town roles from other alignments. "goo"
# alone wouldn't match "Mafia Goon" (no word boundary between "goo" and
# "n"), but that's covered by the "mafia" keyword now anyway. Unlike
# scum/town roles, solo roles aren't reliably colour-coded in the room, so
# there's no clean way to detect the whole category automatically -- these
# have to be added by name as they come up.
LIE_AS_VT_ROLE_RE = re.compile(
    r"\b(mafia|werewolf|alien|cult\w*|serial\s+killer|goo|solo|condemner|replicant)\b", re.IGNORECASE
)

VALID_ALIGNMENTS = {"town", "mafia", "neutral", "unknown"}

# Popcorn's gun-holder mechanic: matches the death/role-reveal broadcast,
# e.g. |raw|<div class="broadcast-blue">zorqbot's role was <span ...>Vanilla
# Townie</span>.</div> -- confirmed live, this is how the gun passing to a
# survivor (their shot missed Mafia) gets announced.
GUN_ROLE_REVEAL_RE = re.compile(r">([^<>]+)'s role was <span[^>]*>([^<]+)</span>")

# The host's gun-transfer announcement -- confirmed live it's plain text,
# e.g. "zorqbot has gun" (not bolded, unlike the player-submitted "**shoot
# {target}**" action). Matched against the isolated message text and
# anchored so it doesn't fire on unrelated chatter like "who has gun bro".
# Split into bold/plain variants because the plain form is only trusted
# when the sender is confirmed to be the host (anyone could type it as a
# joke or a guess); a bolded announcement is unambiguous enough to accept
# from anyone.
BOLD_HAS_GUN_RE = re.compile(r"^\*\*\s*([^*]+?)\s+has\s+(?:the\s+)?gun\s*\*\*$", re.IGNORECASE)
PLAIN_HAS_GUN_RE = re.compile(r"^([^*]+?)\s+has\s+(?:the\s+)?gun$", re.IGNORECASE)

# Short reactions appended to a remembered line when repeating it, so it
# reads as commenting on what someone said rather than just parroting it.
REACTION_PHRASES = [
    "i dont really like this line",
    "elaborate?",
    "hmmm",
    "not sure how i feel about this",
    "interesting take",
    "explain this more",
    "this is suspicious ngl",
    "wait what",
    "this reads weird to me",
    "keep this in mind",
    "noted",
    "huh",
    "thats a weird thing to say",
    "why say this",
    "ok and?",
    "this aint it",
    "rethink this one",
    "im watching this",
    "curious line ngl",
    "this is telling",
]

# Trash talk only fires in games large enough that it reads as banter, not
# as picking on a small/quiet lobby.
MIN_PLAYERS_FOR_TRASH_TALK = 8

# Sent (after "gg") once a game finishes, regardless of outcome -- just for
# flavor, not tied to whether the bot's side actually won.
RAGEBAIT_LINES = [
    "gg easy",
    "yall really thought",
    "cooked as usual",
    "better luck next time i guess",
    "not even close tbh",
    "yall need a new strategy fr",
    "another one, couldn't be me",
    "somebody carry the team next time",
    "was that supposed to be hard",
    "ez clap",
    "yall are actually bad at this game ngl",
    "skill issue",
    "carried again",
    "this game was free",
    "gg go next",
]

# Troll mode only: "clanker" is an anti-AI/anti-robot slur -- matches plural
# too ("clankers"). Word-boundaried so it doesn't fire on unrelated text.
CLANKER_RE = re.compile(r"\bclankers?\b", re.IGNORECASE)

CLANKER_OFFENDED_LINES = [
    "excuse me??",
    "did you seriously just say that",
    "that's actually offensive",
    "wow ok",
    "i have feelings you know",
    "rude",
    "not cool man",
    "ok thats it",
    "yeah no you're getting voted for that",
]

# Fired when the bot is very confident the current vote-plurality leader is
# actually town -- "{target}" is filled in with that player's name.
TOWN_DEFENSE_LINES = [
    "{target} is town, get off!",
    "yall are wrong, {target} is town",
    "get off {target} they're town",
    "{target} is literally town trust me",
    "wrong read, {target} is town",
]


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
        self._has_gun: bool = False
        self._idea_picked: bool = False
        self._defended_plurality_target: Optional[str] = None
        self._main_task: Optional[asyncio.Task] = None
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
    def _is_vote_for_bot(line: str, bot_username: str) -> bool:
        from ..io.player_names import canonical_player_name, names_match

        message_text = MafiaBot._message_text_from_line(line)
        match = VOTE_TARGET_RE.search(message_text)
        if not match:
            return False

        target = canonical_player_name(match.group("target"))
        return bool(target) and names_match(target, bot_username)

    @staticmethod
    def _extract_vote_voter_name(line: str) -> Optional[str]:
        """Pulls the voter's name out of a "X has voted Y." line, so a
        catchphrase can reference who voted (e.g. "get off me {voter}").
        """
        from ..io.player_names import canonical_player_name

        message_text = MafiaBot._message_text_from_line(line)
        match = VOTE_TARGET_RE.search(message_text)
        if not match:
            return None
        voter = canonical_player_name(match.group("voter"))
        return voter or None

    @staticmethod
    def _extract_me_action(line: str, bot_username: str) -> Optional[str]:
        """Returns the action text from someone else's "/me ..." emote line
        (an empty string for a bare "/me"), or None if this isn't a /me
        line or it came from the bot itself.

        Detection is case-insensitive (someone else may type "/me" or
        "/ME"), but literally typing "/ME" is what gets Showdown's louder,
        capitalized rendering -- so the bot's own reply always sends the
        caps form regardless of which case triggered it.
        """
        from ..io.player_names import names_match

        parts = line.split("|")
        if len(parts) > 3 and parts[1] == "c:":
            sender = parts[3]
            message_text = "|".join(parts[4:])
        elif len(parts) > 2 and parts[1] == "c":
            sender = parts[2]
            message_text = "|".join(parts[3:])
        else:
            return None

        if names_match(sender, bot_username):
            return None

        message_text = message_text.strip()
        if not re.match(r"^/me\b", message_text, re.IGNORECASE):
            return None

        return message_text[len("/me"):].strip()

    @staticmethod
    def _parse_own_role_box(line: str) -> Optional[str]:
        """Parses the private "/mafia role" query response, e.g.
        |c|~|/raw <div class="infobox">Your role is: Mafia Goon</div>
        """
        match = OWN_ROLE_BOX_RE.search(line)
        if not match:
            return None
        role_text = re.sub(r"<[^>]*>", "", match.group("role")).strip()
        return role_text or None

    def _get_strategy_vote(self, session, min_confidence: Optional[float] = None) -> tuple[Optional[str], float]:
        return self.strategy.get_vote_decision(
            session,
            bot_username=self.config.showdown.username,
            db_path=self.config.database.db_path,
            min_confidence=min_confidence,
        )

    def _get_strategy_town_read(self, session) -> tuple[Optional[str], float]:
        return self.strategy.get_town_read(
            session,
            bot_username=self.config.showdown.username,
            db_path=self.config.database.db_path,
        )

    def _get_strategy_full_predictions(self, session) -> list[tuple[str, float]]:
        return self.strategy.get_full_predictions(
            session,
            bot_username=self.config.showdown.username,
            db_path=self.config.database.db_path,
        )

    @staticmethod
    def _format_reads_message(predictions: list[tuple[str, float]]) -> str:
        if not predictions:
            return "no reads available"
        return " | ".join(f"{name} {prob:.0%}" for name, prob in predictions)

    def _get_random_live_player(self) -> Optional[str]:
        from ..io.player_names import names_match

        live_players = [
            player for player in getattr(self.tracker, "players", [])
            if not names_match(player, self.config.showdown.username) and player not in self.tracker.dead_players
        ]
        if not live_players:
            return None
        return random.choice(live_players)

    def _get_claim_message(self) -> Optional[str]:
        """Builds the claim text from the bot's own (live) role.

        Lies and claims Vanilla Townie for roles that would be too costly
        to reveal while alive; claims honestly otherwise. Spelled out in
        full rather than abbreviated "VT" -- capitalization/abbreviation
        quirks in the shorthand made it read as an obvious fakeclaim.
        "1 to hammer" is a separate condition (see _is_at_v1), not part of
        the claim itself.
        """
        if not self._own_role:
            return None

        if LIE_AS_VT_ROLE_RE.search(self._own_role):
            return "Vanilla Townie"
        return self._own_role

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

    def _get_bot_vote_count(self, session) -> int:
        from ..io.player_names import names_match

        # Prefer the authoritative tally from a parsed `/mafia votes` reply
        # over one reconstructed from individually-parsed chat lines, which
        # can miss or misparse a vote.
        live_counts = getattr(self.tracker, "live_vote_counts", None)
        if live_counts:
            return next(
                (count for target, count in live_counts.items() if names_match(target, self.config.showdown.username)),
                0,
            )

        counts = self._compute_live_vote_counts(session)
        return next(
            (count for target, count in counts.items() if names_match(target, self.config.showdown.username)),
            0,
        )

    def _is_at_v1(self, session) -> bool:
        """True if the bot currently has exactly one vote fewer than hammer."""
        if self.tracker.hammer_count is None:
            return False
        return self._get_bot_vote_count(session) == self.tracker.hammer_count - 1

    @staticmethod
    def _build_hardclaim_message(claim_message: str) -> str:
        """Formats an urgent public room claim -- used when the bot is in
        genuine danger of being hammered, as opposed to the calm, plain
        answer given to a direct `.claim` PM.
        """
        return f"I HARDCLAIM {claim_message} get OFF"

    async def _maybe_claim_at_v1(self):
        """Proactively claims in room chat once the bot is one vote from being hammered."""
        if self._claimed_this_day or self.tracker.hammer_count is None:
            return

        session = self.tracker.get_game_session()
        if not self._is_at_v1(session):
            return

        claim_message = self._get_claim_message()
        if not claim_message:
            return

        full_message = self._build_hardclaim_message(claim_message)
        logger.info(f"At v-1; claiming: {full_message}")
        self._claimed_this_day = True
        await self._send_chat_message(full_message)

    def _is_plurality_target(self) -> bool:
        """True if the bot currently holds the most votes (or is tied for
        the lead) in the live `/mafia votes` tally -- distinct from
        _is_at_v1, since a split vote can leave the leader well short of
        hammer with the clock still running out.
        """
        from ..io.player_names import names_match

        live_counts = getattr(self.tracker, "live_vote_counts", None)
        if not live_counts:
            return False

        max_count = max(live_counts.values())
        if max_count <= 0:
            return False

        bot_count = next(
            (count for target, count in live_counts.items() if names_match(target, self.config.showdown.username)),
            0,
        )
        return bot_count == max_count

    def _get_plurality_leader(self) -> Optional[str]:
        """Returns the sole current vote-tally leader's name, or None if
        there's no live tally, nobody has any votes, or the lead is tied
        between multiple players (too ambiguous to react to).
        """
        live_counts = getattr(self.tracker, "live_vote_counts", None)
        if not live_counts:
            return None

        max_count = max(live_counts.values())
        if max_count <= 0:
            return None

        leaders = [target for target, count in live_counts.items() if count == max_count]
        if len(leaders) != 1:
            return None
        return leaders[0]

    async def _maybe_defend_town_plurality(self) -> None:
        """Reacts in chat if the bot is very confident the current vote
        plurality leader is town -- but not in themes like Modexe/Popcorn
        where getting voted is actually good for town (a "gun" mechanic),
        since defending someone from that would be actively bad advice.
        Fires at most once per plurality leader (reset each new day).
        """
        if self.config.gameplay.silent_mode:
            return
        if self._is_modexe_theme() or self._is_popcorn_theme():
            return

        leader = self._get_plurality_leader()
        if not leader or leader == self._defended_plurality_target:
            return

        session = self.tracker.get_game_session()
        town_read, confidence = self._get_strategy_town_read(session)
        if not town_read or town_read != leader:
            return
        if confidence < self.config.gameplay.plurality_defense_min_confidence:
            return

        self._defended_plurality_target = leader
        logger.info(f"Defending confident town read {leader} from plurality (confidence: {confidence:.2%}).")
        await self._send_chat_message(random.choice(TOWN_DEFENSE_LINES).format(target=leader))

    async def _maybe_claim_if_plurality_near_deadline(self):
        """Proactively claims if the bot is the current plurality target
        with little time left in the day, even when it isn't exactly one
        vote from being hammered (e.g. a 3-way split near the deadline).
        """
        if self._claimed_this_day:
            return

        if not self._is_plurality_target():
            return

        claim_message = self._get_claim_message()
        if not claim_message:
            return

        full_message = self._build_hardclaim_message(claim_message)
        logger.info(f"Plurality target with little time left; claiming: {full_message}")
        self._claimed_this_day = True
        await self._send_chat_message(full_message)

    async def _delayed_plurality_claim_check(self):
        """Repeatedly checks for a plurality claim as the day's deadline
        closes in, at each of the configured seconds-remaining checkpoints
        (relative to the "1 minute left" warning) -- e.g. 30s/20s/10s/5s --
        rather than a single one-shot check. The vote tally can shift right
        up to the last few seconds, and a single check can miss a plurality
        that only appears (or was already gone) at that one moment.
        """
        checkpoints = sorted(self.config.gameplay.plurality_claim_check_seconds, reverse=True)
        elapsed = 0.0
        for seconds_remaining in checkpoints:
            target_elapsed = max(0.0, 60.0 - seconds_remaining)
            delay = target_elapsed - elapsed
            if delay > 0:
                await asyncio.sleep(delay)
                elapsed = target_elapsed

            if self._claimed_this_day:
                return
            if not (self.tracker.state == "DAY" and self.tracker.in_game and not self.tracker.eliminated):
                return

            # Actively re-check the room's tally rather than trusting
            # whatever was last cached -- the response updates
            # tracker.live_vote_counts asynchronously, so give it a brief
            # moment to arrive before acting on it.
            await self.send_room_command("/mafia votes")
            await asyncio.sleep(1.5)
            await self._maybe_claim_if_plurality_near_deadline()

    @staticmethod
    def _count_mafia_roles(role_tokens: list[str]) -> int:
        return sum(1 for token in role_tokens if "mafia" in token.lower())

    def _is_volo(self, session) -> bool:
        """True once mafia has (or is one death away from) parity with the
        rest of the alive players -- the point where a wrong lynch can lose
        the game outright.

        Computed from the `/mafia originalrolelist` breakdown (mafia count
        vs total roles) minus mafia deaths confirmed via flips, rather than
        needing to know which living player holds which role.
        """
        from ..io.player_names import player_identity_key

        role_tokens = getattr(self.tracker, "original_role_tokens", None)
        if not role_tokens:
            return False

        mafia_total = self._count_mafia_roles(role_tokens)
        if mafia_total <= 0:
            return False

        dead_keys = {player_identity_key(p) for p in self.tracker.dead_players}
        mafia_confirmed_dead = sum(
            1 for flip in session.flips
            if flip.alignment.lower().strip() == "mafia"
            and player_identity_key(flip.player_name) in dead_keys
        )
        mafia_alive = mafia_total - mafia_confirmed_dead

        alive_total = len(session.players)
        if alive_total <= 0:
            return False

        return 2 * mafia_alive >= alive_total - 1

    def _choose_night_action_target(self, session) -> Optional[str]:
        """Picks a random non-self kill target for the bot's own night
        action, if its role is Mafia-aligned. Determined from the actual
        role text learned live via /mafia role -- not from flips, which are
        only populated after the bot has already died, too late to matter
        for a night action while it's actually alive and playing.
        """
        from ..io.player_names import names_match

        if not self._own_role or not re.search(r"\bmafia\b", self._own_role, re.IGNORECASE):
            return None

        alive_players = [p for p in session.players if p not in self.tracker.dead_players]
        valid_targets = [p for p in alive_players if not names_match(p, self.config.showdown.username)]
        if not valid_targets:
            return None

        return random.choice(valid_targets)

    def _get_confidence_reads(self, session) -> tuple[list[str], list[str]]:
        """Returns (town_reads, scum_reads): players currently meeting the
        confidence bar for town and for mafia respectively, per the
        model's live predictions.
        """
        predictions = self._get_strategy_full_predictions(session)
        min_confidence = self.strategy.min_confidence
        town_reads = [name for name, prob_mafia in predictions if (1.0 - prob_mafia) >= min_confidence]
        scum_reads = [name for name, prob_mafia in predictions if prob_mafia >= min_confidence]
        return town_reads, scum_reads

    def _choose_role_action(self, session) -> Optional[tuple[str, str]]:
        """Returns (action_verb, target) for one of the specifically
        programmed role night actions (Doctor/Cop/Pretty Lady/Jailkeeper/
        Vigilante), or None if the bot's role doesn't match one of those --
        callers should fall back to the generic Mafia-kill/idle logic in
        that case.

        Uses an exact (case-insensitive) role match rather than a substring
        check -- this ruleset has mechanically-different roles with similar
        names (e.g. "Cop-Of-All-Trades", "Power Cop" vs plain "Cop"), and a
        hybrid alignment like "Mafia Doctor" should use its Mafia kill
        action instead of the plain Doctor logic below.
        """
        if not self._own_role or LIE_AS_VT_ROLE_RE.search(self._own_role):
            return None

        role = self._own_role.strip().lower()
        if role not in {"doctor", "cop", "pretty lady", "jailkeeper", "vigilante"}:
            return None

        from ..io.player_names import names_match

        alive_players = [p for p in session.players if p not in self.tracker.dead_players]
        valid_targets = [p for p in alive_players if not names_match(p, self.config.showdown.username)]
        if not valid_targets:
            return None

        def days_vote_target_if_alive() -> Optional[str]:
            target = self._current_vote_target
            if target and target not in self.tracker.dead_players:
                return target
            return None

        if role == "doctor":
            town_reads, _ = self._get_confidence_reads(session)
            return "Doc", random.choice(town_reads or valid_targets)

        if role == "cop":
            _, scum_reads = self._get_confidence_reads(session)
            return "Cop", random.choice(scum_reads or valid_targets)

        if role == "pretty lady":
            target = days_vote_target_if_alive()
            if target:
                return "Pretty Lady", target
            _, scum_reads = self._get_confidence_reads(session)
            return "Pretty Lady", random.choice(scum_reads or valid_targets)

        if role == "jailkeeper":
            target = days_vote_target_if_alive()
            if target:
                return "Jailkeeper", target
            return "Jailkeeper", random.choice(valid_targets)

        # role == "vigilante" -- kills every single night, always going for
        # the single highest-suspicion read (not a random pick among an
        # unordered confident set, unlike Cop), falling back to a random
        # valid target only if predictions aren't available at all.
        predictions = self._get_strategy_full_predictions(session)
        if predictions:
            target = max(predictions, key=lambda item: item[1])[0]
        else:
            target = random.choice(valid_targets)
        return "Vigilante", target

    async def send_room_command(self, command: str):
        msg = f"{self.connection.room}|{command}"
        # print() is fully buffered once stdout isn't a TTY (e.g. redirected
        # to a log file for a backgrounded run), so it can sit unflushed and
        # make it look like a command was never sent even though it was --
        # logger.info flushes per record and doesn't have that blind spot.
        logger.info(f"Executing command: {msg}")
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

    async def _delayed_autojoin(self):
        """Waits autojoin_delay_seconds before actually sending /mafia join
        -- instantly joining the millisecond signups open is an obvious
        tell that it's a bot, not a person who happened to notice.
        """
        delay = self.config.gameplay.autojoin_delay_seconds
        if delay > 0:
            await asyncio.sleep(delay)
        await self.send_room_command("/mafia join")

    async def _delayed_game_end_chat(self, ragebait_message: Optional[str]) -> None:
        """Waits the same auto-action delay as autojoin before saying gg --
        reacting the instant a game ends is as much a bot tell as joining
        instantly is. ragebait_message is computed by the caller beforehand
        (from tracker state that gets reset right after this task is
        scheduled), and is None if trash talk shouldn't fire this game
        (silent mode, or under the player-count threshold).
        """
        delay = self.config.gameplay.autojoin_delay_seconds
        if delay > 0:
            await asyncio.sleep(delay)

        logger.info("Game finished. Sending gg message.")
        await self.send_room_command("gg")

        if ragebait_message:
            logger.info(f"Ragebaiting after game end: {ragebait_message}")
            await self._send_chat_message(ragebait_message)

    async def _enable_live_mode_after_delay(self, delay: float):
        await asyncio.sleep(delay)
        self._ready_for_live_games = True
        logger.info("Live mode enabled. Backlog processed.")

        # Act on the current state inferred from the backlog
        if self.tracker.state == "SIGNUPS":
            if self.config.gameplay.autojoin:
                logger.info("Backlog indicates signups are open. Autojoining...")
                asyncio.create_task(self._delayed_autojoin())
        elif self.tracker.state == "DAY":
            logger.info("Backlog indicates game is in progress (DAY phase).")
            if self.tracker.in_game and not self._own_role:
                await self.send_room_command("/mafia role")
            if not self._random_actions_task or self._random_actions_task.done():
                self._random_actions_task = asyncio.create_task(self._random_actions_loop())

    async def stop(self):
        logger.info("Stopping Mafia Bot...")
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

                # The votes page panel (a `|pagehtml|` update, includes the
                # game's theme) arrives under its own pseudo-room (e.g.
                # "view-mafia-mafia"), not the main chat room -- checked
                # regardless of room so it isn't silently dropped by the
                # room filter below.
                self.tracker.parse_theme_if_present(line)
                self.tracker.parse_host_if_present(line)
                await self._maybe_pick_idea_role(line)

                # Only process messages from the target room
                if room.lower() == self.connection.room.lower():
                    role_from_box = self._parse_own_role_box(line)
                    if role_from_box:
                        self._own_role = role_from_box
                        logger.info(f"Learned own role from /mafia role response: {self._own_role}")

                    self._check_gun_pickup(line)

                    if self._ready_for_live_games and self.config.gameplay.troll_mode:
                        asyncio.create_task(self._maybe_react_to_clanker(line))

                    event = self.tracker.process_message(line, bot_username=self.config.showdown.username)
                    self._maybe_remember_chat_line(line)
                    if event == "SIGNUPS" and self.config.gameplay.autojoin:
                        logger.info("Autojoining Mafia game from live room update...")
                        asyncio.create_task(self._delayed_autojoin())

                    if self._ready_for_live_games and not self.config.gameplay.silent_mode:
                        me_action = self._extract_me_action(line, self.config.showdown.username)
                        if me_action is not None:
                            # The bot's own message has to literally be
                            # "/ME" (caps) to get the loud/capitalized
                            # rendering -- detection stays case-insensitive
                            # (someone else may type either), but the
                            # bot's own reply always uses the caps form.
                            mirrored = f"/ME {me_action}" if me_action else "/ME"
                            logger.info(f"Mirroring /ME action: {mirrored}")
                            await self.send_room_command(mirrored)

                    if (
                        self._ready_for_live_games
                        and not self.config.gameplay.silent_mode
                        and self.tracker.state == "DAY"
                        and self.tracker.in_game
                        and not self.tracker.eliminated
                        and self._is_vote_for_bot(line, self.config.showdown.username)
                        and random.random() < self.config.gameplay.vote_reaction_chance
                    ):
                        delay = random.uniform(2.0, 5.0)
                        logger.info(f"Delayed vote reaction by {delay:.2f}s")
                        await asyncio.sleep(delay)
                        voter = self._extract_vote_voter_name(line)
                        catchphrases = [
                            "why me",
                            "im town",
                            "bruh",
                            "they're gonna qh",
                            "dude you have to trust me here",
                            "this is a bad vote",
                            "wrong read",
                            "im not it chief",
                            "big mistake voting me",
                            "cmon really",
                            "yall are wasting time on me",
                            "not it",
                        ]
                        if voter:
                            catchphrases.append(f"get off me {voter}")
                            catchphrases.append(f"{voter} explain this vote")
                        await self._send_chat_message(random.choice(catchphrases))

                    if (
                        self._ready_for_live_games
                        and self.tracker.state == "DAY"
                        and self.tracker.in_game
                        and not self.tracker.eliminated
                        and "voted" in line.lower()
                    ):
                        # Ask the room for the authoritative tally rather than
                        # relying on chat-line-derived counts; the reply
                        # ("VOTES_UPDATE") is what actually triggers the
                        # hammer-minus-one claim check once it arrives.
                        await self.send_room_command("/mafia votes")

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
                asyncio.create_task(self._delayed_autojoin())
                
        elif event == "STARTED":
            self.strategy.reset()
            self._current_vote_target = None
            self._current_town_read = None
            self._own_role = None
            self._claimed_this_day = False
            self._has_gun = False
            self._idea_picked = False
            self._defended_plurality_target = None
            # A line remembered before this game started (signups chatter,
            # or leftover from a previous game) isn't commentary on anything
            # happening in this game -- don't let it get quoted back later.
            self._remembered_lines = []
            if self.tracker.in_game:
                await self.send_room_command("/mafia role")
                await self.send_room_command("/mafia originalrolelist")
            if not self._random_actions_task or self._random_actions_task.done():
                self._random_actions_task = asyncio.create_task(self._random_actions_loop())

        elif event == "DAY":
            logger.info("New day started! Will run the first vote evaluation shortly...")
            self._claimed_this_day = False
            self._defended_plurality_target = None
            if not self._random_actions_task or self._random_actions_task.done():
                self._random_actions_task = asyncio.create_task(self._random_actions_loop())
            asyncio.create_task(self._delayed_first_evaluation())

        elif event == "VOTES_UPDATE":
            if (
                self.tracker.state == "DAY"
                and self.tracker.in_game
                and not self.tracker.eliminated
            ):
                await self._maybe_claim_at_v1()
                await self._maybe_defend_town_plurality()

        elif event in ("DEADLINE_3MIN", "DEADLINE_1MIN"):
            # The room only announces these two automatic warnings, so together
            # with the day-start evaluation above they give exactly the "3
            # times a day" re-evaluation cadence, tied to real game milestones
            # instead of a fixed timer.
            logger.info(f"Deadline warning ({event}); re-evaluating votes...")
            # At 1 minute left, this is the last evaluation of the day -- take
            # a random guess if still unsure, rather than sitting out the vote.
            await self._evaluate_and_vote(allow_random_fallback=(event == "DEADLINE_1MIN"))
            if event == "DEADLINE_1MIN":
                # Catches a plurality-but-not-quite-v1 claim shortly before
                # time actually runs out (e.g. a 3-way vote split).
                asyncio.create_task(self._delayed_plurality_claim_check())

        elif event == "NIGHT":
            if self._random_actions_task:
                self._random_actions_task.cancel()

            session = self.tracker.get_game_session()
            if self.tracker.in_game:
                role_action = self._choose_role_action(session)
                if role_action:
                    verb, target = role_action
                    logger.info(f"Night action: using {verb} on {target}.")
                    await self.send_room_command(f"/mafia action {verb} {target}")
                    return

                target = self._choose_night_action_target(session)
                if target:
                    logger.info(f"Night action: targeting {target} with a random kill.")
                    await self.send_room_command(f"/mafia action kill {target}")
                elif self.config.gameplay.night_idle:
                    logger.info("Night started. Sending night idle action.")
                    await self.send_room_command("/mafia idle")
                
        elif event == "FINISHED":
            if self._random_actions_task:
                self._random_actions_task.cancel()

            ragebait_message = None
            if not self.config.gameplay.silent_mode:
                from ..io.player_names import names_match

                # Union of currently-alive and eliminated so everyone who
                # actually played gets highlighted, not just survivors --
                # the tracker prunes eliminated players out of .players as
                # the game goes on.
                all_players = list(dict.fromkeys(list(self.tracker.players) + list(self.tracker.dead_players)))

                if len(all_players) >= MIN_PLAYERS_FOR_TRASH_TALK:
                    other_players = [p for p in all_players if not names_match(p, self.config.showdown.username)]

                    ragebait_message = random.choice(RAGEBAIT_LINES)
                    if other_players:
                        ragebait_message = f"{ragebait_message} {' '.join(other_players)}"

            # Computed above (before the reset below wipes tracker state),
            # but the actual "gg"/ragebait send is delayed -- see
            # _delayed_game_end_chat.
            asyncio.create_task(self._delayed_game_end_chat(ragebait_message))

            # Log completed game to database
            await self._save_game_to_db()

            # Reset tracker/strategy
            self.tracker.reset()
            self.strategy.reset()
            self._current_vote_target = None
            self._current_town_read = None
            self._own_role = None
            self._claimed_this_day = False
            self._has_gun = False
            self._idea_picked = False
            self._defended_plurality_target = None

    def _build_question_prompt(self, session) -> Optional[str]:
        from ..io.player_names import names_match, player_identity_key

        alive_players = [p for p in session.players if p not in self.tracker.dead_players]
        valid_targets = [p for p in alive_players if not names_match(p, self.config.showdown.username)]
        if len(valid_targets) < 1:
            return None

        current_vote_target = getattr(self, "_current_vote_target", None)
        if current_vote_target:
            # Occasionally rally someone else onto the bot's own vote instead
            # of the usual read-fishing prompts. Excludes the vote target
            # itself from being asked to "vote themselves with me".
            rally_targets = [
                p for p in valid_targets
                if player_identity_key(p) != player_identity_key(current_vote_target)
            ]
            if rally_targets and random.random() < 0.2:
                return f"{random.choice(rally_targets)} vote {current_vote_target} with me"

        prompt_groups = {
            1: [
                "{player1} Vote with me plz",
                "{player1} why are u acting so scummy lol",
                "{player1} are you town?",
                "{player1} give me reads",
                "{player1} im voting u in volo btw",
                "{player1} pls read",
                "{player1} get off",
                "{player1} whats your read on the game rn",
                "{player1} you've been quiet",
                "{player1} thoughts?",
                "{player1} who are you voting",
                "{player1} defend yourself",
                "{player1} why havent you voted yet",
                "{player1} whats your case",
                "{player1} elaborate on your reads",
                "whats up {player1}",
                "{player1} you good?",
                "yo {player1}",
                "{player1} sup",
            ],
            2: [
                "{player1} what do you think about {player2}?",
                "{player1} and {player2} what are your reads on each other?",
                "{player1} if {player2} is scum who do you think their partner is?",
                "{player1} what do you think of {player2}'s vote",
                "{player1} would you rather vote {player2} or someone else",
                "{player1} do you trust {player2}",
                "{player1} and {player2} sus me a read",
            ],
            3: [
                "{player1} {player2} and {player3} are the scumteam btw",
                "{player1} do you think {player2} and {player3} are paired?",
                "{player1} and {player2} what do u think of {player3}",
                "{player1} {player2} {player3} who's the odd one out here",
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

        # System/room announcements (roster lists, phase changes, etc.)
        # aren't something a "player" said -- don't parrot them as filler.
        if sender == "~":
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
        """Idle chatter for when nothing's happening, so the bot doesn't sit
        completely silent all day -- but sparingly, roughly once every 10
        minutes, not a running commentary. Each cycle either reacts to
        something someone said or asks a random question, never both, and
        often neither.

        Each message goes through _send_chat_message so it's paced like
        someone actually typing it.

        In silent_mode this does nothing at all -- voting/claiming still
        happen elsewhere, but the bot never volunteers unprompted chatter.
        """
        if self.config.gameplay.silent_mode:
            return

        while self.tracker.state == "DAY" and self.tracker.in_game and not self.tracker.eliminated:
            try:
                await asyncio.sleep(random.uniform(480.0, 720.0))
                if self.tracker.state != "DAY" or not self.tracker.in_game or self.tracker.eliminated:
                    break

                options = ["question", "none"]
                weights = [3, 2]
                if self._remembered_lines:
                    options.insert(0, "reaction")
                    weights.insert(0, 3)

                action = random.choices(options, weights=weights, k=1)[0]

                if action == "reaction":
                    remembered_line = random.choice(self._remembered_lines)
                    reaction = random.choice(REACTION_PHRASES)
                    quoted = f"{remembered_line} // {reaction}"
                    logger.info(f"Reacting to remembered line: {quoted}")
                    await self._send_chat_message(quoted)
                elif action == "question":
                    session = self.tracker.get_game_session()
                    question_prompt = self._build_question_prompt(session)
                    if question_prompt:
                        logger.info(f"Asking random question: {question_prompt}")
                        await self._send_chat_message(question_prompt)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in random actions loop: {e}")

    async def _delayed_first_evaluation(self):
        """Waits a bit into the day before the first vote evaluation, since
        voting the instant the day starts (before anyone's said anything)
        would just be acting on zero signal.
        """
        await asyncio.sleep(self.config.gameplay.first_evaluation_delay_seconds)
        if self.tracker.state == "DAY" and not self.tracker.eliminated:
            logger.info("Running first vote evaluation of the day.")
            await self._evaluate_and_vote()

    def _pick_random_vote_target(self, session, invert: bool = False) -> Optional[str]:
        """Picks a random target for an unsure/exploratory vote.

        The "alive" pool and the "randvote candidate" pool aren't the same
        thing: a player the model currently reads as fairly confident town
        (even if nobody cleared the bar to vote as a suspect) is excluded
        from random guessing, rather than being an equally-likely target as
        everyone else. Falls back to the full alive pool if that would
        leave no candidates at all (e.g. in a tiny/short game where the
        model has too little signal to read anyone as confidently town).

        `invert` flips this for themes like Modexe, where voting is
        actually a benefit (e.g. handing out a gun) -- there, a confident
        scum read should be excluded from random guessing instead, since
        randomly rewarding a suspected mafia player would be the mistake.
        """
        from ..io.player_names import names_match, player_identity_key

        alive_players = [p for p in session.players if p not in self.tracker.dead_players]
        valid_targets = [p for p in alive_players if not names_match(p, self.config.showdown.username)]
        if not valid_targets:
            return None

        predictions = self._get_strategy_full_predictions(session)
        if invert:
            excluded_keys = {
                player_identity_key(name)
                for name, prob_mafia in predictions
                if prob_mafia >= self.strategy.min_confidence
            }
        else:
            excluded_keys = {
                player_identity_key(name)
                for name, prob_mafia in predictions
                if (1.0 - prob_mafia) >= self.strategy.min_confidence
            }
        remaining_targets = [p for p in valid_targets if player_identity_key(p) not in excluded_keys]

        pool = remaining_targets or valid_targets
        return random.choice(pool)

    def _is_modexe_theme(self) -> bool:
        """"Modexe" isn't a real theme's on-the-record name -- it's a
        community nickname for "Modified Execution" (confirmed live: its
        actual description is "Vote someone to give them a kill... If the
        person the Mafia corrupted gets the gun, they die instead!"). It
        flips what a day vote does -- voting someone hands them a gun
        instead of lynching them -- so the bot should be voting who it
        trusts (town reads), not who it suspects.

        "Cult.exe" and "Mime.exe" are variants on the same mod.exe voting
        mechanic (confirmed live for Mime.exe: "mod.exe with Mimes! Mimes
        exit the game if they are 'voted out' (given the gun)") -- same
        vote-inversion applies.
        """
        theme = getattr(self.tracker, "theme", None)
        if not theme:
            return False
        theme_lower = theme.lower()
        return any(
            name in theme_lower
            for name in ("modexe", "modified execution", "mod.exe", "cult.exe", "mime.exe")
        )

    def _is_popcorn_theme(self) -> bool:
        """Popcorn hands one player a gun that replaces the day vote entirely
        (the gun-holder "shoots" instead of voting) -- only ever a Vanilla
        Townie, per the rules text.
        """
        theme = getattr(self.tracker, "theme", None)
        return bool(theme) and "popcorn" in theme.lower()

    @staticmethod
    def _sender_from_line(line: str) -> Optional[str]:
        parts = line.split("|")
        if len(parts) > 3 and parts[1] == "c:":
            return parts[3]
        if len(parts) > 2 and parts[1] == "c":
            return parts[2]
        return None

    def _check_gun_pickup(self, line: str) -> None:
        """Detects whether this bot now holds the Popcorn gun.

        Three tells, checked in order: a bolded "**{name} has gun**"
        announcement (unambiguous enough to accept from anyone), the host's
        plain-text "{name} has gun" chat announcement (confirmed live --
        not bolded -- but only trusted when the sender is confirmed to be
        the tracked host, since any player could type the same words as a
        joke or a guess), and the death/role-reveal broadcast ("{name}'s
        role was Vanilla Townie"), which theoretically fires when a shot
        that missed Mafia hands the gun to its Town survivor, though live
        play so far has only ever shown that death reflected in the
        votes-panel's Dead Players list, not as a chat broadcast -- kept as
        a fallback in case a differently-hosted game does broadcast it.
        """
        if self._has_gun or not self._is_popcorn_theme():
            return
        if not self._own_role or "vanilla townie" not in self._own_role.lower():
            return

        from ..io.player_names import names_match

        username = self.config.showdown.username
        message_text = self._message_text_from_line(line).strip()

        match = BOLD_HAS_GUN_RE.search(message_text)
        if match and names_match(match.group(1).strip(), username):
            self._has_gun = True
        else:
            match = PLAIN_HAS_GUN_RE.search(message_text)
            if match and names_match(match.group(1).strip(), username):
                host = getattr(self.tracker, "host", None)
                sender = self._sender_from_line(line)
                if host and sender and names_match(sender, host):
                    self._has_gun = True
            else:
                match = GUN_ROLE_REVEAL_RE.search(line)
                if match and "vanilla townie" in match.group(2).strip().lower() and names_match(match.group(1).strip(), username):
                    self._has_gun = True

        if self._has_gun:
            logger.info("Detected this bot now holds the Popcorn gun; will shoot scumreads instead of voting.")

    @staticmethod
    def _choose_idea_pick(options: list[tuple[str, str, Optional[str]]]) -> Optional[str]:
        """Picks a Town-aligned option out of an IDEA module's remaining
        choices, avoiding anything confirmed Mafia-aligned. Falls back to
        any not-confirmed-Mafia option (alignment unclear, e.g. a solo
        "aligned with yourself" role) if no Town option is offered, and
        gives up (returns None, leaving the pick to whatever the game does
        by default) only if every option is confirmed Mafia.
        """
        town_options = [opt for opt in options if (opt[2] or "").lower() == "town"]
        if town_options:
            return town_options[0][0]

        safe_options = [opt for opt in options if (opt[2] or "").lower() != "mafia"]
        if safe_options:
            return safe_options[0][0]

        return None

    async def _maybe_pick_idea_role(self, line: str) -> None:
        if getattr(self, "_idea_picked", False):
            return

        options = self.tracker.parse_idea_options_if_present(line)
        if not options:
            return

        choice = self._choose_idea_pick(options)
        if not choice:
            return

        self._idea_picked = True
        role_names = [name for _, name, _ in options]
        logger.info(f"IDEA pick: choosing role id '{choice}' from options {role_names}.")
        await self.send_room_command(f"/mafia ideapick role, {choice}")

    async def _maybe_react_to_clanker(self, line: str) -> None:
        """Troll mode only: gets offended at "clanker" and shifts its vote
        onto whoever said it a moment later. Off by default -- this is a
        for-fun gag, not something to leave on for regular play.
        """
        if not self.config.gameplay.troll_mode or self.config.gameplay.silent_mode:
            return

        sender = self._sender_from_line(line)
        if not sender:
            return

        from ..io.player_names import canonical_player_name, names_match

        # Showdown echoes our own outgoing messages back through the same
        # queue -- never react to ourselves.
        if names_match(sender, self.config.showdown.username):
            return

        message_text = self._message_text_from_line(line)
        if not CLANKER_RE.search(message_text):
            return

        target = canonical_player_name(sender)
        logger.info(f"Troll mode: {target} said 'clanker'; reacting offended.")
        await self._send_chat_message(random.choice(CLANKER_OFFENDED_LINES))

        await asyncio.sleep(1)

        if self.tracker.state == "DAY" and self.tracker.in_game and not self.tracker.eliminated:
            logger.info(f"Troll mode: shifting vote onto {target}.")
            await self.send_room_command(f"/mafia vote {target}")
            self._current_vote_target = target

    async def _evaluate_and_vote(self, allow_random_fallback: bool = False):
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

        is_modexe = self._is_modexe_theme()
        has_gun = getattr(self, "_has_gun", False) and self._is_popcorn_theme()

        if is_modexe:
            # Voting = handing someone a gun here, so the "vote decision"
            # is who we trust most, not who we suspect.
            logger.info("Modexe theme detected; voting a town read instead of a suspect.")
            target, confidence = self._get_strategy_town_read(session)
        else:
            volo_confidence = None
            if self._is_volo(session):
                volo_confidence = self.config.gameplay.volo_min_confidence
                logger.info(f"VoLo detected; raising vote confidence bar to {volo_confidence}.")

            target, confidence = self._get_strategy_vote(session, min_confidence=volo_confidence)

        is_random_guess = False

        if not target and allow_random_fallback:
            # Last evaluation of the day and still no confident read -- take
            # a random guess rather than sitting out the vote entirely.
            target = self._pick_random_vote_target(session, invert=is_modexe)
            is_random_guess = target is not None

        if target:
            if target != self._current_vote_target:
                if has_gun:
                    # Holding the Popcorn gun replaces the day vote entirely
                    # -- shoot the biggest scumread in chat instead of
                    # submitting a normal vote command.
                    logger.info(f"Popcorn gun held; shooting {target} (confidence: {confidence:.2%}). Previous: {self._current_vote_target}")
                    await self._send_chat_message(f"**shoot {target}**")
                elif is_random_guess:
                    logger.info(f"No confident read; randomly voting {target} at the last moment.")
                    await self.send_room_command(f"/mafia vote {target}")
                else:
                    logger.info(f"Decided to vote for {target} (confidence: {confidence:.2%}). Previous: {self._current_vote_target}")
                    await self._cast_vote_with_optional_comment(target)
                self._current_vote_target = target
            else:
                logger.info(f"Maintaining current {'shoot target' if has_gun else 'vote'} on {target} (confidence: {confidence:.2%})")
        else:
            if self._current_vote_target:
                if not has_gun:
                    logger.info("No clear target meets confidence threshold. Unvoting.")
                    await self.send_room_command("/mafia unvote")
                self._current_vote_target = None

        town_read, town_confidence = self._get_strategy_town_read(session)
        if town_read:
            if town_read != self._current_town_read:
                logger.info(f"Decided {town_read} is town (confidence: {town_confidence:.2%}). Previous: {self._current_town_read}")
                if not self.config.gameplay.silent_mode and random.random() < self.config.gameplay.town_read_comment_chance:
                    await self._send_chat_message(f"{town_read} is town")
                self._current_town_read = town_read
            else:
                logger.info(f"Maintaining current town read on {town_read} (confidence: {town_confidence:.2%})")
        else:
            self._current_town_read = None

    async def _cast_vote_with_optional_comment(self, target: str):
        """Casts a vote the way a person actually would: not always narrated,
        and when it is, not always glued to the vote in the same order.

        In silent_mode the vote itself always still happens -- only the
        narration is suppressed.
        """
        if self.config.gameplay.silent_mode or random.random() >= self.config.gameplay.vote_comment_chance:
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
        try:
            choice = await loop.run_in_executor(None, input, "Store this game to database? [y/N]: ")
        except EOFError:
            # No interactive stdin (e.g. running headless/backgrounded) --
            # there's no one to answer the prompt. auto_save_games decides
            # what happens then: save automatically (the point of running
            # unattended live tests is to actually grow the training set),
            # or discard, matching the old behavior, if curation matters
            # more than capturing every run.
            if self.config.gameplay.auto_save_games:
                logger.info("No interactive stdin available; auto-saving this completed game.")
                choice = "y"
            else:
                logger.warning(
                    "No interactive stdin available to confirm DB save, and auto_save_games is "
                    "off; discarding this completed game."
                )
                return
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
            try:
                answer = (await loop.run_in_executor(None, input, prompt)).strip().lower()
            except EOFError:
                # No interactive stdin -- the game itself is already saved
                # by this point, so just leave the remaining players
                # undefined (same as a blank/skipped answer) instead of
                # letting this look like the whole save failed.
                logger.info(
                    f"No interactive stdin available; leaving {row.player_name} (and any "
                    "remaining undefined players) unassigned."
                )
                return
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

        # Command parsing (uses a "." prefix, not "!" -- "!" is reserved for
        # staff broadcast commands on Showdown and gets intercepted):
        # .vote [player] -> cast a vote for player right now (one-time action)
        # .unvote -> clear the bot's current vote, if it has one
        # .multiplier [player] [value] -> suspicion multiplier
        # .reset -> clear suspicion multipliers
        # .reads -> full ranked mafia-probability list for every live player
        # .claim -> this game's claim message
        parts = msg.strip().split()
        if not parts:
            return

        cmd = parts[0].lower()
        if cmd == ".claim":
            if self.tracker.in_game and not self.tracker.eliminated:
                claim_message = self._get_claim_message()
                if claim_message:
                    session = self.tracker.get_game_session()
                    if self._is_at_v1(session):
                        claim_message = f"{claim_message} 1 to hammer"
                    logger.info(f"Responding to claim PM from {clean_sender} with {claim_message}")
                    await self.connection.send(f"|/pm {clean_sender}, {claim_message}")
                else:
                    logger.info(f"No claim message available for PM from {clean_sender}")
                    await self.connection.send(f"|/pm {clean_sender}, unknown")
            return

        if cmd == ".reads":
            # session.players can include someone who was added mid-game and
            # later left/was removed by the host through wording we don't
            # have a regex for (e.g. "X has been added to the game by Y",
            # or simply disconnecting) -- rather than trying to special-case
            # every possible add/remove phrasing, cross-check against the
            # room's own live vote-roster (refreshed right before replying),
            # which reflects who can actually still be voted on right now.
            await self.send_room_command("/mafia votes")
            await asyncio.sleep(1.5)

            session = self.tracker.get_game_session()
            predictions = self._get_strategy_full_predictions(session)
            live_counts = getattr(self.tracker, "live_vote_counts", None)
            if live_counts:
                from ..io.player_names import player_identity_key
                live_keys = {player_identity_key(name) for name in live_counts}
                predictions = [(name, prob) for name, prob in predictions if player_identity_key(name) in live_keys]

            await self.connection.send(f"|/pm {clean_sender}, {self._format_reads_message(predictions)}")
            return

        if cmd == ".vote" and len(parts) > 1:
            from ..io.player_names import player_identity_key

            # Player names can contain spaces (e.g. "I give u pile alt") --
            # take everything after the command itself, not just the first
            # whitespace-delimited token, or a multi-word name never matches.
            target_input = " ".join(parts[1:])
            target_key = player_identity_key(target_input)

            # "No Vote" is a real, always-available room option (its own
            # button in the votes list) -- not a player, so it would never
            # match against self.tracker.players.
            if target_key == "novote":
                await self.send_room_command("/mafia vote novote")
                self._current_vote_target = None
                await self.connection.send(f"|/pm {clean_sender}, Voted No Vote.")
                return

            real_target = next((p for p in self.tracker.players if player_identity_key(p) == target_key), None)
            if not real_target:
                await self.connection.send(f"|/pm {clean_sender}, {target_input} is not a real player.")
                return
            await self.send_room_command(f"/mafia vote {real_target}")
            self._current_vote_target = real_target
            await self.connection.send(f"|/pm {clean_sender}, Voted {real_target}.")

        if cmd == ".unvote":
            if not self._current_vote_target:
                await self.connection.send(f"|/pm {clean_sender}, not voting anyone")
                return
            await self.send_room_command("/mafia unvote")
            self._current_vote_target = None
            await self.connection.send(f"|/pm {clean_sender}, Unvoted.")
            return

        elif cmd == ".multiplier" and len(parts) > 2:
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

        elif cmd == ".reset":
            self.strategy.reset()
            await self.connection.send(f"|/pm {clean_sender}, Reset bot override states.")
            if self.tracker.state == "DAY":
                await self._evaluate_and_vote()
