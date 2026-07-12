import logging
import re
from typing import List, Set, Optional, Tuple

from ..data.flips import extract_flips
from ..data.models import GameSession
from ..io.parser import (
    parse_showdown_log,
    split_combined_system_lines,
    PLAYERS_LIST_RE,
    DAY_MARKER_RE,
    ELIMINATION_RE,
    REVEAL_RE,
)
from ..io.player_names import canonical_player_name, player_identity_key

logger = logging.getLogger("mafia_bot.tracker")

# Regex to detect game endings
GAME_END_RE = re.compile(
    r"(mafia\s+game\s+has\s+ended|the\s+town\s+has\s+won|the\s+mafia\s+has\s+won|the\s+neutral\s+has\s+won|game\s+over|game\s+ended)",
    re.IGNORECASE
)

# Regex to detect signup phase start
SIGNUPS_START_RE = re.compile(
    r"(A\s+(?:new\s+)?game\s+of\s+mafia\s+(?:was|has\s+been|is)\s+(?:created|started)|mafia\s+signups\s+have\s+begun|\*\*Players\s*\(\d+\)\*\*:\s*$|Join the game|new\s+mafia\s+game)",
    re.IGNORECASE
)

GAME_ENDED_RE = re.compile(
    r"The\s+game\s+of\s+Mafia\s+has\s+ended",
    re.IGNORECASE,
)

NO_GAME_RE = re.compile(
    r"This\s+command\s+requires\s+a\s+game\s+of\s+Mafia\s*\(this\s+room\s+has\s+no\s+game\)",
    re.IGNORECASE,
)

SPECTATE_STATE_RE = re.compile(
    r"A\s+game\s+of\s+Mafia\s+is\s+in\s+progress.*Become\s+a\s+substitute.*Spectate\s+the\s+game",
    re.IGNORECASE,
)

# Regex to detect night phase start. Some hosts/rulesets phrase this as
# "Night 2. Submit whether you are using an action or idle..." (confirmed
# live) or "It's night in the game of Mafia! Send in an action or idle."
# instead of "Night X has begun" -- both are matched too, or a night phase
# would silently never register (and the bot would never submit its idle
# or night action) for those hosts.
NIGHT_START_RE = re.compile(
    r"(night\s+(?P<night>\d+)\s+has\s+begun|night\s+(?P<night_num>\d+)\s+starts|day\s+\d+\s+has\s+ended|day\s+\d+\s+ended|night\s+\d+\.\s*submit|it'?s\s+night\s+in\s+the\s+game\s+of\s+mafia)",
    re.IGNORECASE
)

# Regex to extract the current hammer count from a day-marker message
# (e.g. "Day 5. The hammer count is set at 3")
HAMMER_COUNT_RE = re.compile(r"hammer\s+count\s+is\s+set\s+at\s+(?P<hammer>\d+)", re.IGNORECASE)

# The `/mafia votes` reply is a single raw-HTML blob shaped like:
#   Votes (Hammer: 2)
#   2* A Flowers Dream (Brady1014, mist)
#   1 Brady1014 (A Flowers Dream)
# ("*" just flags the current plurality leader -- not parsed, not needed).
VOTES_HEADER_RE = re.compile(r"votes\s*\(hammer:\s*(?P<hammer>\d+)\)", re.IGNORECASE)
VOTES_LINE_RE = re.compile(r"^(?P<count>\d+)\*?\s+(?P<target>.+?)\s+\((?P<voters>.+?)\)\s*$", re.IGNORECASE)

# The `/mafia originalrolelist` reply, e.g. "Original Rolelist: mafia, ic, vt"
# -- a flat, comma-separated list of role tokens for the whole game as it
# started (not who has which role, just the overall composition), used to
# work out the mafia-vs-total split for VoLo/parity tracking.
ORIGINAL_ROLELIST_RE = re.compile(r"original\s+rolelist:\s*(?P<roles>.+)", re.IGNORECASE)

# The room's page panel (a `|pagehtml|` update, e.g. from viewing/refreshing
# the votes page) includes a line like:
#   <span style="font-weight:bold;">Theme</span>: CCTV
# This panel arrives under its own pseudo-room (e.g. "view-mafia-mafia"),
# not the main chat room, so it's parsed independently of the normal
# room-gated message flow rather than through process_message.
THEME_RE = re.compile(r"Theme</span>:\s*([^<]+)", re.IGNORECASE)

# Same panel also includes the host's name, e.g. <h3>Host: ghostlyplanets</h3>
HOST_RE = re.compile(r"Host:\s*([^<]+)</h3>", re.IGNORECASE)

# For Mafia-aligned roles, the same panel lists co-faction teammates, e.g.
# <p><span style="font-weight:bold">Partners</span>: Trimmerz</p> -- a
# comma-separated list for bigger factions. Needed so a random night kill
# never targets the bot's own partner.
PARTNERS_RE = re.compile(r"Partners</span>:\s*([^<]+)", re.IGNORECASE)

# During an IDEA module, the same panel gets an "IDEA information" section
# with one clickable button per still-available role option, e.g.
#   <button class="button" name="send" value="/msgroom mafia,/mafia ideapick
#   role, mafiaoneshotstrongman">Mafia One-Shot Strongman</button>
# The already-picked option (and "clear" before any pick is made) render as
# `class="button disabled"` with no `name="send"`/`value`, so this pattern
# naturally only matches options that are still choosable. The "clear"
# button itself (once enabled, after a pick) has an empty role id after the
# comma, filtered out by the caller.
IDEA_OPTION_RE = re.compile(
    r'<button class="button" name="send" value="/msgroom mafia,/mafia ideapick role,\s*([^"]*)">([^<]+)</button>'
)

# The panel's "Role details" section spells out each option's alignment,
# e.g. "...You are aligned with the <span ...>Town</span>." -- paired with
# IDEA_OPTION_RE by matching role name to `<summary>{name}</summary>`.
IDEA_ROLE_ALIGNMENT_RE = re.compile(
    r"<summary>([^<]+)</summary>.*?aligned with (?:the\s+)?<span[^>]*>([^<]+)</span>",
    re.DOTALL,
)

# The room automatically posts these as the day's deadline approaches, e.g.
# "**3 minutes left!**" / "**1 minute left!**"
THREE_MIN_LEFT_RE = re.compile(r"3\s*minutes?\s*left", re.IGNORECASE)
ONE_MIN_LEFT_RE = re.compile(r"1\s*minute\s*left", re.IGNORECASE)

# Player substitutions, e.g. "Blue flare fusion has been subbed out." /
# "mist has joined the game." -- these are anchored (^...$) and matched
# per-line after splitting combined announcements, not searched anywhere in
# the text, to avoid the same non-greedy over-capture problem eliminations had.
SUB_OUT_RE = re.compile(r"^(?P<player>.+?)\s+has\s+been\s+subbed\s+out\.?$", re.IGNORECASE)
SUB_IN_RE = re.compile(r"^(?P<player>.+?)\s+has\s+joined\s+the\s+game\.?$", re.IGNORECASE)

class GameTracker:
    def __init__(self):
        self.state = "IDLE"  # IDLE, SIGNUPS, DAY, NIGHT
        self.players: List[str] = []
        self.accumulated_lines: List[str] = []
        self.current_day = 1
        self.raw_text_history: List[str] = []
        self.in_game = False
        self.eliminated = False
        self.dead_players: Set[str] = set()
        self.bot_username: Optional[str] = None
        self.hammer_count: Optional[int] = None
        self.live_vote_counts: dict = {}
        self.original_role_tokens: List[str] = []
        self.theme: Optional[str] = None
        self.host: Optional[str] = None
        self.partners: List[str] = []
        self.deadline_warning: Optional[str] = None  # None, "3_minutes", or "1_minute"
        self._pending_sub_out: Optional[str] = None

    @staticmethod
    def _normalize_message_text(line: str) -> str:
        parts = line.split("|")
        username: Optional[str] = None
        if len(parts) > 3 and parts[1] == "c:":
            username = parts[3]
            message_text = "|".join(parts[4:])
        elif len(parts) > 2 and parts[1] == "c":
            username = parts[2]
            message_text = "|".join(parts[3:])
        elif len(parts) > 2 and parts[1] == "raw":
            message_text = parts[2]
        else:
            message_text = line

        message_text = re.sub(r"<[^>]*>", "", message_text).strip()
        # Real chat lines need "username: message" for downstream parsing
        # (CHAT_LINE_RE) to attribute the message to a player. Server/system
        # broadcasts (posted as "~") are left bare, matching the format the
        # elimination/reveal/vote regexes expect.
        if username and username != "~":
            message_text = f"{username}: {message_text}"
        return message_text

    def reset(self):
        logger.info("Resetting game tracker state.")
        self.state = "IDLE"
        self.players = []
        self.accumulated_lines = []
        # _prune_dead_players re-scans this on every call to find eliminations
        # -- if it isn't cleared here, a stale "X was eliminated!" line from a
        # previous completed game keeps getting re-detected in every future
        # game (and if X was ever the bot itself, a brand new game would get
        # silently marked as already-eliminated for the bot).
        self.raw_text_history = []
        self.current_day = 1
        self.in_game = False
        self.eliminated = False
        self.dead_players = set()
        self.hammer_count = None
        self.live_vote_counts = {}
        self.original_role_tokens = []
        self.theme = None
        self.host = None
        self.partners = []
        self.deadline_warning = None
        self._pending_sub_out = None

    def process_message(self, line: str, bot_username: Optional[str] = None) -> Optional[str]:
        """
        Processes a single chat/system message line from the target room.
        Returns:
            - "SIGNUPS" if signup phase started.
            - "STARTED" if game started (roster announced).
            - "DAY" if day phase started.
            - "NIGHT" if night phase started.
            - "FINISHED" if the game ended.
            - "VOTES_UPDATE" if a `/mafia votes` reply was parsed.
            - None otherwise.
        """
        if bot_username:
            self.bot_username = bot_username

        # Save raw line in global history
        self.raw_text_history.append(line)

        # Determine whether this line actually came from the room/server
        # ("~") rather than a real player, so phase/state transitions can
        # only ever be driven by genuine system announcements -- never by a
        # player simply typing something that happens to resemble one (e.g.
        # "lol night 2 has begun already" or "the town has won this fr").
        parts = line.split("|")
        if len(parts) > 3 and parts[1] == "c:":
            is_system_message = parts[3] == "~"
        elif len(parts) > 2 and parts[1] == "c":
            is_system_message = parts[2] == "~"
        else:
            is_system_message = True

        # Parse contents of the message
        clean_text = self._normalize_message_text(line)

        if not is_system_message:
            # Still accumulate for later message-content parsing (tells,
            # votes typed in chat, etc.), but never let player chat drive
            # phase/state transitions.
            if self.state in {"DAY", "NIGHT"}:
                self.accumulated_lines.append(line)
            return None

        if self._parse_votes_response(line):
            return "VOTES_UPDATE"

        if self._parse_original_rolelist(line):
            return None

        # If we are in IDLE or SIGNUPS, check for signup start
        if self.state == "IDLE" and SIGNUPS_START_RE.search(clean_text):
            self.state = "SIGNUPS"
            self.in_game = False
            self.eliminated = False
            logger.info("Detected Mafia Signups phase.")
            return "SIGNUPS"

        if GAME_ENDED_RE.search(clean_text):
            if self.state == "IDLE":
                # Already handled -- the room can send more than one
                # "game has ended"-shaped message for the same finish (e.g.
                # a win announcement followed by a separate wrap-up
                # message), and this check runs unconditionally on every
                # message. Without this guard each one re-fired FINISHED,
                # causing "gg" and the ragebait line to send multiple times
                # for a single game.
                return None
            self.state = "IDLE"
            self.in_game = False
            self.eliminated = False
            logger.info("Detected mafia game end message; resetting tracker state.")
            return "FINISHED"

        if NO_GAME_RE.search(clean_text):
            self.state = "IDLE"
            self.in_game = False
            self.eliminated = False
            self.players = []
            self.accumulated_lines = []
            logger.info("Detected no-active-mafia-game error; resetting tracker state.")
            return None

        if SPECTATE_STATE_RE.search(clean_text):
            # This "in progress / become a substitute / spectate" broadcast
            # is sent generically to anyone joining or refreshing the room
            # while a game is active -- including actual participants, not
            # just spectators. It doesn't reliably mean "we specifically are
            # not playing", so it must never override an in_game=True that
            # was already established from an authoritative roster match
            # (this was silently breaking every subsequent action -- vote
            # reactions, claims, night actions -- for a game the bot was
            # genuinely still playing in).
            if not self.in_game:
                self.eliminated = False
                logger.info("Detected spectate-only mafia state; bot is not participating.")
            return None

        # Check for player roster list (signifies game start / active players)
        players_match = PLAYERS_LIST_RE.search(line)
        if players_match:
            # Re-parse players list using the existing helper
            from ..io.parser import parse_players_list
            parsed_players = parse_players_list(line)
            if parsed_players:
                if parsed_players == self.players and self.state in {"DAY", "NIGHT"}:
                    # The room sometimes re-broadcasts the exact same roster
                    # line twice around game start (e.g. once at roster
                    # lock, once when the game actually begins) -- treating
                    # a byte-identical repeat as a brand new game would
                    # re-fire every per-game reset (claimed_this_day,
                    # current vote/town read, remembered lines) mid-game.
                    self.accumulated_lines.append(line)
                    return None
                self.players = parsed_players
                # Roster lock doesn't mean Day 1 has actually begun -- there's
                # a rolling/role-distribution period (often shown as "Night
                # 0") before the real "Day 1. The hammer count is set at..."
                # marker arrives. Treating that gap as already "DAY" made the
                # bot start chatting (filler, reactions, questions) before
                # the game had actually started. NIGHT is a closer fit --
                # nothing here talks during NIGHT -- and the real DAY_MARKER_RE
                # match below flips it to DAY for real once Day 1 starts.
                self.state = "NIGHT"
                self.accumulated_lines.append(line)
                self.in_game = self._is_bot_on_roster(bot_username)
                self.eliminated = False
                logger.info(f"Detected game start with players: {self.players} (in_game={self.in_game})")
                return "STARTED"

        # If a game is active (not IDLE or SIGNUPS)
        if self.state in {"DAY", "NIGHT"}:
            self.accumulated_lines.append(line)

            self._handle_substitutions(clean_text, bot_username)

            if ELIMINATION_RE.search(clean_text):
                self._prune_dead_players(bot_username=bot_username)

            # Check for the room's automatic deadline warnings first, used to
            # trigger re-evaluation at meaningful points in the day rather
            # than on a fixed timer. These must be checked before the day
            # marker below, since its generic "**"-prefixed pattern would
            # otherwise swallow them (e.g. "**3 minutes left!**").
            if self.state == "DAY":
                if ONE_MIN_LEFT_RE.search(clean_text):
                    already_warned = self.deadline_warning == "1_minute"
                    self.deadline_warning = "1_minute"
                    if already_warned:
                        return None
                    logger.info("Deadline warning: 1 minute left.")
                    return "DEADLINE_1MIN"
                if THREE_MIN_LEFT_RE.search(clean_text):
                    already_warned = self.deadline_warning == "3_minutes"
                    self.deadline_warning = "3_minutes"
                    if already_warned:
                        return None
                    logger.info("Deadline warning: 3 minutes left.")
                    return "DEADLINE_3MIN"

            # Check for day marker
            day_match = DAY_MARKER_RE.search(clean_text)
            if day_match:
                day_text = day_match.group("day")
                new_day = int(day_text) if day_text else self.current_day
                if self.state == "DAY" and new_day == self.current_day:
                    # DAY_MARKER_RE also matches generic decorative
                    # separators (---, ***, etc.) with no requirement to
                    # actually say "day"/"hammer", so an unrelated system
                    # message (e.g. a reveal announcement that happens to
                    # include a divider) can accidentally match this and
                    # re-fire an already-active day, cascading into
                    # redundant vote re-evaluations and duplicated
                    # random-actions tasks.
                    return None
                self.state = "DAY"
                self.current_day = new_day
                hammer_match = HAMMER_COUNT_RE.search(clean_text)
                if hammer_match:
                    self.hammer_count = int(hammer_match.group("hammer"))
                self.deadline_warning = None
                self._prune_dead_players()
                logger.info(f"Phase change: Day {self.current_day}")
                return "DAY"

            # Check for night marker
            if NIGHT_START_RE.search(clean_text):
                self.state = "NIGHT"
                logger.info("Phase change: Night started.")
                return "NIGHT"

            # Check for game end
            if GAME_END_RE.search(clean_text):
                logger.info("Game completion detected.")
                self.state = "IDLE"
                return "FINISHED"

        return None

    def _parse_votes_response(self, line: str) -> bool:
        """Parses the room's `/mafia votes` reply into live_vote_counts +
        hammer_count. The reply arrives as one raw-HTML blob with <br>-style
        line breaks rather than one message per row, so those have to be
        restored to real newlines before the per-row regex can run.
        """
        parts = line.split("|")
        if len(parts) > 3 and parts[1] == "c:":
            payload = "|".join(parts[4:])
        elif len(parts) > 2 and parts[1] == "c":
            payload = "|".join(parts[3:])
        elif len(parts) > 2 and parts[1] == "raw":
            payload = "|".join(parts[2:])
        else:
            payload = line

        payload = re.sub(r"<br\s*/?>", "\n", payload, flags=re.IGNORECASE)
        payload = re.sub(r"<[^>]*>", "", payload)

        header_match = VOTES_HEADER_RE.search(payload)
        if not header_match:
            return False

        self.hammer_count = int(header_match.group("hammer"))
        counts: dict = {}
        for text_line in payload.splitlines():
            row_match = VOTES_LINE_RE.match(text_line.strip())
            if not row_match:
                continue
            target = canonical_player_name(row_match.group("target").strip())
            # The reply also lists a "No Vote" bucket for idling players --
            # that's not a real target, and including it would make the
            # non-voting count masquerade as the plurality leader.
            if target and player_identity_key(target) != "novote":
                counts[target] = int(row_match.group("count"))

        self.live_vote_counts = counts
        logger.info(f"Parsed /mafia votes response: hammer={self.hammer_count}, counts={self.live_vote_counts}")
        return True

    def _parse_original_rolelist(self, line: str) -> bool:
        """Parses the room's `/mafia originalrolelist` reply into a flat
        list of role tokens (e.g. "mafia, ic, vt" -> ["mafia", "ic", "vt"]),
        so the mafia-vs-total split can be computed later without needing to
        know which living player holds which role.
        """
        parts = line.split("|")
        if len(parts) > 3 and parts[1] == "c:":
            payload = "|".join(parts[4:])
        elif len(parts) > 2 and parts[1] == "c":
            payload = "|".join(parts[3:])
        elif len(parts) > 2 and parts[1] == "raw":
            payload = "|".join(parts[2:])
        else:
            payload = line

        payload = re.sub(r"<br\s*/?>", "\n", payload, flags=re.IGNORECASE)
        payload = re.sub(r"<[^>]*>", "", payload)

        match = ORIGINAL_ROLELIST_RE.search(payload)
        if not match:
            return False

        tokens = [token.strip().lower() for token in match.group("roles").split(",") if token.strip()]
        self.original_role_tokens = tokens
        logger.info(f"Parsed original rolelist: {tokens}")
        return True

    def parse_theme_if_present(self, line: str) -> bool:
        """Extracts the game's theme name from a `/mafia votes` page-panel
        dump (a `|pagehtml|` update). Called independently of
        process_message, since this panel arrives under its own pseudo-room
        rather than the main chat room.
        """
        match = THEME_RE.search(line)
        if not match:
            return False

        theme = match.group(1).strip()
        if theme and theme != self.theme:
            self.theme = theme
            logger.info(f"Detected game theme: {theme}")
        return True

    def parse_host_if_present(self, line: str) -> bool:
        """Extracts the game host's name from the same page-panel dump that
        carries the theme (see parse_theme_if_present).
        """
        match = HOST_RE.search(line)
        if not match:
            return False

        host = match.group(1).strip()
        if host and host != self.host:
            self.host = host
            logger.info(f"Detected game host: {host}")
        return True

    def parse_partners_if_present(self, line: str) -> bool:
        """Extracts co-faction teammate names from the same page-panel dump
        (see parse_theme_if_present) -- only present for Mafia-aligned
        roles. Needed so a random night kill never targets a partner.
        """
        match = PARTNERS_RE.search(line)
        if not match:
            return False

        partners = [name.strip() for name in match.group(1).split(",") if name.strip()]
        if partners and partners != self.partners:
            self.partners = partners
            logger.info(f"Detected mafia partners: {partners}")
        return True

    def parse_idea_options_if_present(self, line: str) -> List[Tuple[str, str, Optional[str]]]:
        """Extracts the still-choosable IDEA options from the same
        page-panel dump (see parse_theme_if_present), each as
        (role_id, role_name, alignment_or_None) -- alignment is None when
        the panel's role-details text doesn't spell it out as a plain
        Town/Mafia/etc. span (e.g. solo-flavoured "aligned with yourself"
        wincon text).
        """
        alignments = {
            name.strip(): alignment.strip()
            for name, alignment in IDEA_ROLE_ALIGNMENT_RE.findall(line)
        }

        options = []
        for role_id, role_name in IDEA_OPTION_RE.findall(line):
            role_id = role_id.strip()
            role_name = role_name.strip()
            if not role_id or role_name.lower() == "clear":
                continue
            options.append((role_id, role_name, alignments.get(role_name)))
        return options

    def _prune_dead_players(self, bot_username: Optional[str] = None) -> None:
        from ..io.parser import ELIMINATION_RE

        effective_bot_username = bot_username or self.bot_username
        eliminated = set()
        bot_was_eliminated = False
        bot_key = player_identity_key(effective_bot_username) if effective_bot_username else None
        for line in self.raw_text_history:
            normalized_line = self._normalize_message_text(line)
            match = ELIMINATION_RE.search(normalized_line)
            if match:
                player = canonical_player_name(match.group("player"))
                if player:
                    eliminated.add(player)
                    if bot_key and player_identity_key(player) == bot_key:
                        bot_was_eliminated = True

        if eliminated:
            self.dead_players.update(eliminated)
            self.players = [player for player in self.players if player not in self.dead_players]
            # Elimination is a one-way latch within a game: once we've seen
            # ourselves eliminated, never let a later call (e.g. one made
            # without bot_username) flip it back to False.
            self.eliminated = self.eliminated or bot_was_eliminated
            if self.eliminated:
                self.in_game = False
            logger.info(f"Removed dead players from active roster: {sorted(eliminated)}")

    def _handle_substitutions(self, clean_text: str, bot_username: Optional[str] = None) -> None:
        """Detects "X has been subbed out." / "Y has joined the game." and
        replaces X with Y in the active roster, preserving their slot. A sub
        is a replacement, not an elimination -- the incoming player takes
        over, they don't get marked dead.
        """
        effective_bot_username = bot_username or self.bot_username
        bot_key = player_identity_key(effective_bot_username) if effective_bot_username else None

        for segment in split_combined_system_lines(clean_text).splitlines():
            segment = segment.strip()
            if not segment:
                continue

            sub_out_match = SUB_OUT_RE.match(segment)
            if sub_out_match:
                self._pending_sub_out = canonical_player_name(sub_out_match.group("player"))
                continue

            sub_in_match = SUB_IN_RE.match(segment)
            if sub_in_match and self._pending_sub_out:
                new_player = canonical_player_name(sub_in_match.group("player"))
                old_key = player_identity_key(self._pending_sub_out)
                new_key = player_identity_key(new_player)
                self.players = [new_player if player_identity_key(p) == old_key else p for p in self.players]
                logger.info(f"Player sub: {self._pending_sub_out} -> {new_player}")
                if bot_key and old_key == bot_key:
                    self.in_game = False
                if bot_key and new_key == bot_key:
                    self.in_game = True
                self._pending_sub_out = None

    def _is_bot_on_roster(self, bot_username: Optional[str]) -> bool:
        if not bot_username:
            return False
        bot_key = player_identity_key(bot_username)
        if not bot_key:
            return False
        return any(player_identity_key(player) == bot_key for player in self.players)

    def get_game_session(self) -> GameSession:
        """Constructs a standard GameSession from accumulated lines."""
        normalized_lines = [self._normalize_message_text(line) for line in self.accumulated_lines]
        raw_text = "\n".join(normalized_lines)
        session = parse_showdown_log(raw_text, source="live_bot")
        session.flips = extract_flips(raw_text)
        # Ensure players are loaded
        if not session.players:
            session.players = list(self.players)
        return session
