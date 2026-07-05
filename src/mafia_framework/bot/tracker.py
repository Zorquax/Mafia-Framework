import logging
import re
from typing import List, Set, Optional

from ..data.flips import extract_flips
from ..data.models import GameSession
from ..io.parser import parse_showdown_log, PLAYERS_LIST_RE, DAY_MARKER_RE, ELIMINATION_RE, REVEAL_RE
from ..io.player_names import player_identity_key

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

# Regex to detect night phase start
NIGHT_START_RE = re.compile(
    r"(night\s+(?P<night>\d+)\s+has\s+begun|night\s+(?P<night_num>\d+)\s+starts|day\s+\d+\s+has\s+ended|day\s+\d+\s+ended)",
    re.IGNORECASE
)

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
        self.current_day = 1
        self.in_game = False
        self.eliminated = False
        self.dead_players = set()

    def process_message(self, line: str, bot_username: Optional[str] = None) -> Optional[str]:
        """
        Processes a single chat/system message line from the target room.
        Returns:
            - "SIGNUPS" if signup phase started.
            - "STARTED" if game started (roster announced).
            - "DAY" if day phase started.
            - "NIGHT" if night phase started.
            - "FINISHED" if the game ended.
            - None otherwise.
        """
        # Save raw line in global history
        self.raw_text_history.append(line)

        # Parse contents of the message
        clean_text = self._normalize_message_text(line)

        # If we are in IDLE or SIGNUPS, check for signup start
        if self.state == "IDLE" and SIGNUPS_START_RE.search(clean_text):
            self.state = "SIGNUPS"
            self.in_game = False
            self.eliminated = False
            logger.info("Detected Mafia Signups phase.")
            return "SIGNUPS"

        if GAME_ENDED_RE.search(clean_text):
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
            self.in_game = False
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
                self.players = parsed_players
                self.state = "DAY"
                self.accumulated_lines.append(line)
                self.in_game = self._is_bot_on_roster(bot_username)
                self.eliminated = False
                logger.info(f"Detected game start with players: {self.players} (in_game={self.in_game})")
                return "STARTED"

        # If a game is active (not IDLE or SIGNUPS)
        if self.state in {"DAY", "NIGHT"}:
            self.accumulated_lines.append(line)

            if ELIMINATION_RE.search(clean_text):
                self._prune_dead_players(bot_username=bot_username)

            # Check for day marker
            day_match = DAY_MARKER_RE.search(clean_text)
            if day_match:
                self.state = "DAY"
                day_text = day_match.group("day")
                if day_text:
                    self.current_day = int(day_text)
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

    def _prune_dead_players(self, bot_username: Optional[str] = None) -> None:
        from ..io.parser import ELIMINATION_RE

        eliminated = set()
        bot_was_eliminated = False
        bot_key = player_identity_key(bot_username) if bot_username else None
        for line in self.raw_text_history:
            normalized_line = self._normalize_message_text(line)
            match = ELIMINATION_RE.search(normalized_line)
            if match:
                player = match.group("player").strip()
                if player:
                    eliminated.add(player)
                    if bot_key and player_identity_key(player) == bot_key:
                        bot_was_eliminated = True

        if eliminated:
            self.dead_players.update(eliminated)
            self.players = [player for player in self.players if player not in self.dead_players]
            self.eliminated = bot_was_eliminated
            if self.eliminated:
                self.in_game = False
            logger.info(f"Removed dead players from active roster: {sorted(eliminated)}")

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
