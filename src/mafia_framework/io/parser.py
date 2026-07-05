import re
from datetime import datetime

from ..data.models import GameSession, LogEvent, Message, Vote
from .player_names import (
    canonical_player_name,
    names_match,
    normalize_player_name,
    player_identity_key,
    resolve_to_roster,
)

CHAT_LINE_RE = re.compile(r"^(?:\[(?P<timestamp>[^\]]+)\]\s*)?(?P<username>[^:]+):\s*(?P<message>.+)$")
VOTE_TEXT_RE = re.compile(r"^(?:vote|voted\s+for|->|→)\s+(?P<target>.+)$", re.IGNORECASE)
UNVOTE_TEXT_RE = re.compile(r"^unvote(?:d)?(?:\s+for)?(?:\s+(?P<target>.+))?$", re.IGNORECASE)
SYSTEM_VOTE_RE = re.compile(
    r"^(?:\[[^\]]+\]\s*)?(?:\|c:(?:\|[^|]*)*\|~\|)?(?P<voter>.+?) has voted (?P<target>.+?)(?:\.|$)",
    re.IGNORECASE,
)
SYSTEM_UNVOTE_RE = re.compile(
    r"^(?:\[[^\]]+\]\s*)?(?:\|c:(?:\|[^|]*)*\|~\|)?(?P<voter>.+?) has unvoted(?:\.|$)",
    re.IGNORECASE,
)
PLAYERS_LIST_RE = re.compile(
    r"^(?:\[[^\]]+\]\s*)?(?:\|c:(?:\|[^|]*)*\|~\|)?\*\*Players.*?\*\*:\s*(?P<players>.+)$",
    re.IGNORECASE,
)
DAY_MARKER_RE = re.compile(
    r"^(?:---+|\*\*+|===+|\+\++|day\s+(?P<day>\d+).*?hammer)",
    re.IGNORECASE | re.MULTILINE,
)
ELIMINATION_RE = re.compile(
    r"^(?:\[(?P<timestamp>[^\]]+)\]\s*)?(?:\|c:(?:\|[^|]*)*\|~\|)?(?P<player>[^\n]+?)\s+was eliminated!",
    re.IGNORECASE
)
REVEAL_RE = re.compile(
    r"^(?:\[(?P<timestamp>[^\]]+)\]\s*)?(?:\|c:(?:\|[^|]*)*\|~\|)?(?P<player>[^\n]+?)(?:['’]\s*s)?\s+role was\s+(?P<role>[^\n\.]+)",
    re.IGNORECASE
)

# Some sources (live protocol messages and some game-log exports) bundle an
# elimination notice and the immediate role reveal into one sentence, e.g.
# "X was eliminated! X's role was Y." Without a line break between the two
# clauses, FLIP_RE/REVEAL_RE's non-greedy player capture backtracks across the
# whole first clause, producing a mangled player name. Force a line break
# after the fixed "was eliminated!" phrase before any line-anchored parsing.
COMBINED_ELIMINATION_RE = re.compile(r"(was eliminated!)\s+(?=\S)", re.IGNORECASE)


def split_combined_system_lines(text: str) -> str:
    return COMBINED_ELIMINATION_RE.sub(r"\1\n", text)


def parse_timestamp(timestamp_text: str | None) -> str | None:
    if not timestamp_text:
        return None
    try:
        parsed = datetime.strptime(timestamp_text.strip(), "%H:%M")
        return parsed.strftime("%H:%M")
    except ValueError:
        return timestamp_text.strip()


def parse_players_list(player_line: str) -> list[str]:
    players = []
    match = PLAYERS_LIST_RE.search(player_line)
    if not match:
        return players
    raw_players = match.group("players")
    for name in raw_players.split(","):
        normalized = canonical_player_name(name)
        if normalized:
            players.append(normalized)
    return players


def _register_roster_names(names: list[str], identity_to_canonical: dict[str, str]) -> list[str]:
    roster: list[str] = []
    seen_keys: set[str] = set()
    for name in names:
        canonical = resolve_to_roster(name, identity_to_canonical)
        key = player_identity_key(canonical)
        if key and key not in seen_keys:
            seen_keys.add(key)
            roster.append(canonical)
    return roster


def _player_allowed(name: str, roster_keys: set[str], identity_to_canonical: dict[str, str]) -> bool:
    if not roster_keys:
        return True
    key = player_identity_key(name)
    return bool(key) and key in roster_keys


def _current_vote_target(voter: str, active_votes: dict[str, str]) -> str | None:
    key = player_identity_key(voter)
    return active_votes.get(key)


def _set_vote(voter: str, target: str, active_votes: dict[str, str]) -> None:
    active_votes[player_identity_key(voter)] = player_identity_key(target)


def _clear_vote(voter: str, active_votes: dict[str, str]) -> None:
    active_votes.pop(player_identity_key(voter), None)


def parse_showdown_log(raw_text: str, source: str = "unknown") -> GameSession:
    lines = [line.strip() for line in split_combined_system_lines(raw_text).splitlines() if line.strip()]
    session = GameSession(source=source, raw_text=raw_text)
    current_day = 1
    identity_to_canonical: dict[str, str] = {}

    for line in lines:
        player_list_match = PLAYERS_LIST_RE.search(line)
        if player_list_match:
            session.players.extend(parse_players_list(line))

    session.players = _register_roster_names(session.players, identity_to_canonical)
    roster_keys = {player_identity_key(player) for player in session.players if player_identity_key(player)}
    active_votes: dict[str, str] = {}

    for line in lines:
        day_marker_match = DAY_MARKER_RE.match(line)
        if day_marker_match:
            day_text = day_marker_match.group("day")
            if day_text:
                current_day = int(day_text)
            elif session.messages:
                current_day += 1
            else:
                current_day = 1
            active_votes.clear()
            session.phases.append(line)
            continue

        player_list_match = PLAYERS_LIST_RE.search(line)
        if player_list_match:
            continue

        elimination_match = ELIMINATION_RE.match(line)
        if elimination_match:
            player = canonical_player_name(elimination_match.group("player"))
            player = resolve_to_roster(player, identity_to_canonical)
            timestamp = parse_timestamp(elimination_match.group("timestamp"))
            session.events.append(
                LogEvent(
                    player_name=player,
                    event_type="elimination",
                    text=f"{player} was eliminated!",
                    day=current_day,
                    timestamp=timestamp,
                )
            )
            continue

        reveal_match = REVEAL_RE.match(line)
        if reveal_match:
            player = canonical_player_name(reveal_match.group("player"))
            player = resolve_to_roster(player, identity_to_canonical)
            role = reveal_match.group("role").strip()
            timestamp = parse_timestamp(reveal_match.group("timestamp"))
            session.events.append(
                LogEvent(
                    player_name=player,
                    event_type="reveal",
                    text=f"{player}'s role was {role}",
                    day=current_day,
                    timestamp=timestamp,
                )
            )
            continue

        system_unvote_match = SYSTEM_UNVOTE_RE.search(line)
        if system_unvote_match:
            voter = resolve_to_roster(system_unvote_match.group("voter"), identity_to_canonical)
            if not _player_allowed(voter, roster_keys, identity_to_canonical):
                continue
            session.votes.append(
                Vote(
                    voter_name=voter,
                    target_name="",
                    timestamp=None,
                    text=line,
                    day=current_day,
                    action="unvote",
                )
            )
            _clear_vote(voter, active_votes)
            continue

        system_vote_match = SYSTEM_VOTE_RE.search(line)
        if system_vote_match:
            voter = resolve_to_roster(system_vote_match.group("voter"), identity_to_canonical)
            target = resolve_to_roster(system_vote_match.group("target"), identity_to_canonical)
            if not _player_allowed(voter, roster_keys, identity_to_canonical):
                continue
            if roster_keys and player_identity_key(target) not in roster_keys:
                continue

            prior_target_key = _current_vote_target(voter, active_votes)
            target_key = player_identity_key(target)
            action = "shift" if prior_target_key and prior_target_key != target_key else "vote"
            session.votes.append(
                Vote(
                    voter_name=voter,
                    target_name=target,
                    timestamp=None,
                    text=line,
                    day=current_day,
                    action=action,
                )
            )
            _set_vote(voter, target, active_votes)
            continue

        chat_match = CHAT_LINE_RE.match(line)
        if not chat_match:
            continue

        username = resolve_to_roster(chat_match.group("username"), identity_to_canonical)
        if not _player_allowed(username, roster_keys, identity_to_canonical):
            continue

        timestamp = parse_timestamp(chat_match.group("timestamp"))
        message_text = chat_match.group("message").strip()

        session.messages.append(
            Message(
                player_name=username,
                text=message_text,
                timestamp=timestamp,
                day=current_day,
            )
        )

        unvote_match = UNVOTE_TEXT_RE.match(message_text)
        if unvote_match:
            session.votes.append(
                Vote(
                    voter_name=username,
                    target_name=unvote_match.group("target") or "",
                    timestamp=timestamp,
                    text=message_text,
                    day=current_day,
                    action="unvote",
                )
            )
            _clear_vote(username, active_votes)
            continue

        vote_match = VOTE_TEXT_RE.match(message_text)
        if vote_match:
            target = resolve_to_roster(vote_match.group("target"), identity_to_canonical)
            if roster_keys and player_identity_key(target) not in roster_keys:
                continue
            prior_target_key = _current_vote_target(username, active_votes)
            target_key = player_identity_key(target)
            action = "shift" if prior_target_key and prior_target_key != target_key else "vote"
            session.votes.append(
                Vote(
                    voter_name=username,
                    target_name=target,
                    timestamp=timestamp,
                    text=message_text,
                    day=current_day,
                    action=action,
                )
            )
            _set_vote(username, target, active_votes)

    return session
