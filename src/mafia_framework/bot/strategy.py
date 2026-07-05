import logging
from typing import Optional, Tuple, Dict
from ..services.model_service import predict_session, PredictionRow
from ..data.models import GameSession
from ..io.player_names import player_identity_key

logger = logging.getLogger("mafia_bot.strategy")

class BotStrategy:
    def __init__(self, model_path: str, model_d1_path: str, min_confidence: float = 0.55):
        self.model_path = model_path
        self.model_d1_path = model_d1_path
        self.min_confidence = min_confidence

        # Real-time decision override state
        self.suspicion_multipliers: Dict[str, float] = {}  # player -> multiplier

    def set_suspicion_multiplier(self, player_name: str, multiplier: float):
        logger.info(f"Setting suspicion multiplier for {player_name} to {multiplier}")
        self.suspicion_multipliers[player_name] = multiplier

    def reset(self):
        self.suspicion_multipliers = {}

    def _score_players(self, session: GameSession, bot_username: str, db_path: str) -> list[Tuple[str, float]]:
        """Returns [(player_name, adjusted_mafia_probability), ...] for all valid targets."""
        # Determine if we should use Day 1 model
        # Use day_one model if current day is 1 and day_one model file exists.
        # A freshly-started day 1 has no messages/votes recorded yet -- that's
        # still day 1, not evidence against it, so default to True when there's
        # no data at all rather than treating "no data" as "not day one".
        from pathlib import Path
        known_days = [m.day for m in session.messages] + [v.day for v in session.votes]
        day_one = all(day == 1 for day in known_days) if known_days else True
        model_to_use = self.model_d1_path if (day_one and Path(self.model_d1_path).exists()) else self.model_path

        if not Path(model_to_use).exists():
            logger.warning(f"Model file {model_to_use} not found! Cannot make automated vote decision.")
            return []

        logger.info(f"Running predictions using model: {model_to_use} (day_one={day_one})")
        try:
            predictions = predict_session(
                session,
                model_to_use,
                day_one=day_one,
                db_path=db_path
            )
        except Exception as e:
            logger.error(f"Error during predict_session: {e}")
            return []

        # Get sets of already flipped or dead players using normalized identity keys.
        flipped_player_keys = {player_identity_key(flip.player_name) for flip in session.flips}
        dead_player_keys = {
            player_identity_key(event.player_name)
            for event in session.events
            if event.event_type in ("elimination", "reveal")
        }
        bot_user_key = player_identity_key(bot_username)

        # Score players
        scored_players = []
        for pred in predictions:
            player_key = player_identity_key(pred.player_name)

            # Skip flipped/dead players and self
            if not player_key:
                continue
            if player_key in flipped_player_keys or player_key in dead_player_keys:
                continue
            if player_key == bot_user_key:
                continue

            prob_mafia = pred.probabilities.get("mafia", 0.0)

            # Apply real-time suspicion multiplier if set
            multiplier = self.suspicion_multipliers.get(pred.player_name, 1.0)
            adjusted_prob = prob_mafia * multiplier
            # Cap at 1.0
            adjusted_prob = min(adjusted_prob, 1.0)

            scored_players.append((pred.player_name, adjusted_prob))

        return scored_players

    def get_vote_decision(self, session: GameSession, bot_username: str, db_path: str) -> Tuple[Optional[str], float]:
        """
        Analyzes the GameSession and returns (target_player, probability).
        If no player meets the criteria, returns (None, 0.0).
        """
        scored_players = self._score_players(session, bot_username, db_path)

        # Sort by adjusted probability descending
        scored_players.sort(key=lambda item: item[1], reverse=True)

        if not scored_players:
            logger.info("No valid voting targets found.")
            return None, 0.0

        target, prob = scored_players[0]
        logger.info(f"Top suspect: {target} with probability: {prob:.4f}")

        if prob >= self.min_confidence:
            return target, prob
        else:
            logger.info(f"Top suspect probability ({prob:.4f}) is below min confidence ({self.min_confidence})")
            return None, prob

    def get_full_predictions(self, session: GameSession, bot_username: str, db_path: str) -> list[Tuple[str, float]]:
        """Returns [(player_name, adjusted_mafia_probability), ...] for every
        valid target, sorted most-suspicious first. Unlike get_vote_decision,
        this returns the whole ranked list rather than just the top pick.
        """
        scored_players = self._score_players(session, bot_username, db_path)
        scored_players.sort(key=lambda item: item[1], reverse=True)
        return scored_players

    def get_town_read(self, session: GameSession, bot_username: str, db_path: str) -> Tuple[Optional[str], float]:
        """
        Analyzes the GameSession and returns the player the bot is most
        confident is town, as (target_player, town_probability).
        If no player meets the confidence bar, returns (None, town_probability_of_best_candidate).
        """
        scored_players = self._score_players(session, bot_username, db_path)

        # Sort by adjusted mafia probability ascending (i.e. most town-confident first)
        scored_players.sort(key=lambda item: item[1])

        if not scored_players:
            logger.info("No valid town-read targets found.")
            return None, 0.0

        target, prob_mafia = scored_players[0]
        prob_town = 1.0 - prob_mafia
        logger.info(f"Strongest town read: {target} with probability: {prob_town:.4f}")

        if prob_town >= self.min_confidence:
            return target, prob_town
        else:
            logger.info(f"Best town read probability ({prob_town:.4f}) is below min confidence ({self.min_confidence})")
            return None, prob_town
