from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..data.aliases import apply_player_aliases, load_alias_map
from ..data.flips import extract_flips
from ..io.parser import parse_showdown_log
from ..models.artifact import ModelBundle
from ..models.naive_bayes import train_bundle
from .format_service import format_metric, format_probability
from .tell_service import build_labeled_training_frame, compute_tell_dataframe


@dataclass
class PredictionRow:
    player_name: str
    probabilities: dict[str, float]
    formatted: dict[str, str]
    top_signal: str


@dataclass
class TrainResult:
    bundle: ModelBundle
    output_path: str
    mode: str


def train_model_from_db(
    db_path: str,
    output_path: str,
    *,
    day_one: bool = False,
    exclude_neutral_games: bool = False,
) -> TrainResult:
    X, y, groups = build_labeled_training_frame(
        db_path,
        day_one=day_one,
        exclude_neutral_games=exclude_neutral_games,
    )
    if X.empty or not y:
        raise ValueError("Could not build training data. Ensure your games include flip annotations.")

    from ..analysis.tells.registry import DAY_ONE_FEATURE_NAMES, FEATURE_NAMES, DAY_ONE_FEATURE_SET_VERSION

    feature_names = DAY_ONE_FEATURE_NAMES if day_one else FEATURE_NAMES
    bundle = train_bundle(X, y, groups=groups, feature_names=feature_names)
    if day_one:
        bundle.feature_set_version = DAY_ONE_FEATURE_SET_VERSION
    bundle.save(output_path)
    mode = "day-one" if day_one else "full"
    return TrainResult(bundle=bundle, output_path=output_path, mode=mode)


def predict_from_text(
    raw_text: str,
    model_path: str,
    source: str = "unknown",
    db_path: str | None = None,
    *,
    day_one: bool = False,
) -> list[PredictionRow]:
    session = parse_showdown_log(raw_text, source=source)
    session.flips = extract_flips(raw_text)
    if db_path:
        session = apply_player_aliases(session, load_alias_map(db_path))

    frame = compute_tell_dataframe(session, day_one=day_one, db_path=db_path)
    bundle = ModelBundle.load(model_path)
    prob_frame = bundle.predict_proba_frame(frame.drop(columns=["player_name"], errors="ignore"))

    rows: list[PredictionRow] = []
    feature_cols = [col for col in frame.columns if col != "player_name"]
    for idx, player_name in enumerate(frame["player_name"]):
        probs = prob_frame.iloc[idx].to_dict()
        formatted = format_probability(probs)
        top_signal = _summarize_top_signal(frame.iloc[idx], feature_cols)
        rows.append(
            PredictionRow(
                player_name=player_name,
                probabilities=probs,
                formatted=formatted,
                top_signal=top_signal,
            )
        )
    rows.sort(key=lambda row: row.probabilities.get("mafia", 0.0), reverse=True)
    return rows


def predict_session(
    session,
    model_path: str,
    *,
    day_one: bool = False,
    db_path: str | None = None,
) -> list[PredictionRow]:
    frame = compute_tell_dataframe(session, day_one=day_one, db_path=db_path)
    bundle = ModelBundle.load(model_path)
    prob_frame = bundle.predict_proba_frame(frame.drop(columns=["player_name"], errors="ignore"))
    feature_cols = [col for col in frame.columns if col != "player_name"]
    rows: list[PredictionRow] = []
    for idx, player_name in enumerate(frame["player_name"]):
        probs = prob_frame.iloc[idx].to_dict()
        rows.append(
            PredictionRow(
                player_name=player_name,
                probabilities=probs,
                formatted=format_probability(probs),
                top_signal=_summarize_top_signal(frame.iloc[idx], feature_cols),
            )
        )
    rows.sort(key=lambda row: row.probabilities.get("mafia", 0.0), reverse=True)
    return rows


def get_feature_importance(model_path: str) -> pd.DataFrame:
    bundle = ModelBundle.load(model_path)
    classifier = bundle.pipeline.named_steps["classifier"]
    coefficients = classifier.coef_
    classes = list(classifier.classes_)
    if len(classes) != 2:
        raise ValueError("Feature importance expects a binary classifier.")

    mafia_index = classes.index("mafia") if "mafia" in classes else 1
    coefs = coefficients[mafia_index]
    frame = pd.DataFrame(
        {
            "feature": bundle.feature_names,
            "coefficient": coefs,
            "abs_coefficient": [abs(value) for value in coefs],
            "direction": ["mafia+" if value > 0 else "town+" for value in coefs],
        }
    )
    return frame.sort_values("abs_coefficient", ascending=False)


def get_player_tell_comparison(db_path: str, player_name: str, *, day_one: bool = False) -> pd.DataFrame:
    from ..data.aliases import load_alias_map, resolve_name

    alias_map = load_alias_map(db_path)
    canonical = resolve_name(player_name, alias_map) if alias_map else player_name
    all_tells = compute_all_game_tells_for_training(db_path, day_one=day_one)
    if all_tells.empty:
        return pd.DataFrame()

    player_rows = all_tells[all_tells["canonical_name"] == canonical]
    if player_rows.empty:
        return pd.DataFrame()

    feature_cols = [
        col
        for col in all_tells.columns
        if col not in {"game_id", "player_name", "canonical_name", "alignment"}
    ]
    player_means = player_rows[feature_cols].mean(numeric_only=True)
    town_means = all_tells[all_tells["alignment"] == "town"][feature_cols].mean(numeric_only=True)
    mafia_means = all_tells[all_tells["alignment"] == "mafia"][feature_cols].mean(numeric_only=True)

    comparison = pd.DataFrame(
        {
            "feature": feature_cols,
            "player_mean": [player_means.get(col, 0.0) for col in feature_cols],
            "town_mean": [town_means.get(col, 0.0) for col in feature_cols],
            "mafia_mean": [mafia_means.get(col, 0.0) for col in feature_cols],
        }
    )
    comparison["delta_from_town"] = comparison["player_mean"] - comparison["town_mean"]
    comparison["delta_from_mafia"] = comparison["player_mean"] - comparison["mafia_mean"]
    comparison["abs_delta_from_town"] = comparison["delta_from_town"].abs()
    return comparison.sort_values("abs_delta_from_town", ascending=False)


def compute_all_game_tells_for_training(db_path: str, *, day_one: bool = False) -> pd.DataFrame:
    from ..data.aliases import load_alias_map, resolve_name
    from ..io.ingestion import load_games

    alias_map = load_alias_map(db_path)
    sessions = load_games(db_path)
    frames: list[pd.DataFrame] = []
    for session in sessions:
        frame = compute_tell_dataframe(session, day_one=day_one, db_path=db_path)
        if frame.empty:
            continue
        frame.insert(0, "game_id", session.game_id)
        frame["canonical_name"] = frame["player_name"].apply(
            lambda name: resolve_name(name, alias_map) if alias_map else name
        )
        from .game_service import resolve_flip_map

        resolved = resolve_flip_map(session)
        frame["alignment"] = frame["player_name"].map(resolved)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _summarize_top_signal(row: pd.Series, feature_cols: list[str]) -> str:
    if not feature_cols:
        return "n/a"
    numeric = row[feature_cols].astype(float)
    top = numeric.abs().sort_values(ascending=False).head(1)
    if top.empty:
        return "n/a"
    feature = top.index[0]
    return f"{feature}={float(row[feature]):.2f}"


def format_train_metrics(bundle: ModelBundle) -> dict[str, str]:
    return {key: format_metric(key, value) for key, value in bundle.metrics.items()}
