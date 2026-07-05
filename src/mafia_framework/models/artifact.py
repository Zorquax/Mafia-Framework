from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from ..analysis.tells.registry import FEATURE_NAMES, FEATURE_SET_VERSION


@dataclass
class ModelBundle:
    pipeline: Pipeline
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    feature_set_version: str = FEATURE_SET_VERSION
    alignment_mode: str = "binary"
    metrics: dict[str, float] = field(default_factory=dict)

    def predict_proba_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        aligned = align_features(X, self.feature_names)
        probabilities = self.pipeline.predict_proba(aligned)
        classes = list(self.pipeline.named_steps["classifier"].classes_)
        return pd.DataFrame(probabilities, columns=classes)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: str | Path) -> ModelBundle:
        with Path(path).open("rb") as handle:
            artifact = pickle.load(handle)
        if isinstance(artifact, cls):
            return artifact
        if isinstance(artifact, Pipeline):
            return cls(
                pipeline=artifact,
                feature_names=list(FEATURE_NAMES),
                feature_set_version="legacy",
            )
        raise TypeError(f"Unsupported model artifact type: {type(artifact)!r}")


def align_features(X: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    if "player_name" in X.columns:
        X = X.drop(columns=["player_name"])
    return X.reindex(columns=feature_names, fill_value=0.0)


def validate_feature_set_version(
    bundle: ModelBundle,
    *,
    strict: bool = True,
    allowed_versions: set[str] | None = None,
) -> None:
    allowed = allowed_versions or {FEATURE_SET_VERSION}
    from ..analysis.tells.registry import DAY_ONE_FEATURE_SET_VERSION

    allowed.add(DAY_ONE_FEATURE_SET_VERSION)
    if bundle.feature_set_version in allowed:
        return
    if bundle.feature_set_version == "legacy":
        if strict:
            raise ValueError(
                "Model was saved without feature_set_version. Retrain with the current framework."
            )
        return
    if strict:
        raise ValueError(
            f"Model feature_set_version {bundle.feature_set_version!r} "
            f"does not match allowed versions. Retrain the model."
        )
