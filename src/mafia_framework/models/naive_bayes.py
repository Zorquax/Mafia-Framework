from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .artifact import ModelBundle, align_features
from ..analysis.tells.registry import FEATURE_NAMES, FEATURE_SET_VERSION


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=1000, class_weight="balanced"),
            ),
        ]
    )


def train_model(
    X: pd.DataFrame,
    y: list[str],
    *,
    feature_names: list[str] | None = None,
) -> Pipeline:
    columns = feature_names if feature_names is not None else list(X.columns)
    aligned = align_features(X, columns)
    pipeline = build_pipeline()
    pipeline.fit(aligned, y)
    return pipeline


def train_bundle(
    X: pd.DataFrame,
    y: list[str],
    groups: list[int] | None = None,
    *,
    feature_names: list[str] | None = None,
) -> ModelBundle:
    columns = feature_names if feature_names is not None else list(FEATURE_NAMES)
    pipeline = train_model(X, y, feature_names=columns)
    metrics = evaluate_model(X, y, groups=groups, feature_names=columns)
    return ModelBundle(
        pipeline=pipeline,
        feature_names=columns,
        feature_set_version=FEATURE_SET_VERSION,
        alignment_mode="binary",
        metrics=metrics,
    )


def predict_probabilities(model: Pipeline, X: Any, feature_names: list[str] | None = None) -> list[dict[str, float]]:
    columns = feature_names if feature_names is not None else list(FEATURE_NAMES)
    aligned = align_features(pd.DataFrame(X), columns)
    probabilities = model.predict_proba(aligned)
    class_labels = list(model.named_steps["classifier"].classes_)
    return [dict(zip(class_labels, row)) for row in probabilities]


def evaluate_model(
    X: pd.DataFrame,
    y: list[str],
    groups: list[int] | None = None,
    n_splits: int = 5,
    *,
    feature_names: list[str] | None = None,
) -> dict[str, float]:
    if len(X) < 2:
        return {"accuracy": 0.0, "log_loss": 0.0}

    columns = feature_names if feature_names is not None else list(X.columns)
    aligned = align_features(X, columns)
    y_array = np.array(y)

    unique, counts = np.unique(y_array, return_counts=True)
    min_class_count = int(counts.min())
    if min_class_count < 2:
        return {"accuracy": 0.0, "log_loss": 0.0}

    pipeline = build_pipeline()
    scoring = ["accuracy", "neg_log_loss"]

    if groups is not None and len(set(groups)) >= 2:
        adjusted_splits = min(n_splits, len(set(groups)))
        cv = GroupKFold(n_splits=adjusted_splits)
        results = cross_validate(
            pipeline,
            aligned,
            y_array,
            cv=cv,
            groups=np.array(groups),
            scoring=scoring,
            return_train_score=False,
        )
    else:
        adjusted_splits = min(n_splits, min_class_count)
        from sklearn.model_selection import StratifiedKFold

        cv = StratifiedKFold(n_splits=adjusted_splits)
        results = cross_validate(
            pipeline,
            aligned,
            y_array,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
        )

    log_loss = float(-results["test_neg_log_loss"].mean())
    return {
        "accuracy": float(results["test_accuracy"].mean()),
        "log_loss": log_loss,
    }
