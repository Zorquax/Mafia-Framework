import tempfile
import unittest
from pathlib import Path

import pandas as pd

from mafia_framework.analysis.tells.base import TellFeatures
from mafia_framework.analysis.tells.line_count import LineCountTell
from mafia_framework.analysis.tells.registry import FEATURE_NAMES
from mafia_framework.data.models import Flip, GameSession, Message
from mafia_framework.models.artifact import ModelBundle, align_features
from mafia_framework.models.feature_engineering import build_feature_dataframe, build_training_dataset
from mafia_framework.models.naive_bayes import train_bundle, evaluate_model, train_model


class TestModel(unittest.TestCase):

    def test_model_training(self):
        tells = [
            TellFeatures(player_name="Alice", features={"line_count": 3.0, "keyword_score": 2.0}),
            TellFeatures(player_name="Bob", features={"line_count": 1.0, "keyword_score": -1.0}),
        ]
        df = build_feature_dataframe(tells, feature_names=["line_count", "keyword_score"])
        X = df.drop(columns=["player_name"])
        y = ["town", "mafia"]

        model = train_model(X, y, feature_names=["line_count", "keyword_score"])
        self.assertIsNotNone(model)
        predictions = model.predict(X)
        self.assertEqual(len(predictions), 2)

        metrics = evaluate_model(X, y, n_splits=2, feature_names=["line_count", "keyword_score"])
        self.assertIn("accuracy", metrics)
        self.assertIn("log_loss", metrics)

    def test_build_training_dataset_infers_town_for_nonflipped_players(self):
        session = GameSession(
            source="test",
            raw_text="",
            players=["MafiaGuy", "Townie1", "Townie2"],
            messages=[
                Message(player_name="MafiaGuy", text="I am mafia", timestamp="00:01", day=1),
                Message(player_name="Townie1", text="Hi", timestamp="00:02", day=1),
                Message(player_name="Townie2", text="Hello", timestamp="00:03", day=1),
            ],
            flips=[Flip(player_name="MafiaGuy", alignment="mafia")],
            game_id=1,
        )
        tell_extractors = [LineCountTell()]
        X, y, groups = build_training_dataset(
            [session],
            tell_extractors,
            feature_names=["line_count"],
        )

        self.assertEqual(len(X), 3)
        self.assertCountEqual(y, ["mafia", "town", "town"])
        self.assertEqual(groups, [1, 1, 1])

    def test_binary_training_excludes_neutral(self):
        session = GameSession(
            source="test",
            raw_text="",
            players=["NeutralGuy", "Townie"],
            messages=[
                Message(player_name="NeutralGuy", text="hi", day=1),
                Message(player_name="Townie", text="hello", day=1),
            ],
            flips=[
                Flip(player_name="NeutralGuy", alignment="neutral"),
                Flip(player_name="Townie", alignment="town"),
            ],
            game_id=2,
        )
        X, y, _ = build_training_dataset(
            [session],
            [LineCountTell()],
            feature_names=["line_count"],
            binary_only=True,
        )
        self.assertEqual(len(X), 1)
        self.assertEqual(y, ["town"])

    def test_model_bundle_roundtrip(self):
        X = pd.DataFrame(
            [
                {"line_count": 3.0, "keyword_score": 2.0},
                {"line_count": 1.0, "keyword_score": -1.0},
            ]
        )
        y = ["town", "mafia"]
        bundle = train_bundle(X, y, feature_names=["line_count", "keyword_score"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pkl"
            bundle.save(path)
            loaded = ModelBundle.load(path)
            self.assertEqual(loaded.feature_names, ["line_count", "keyword_score"])
            aligned = align_features(X, loaded.feature_names)
            self.assertEqual(list(aligned.columns), ["line_count", "keyword_score"])
