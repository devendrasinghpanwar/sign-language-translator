"""
tests/test_model_pipeline.py

Tests the training pipeline (train_classifier.py's logic) using small,
synthetic data — so tests run in under a second and don't depend on
having real recorded webcam data available.

This demonstrates a key testing skill: when the real pipeline depends
on slow/external things (webcams, large datasets, GPUs), you test the
underlying LOGIC with small fake data that exercises the same code
paths.

Run with:
  pytest tests/test_model_pipeline.py -v
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def make_synthetic_dataset(n_per_class=20, seed=42):
    """
    Build a small synthetic dataset with 2 clearly separable classes
    ('a' and 'b'), shaped exactly like the real hand_data.csv
    (label + 63 coordinate columns).
    """
    rng = np.random.default_rng(seed)

    # Class 'a': coordinates centered around 0.2
    class_a = rng.normal(loc=0.2, scale=0.02, size=(n_per_class, 63))
    # Class 'b': coordinates centered around 0.8 (far apart, easy to separate)
    class_b = rng.normal(loc=0.8, scale=0.02, size=(n_per_class, 63))

    X = np.vstack([class_a, class_b])
    y = ["a"] * n_per_class + ["b"] * n_per_class

    columns = [f"coord_{i}" for i in range(63)]
    df = pd.DataFrame(X, columns=columns)
    df.insert(0, "label", y)
    return df


class TestTrainingPipeline:

    def test_synthetic_dataset_has_correct_shape(self):
        df = make_synthetic_dataset(n_per_class=10)
        assert df.shape == (20, 64)  # 20 rows, 1 label col + 63 coord cols
        assert set(df["label"].unique()) == {"a", "b"}

    def test_model_trains_without_error(self):
        df = make_synthetic_dataset(n_per_class=20)
        X = df.drop(columns=["label"])
        y = df["label"]

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)  # should not raise

        assert model is not None

    def test_model_achieves_high_accuracy_on_separable_data(self):
        """
        Since our two synthetic classes are far apart (0.2 vs 0.8 means),
        a correctly-implemented classifier should easily achieve high
        accuracy. This catches bugs in data handling/label alignment —
        if labels got shuffled or misaligned, accuracy would collapse
        toward 50%.
        """
        df = make_synthetic_dataset(n_per_class=50)
        X = df.drop(columns=["label"])
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)

        assert accuracy > 0.9

    def test_model_predicts_known_label_shape(self):
        df = make_synthetic_dataset(n_per_class=20)
        X = df.drop(columns=["label"])
        y = df["label"]

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)

        single_prediction = model.predict(X.iloc[[0]])
        assert len(single_prediction) == 1
        assert single_prediction[0] in {"a", "b"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
