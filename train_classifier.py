"""
train_classifier.py

STEP 2 of the project.

WHAT THIS SCRIPT DOES (in plain English):
  1. Reads hand_data.csv (created by collect_data.py)
  2. Splits it into training data and testing data
  3. Trains a Random Forest classifier — this is a much simpler model
     than deep learning (LSTM/neural networks). It works by building
     many small decision trees and having them "vote" on the answer.
     It's a great beginner-friendly model because it's fast, doesn't
     need much data, and is easy to reason about.
  4. Tests how accurate it is
  5. Saves the trained model to disk so app.py can use it

WHY RANDOM FOREST INSTEAD OF DEEP LEARNING HERE?
  Since each sign in this version is a single hand POSE (not a moving
  sequence), we don't need to model time/motion. That means we don't
  need LSTMs or neural networks at all — a classical ML model works
  great and trains in under a second. This keeps things simple while
  you're learning, and still counts as a real machine learning project.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

CSV_PATH = "hand_data.csv"
MODEL_PATH = "sign_classifier.joblib"


def main():
    print("Loading data...")
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} samples across signs: {sorted(df['label'].unique())}")

    if len(df) < 20:
        print("WARNING: Very little data. Run collect_data.py first and "
              "collect at least 20-30 samples per sign.")
        return

    # X = the input features (63 hand coordinate numbers)
    # y = the label we're trying to predict (which sign it is)
    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training Random Forest classifier...")
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"\nTest accuracy: {accuracy * 100:.2f}%")
    print("\nDetailed report:")
    print(classification_report(y_test, predictions))

    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    print("Next step: run app.py to test it locally, then deploy it!")


if __name__ == "__main__":
    main()
