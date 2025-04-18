#!/usr/bin/env python3
# baseline_bow.py

import os
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def load_imdb_json(split_dir: str):
    """Load train & test reviews + labels from JSON files."""
    train_path = os.path.join(split_dir, "train.json")
    test_path  = os.path.join(split_dir, "test.json")

    with open(train_path, "r") as f:
        train = json.load(f)
    with open(test_path, "r") as f:
        test = json.load(f)

    X_train = [ex["review"] for ex in train]
    y_train = [ex["label"]  for ex in train]
    X_test  = [ex["review"] for ex in test]
    y_test  = [ex["label"]  for ex in test]

    return X_train, y_train, X_test, y_test


def main():
    SPLIT_DIR = "imdb_splits_90_10"
    MAX_FEATURES = 5_000   # size of your BoW vocabulary

    # 1) Load data
    X_train, y_train, X_test, y_test = load_imdb_json(SPLIT_DIR)

    # 2) Vectorize with simple token counts
    vect = CountVectorizer(max_features=MAX_FEATURES)
    X_train_counts = vect.fit_transform(X_train)
    X_test_counts  = vect.transform(X_test)

    # 3) Train a logistic regression classifier
    clf = LogisticRegression(max_iter=1000)
    print("Fitting LogisticRegression on bag‑of‑words counts…")
    clf.fit(X_train_counts, y_train)

    # 4) Evaluate
    preds = clf.predict(X_test_counts)
    acc   = accuracy_score(y_test, preds)
    print(f"\nTest accuracy: {acc*100:.2f}%\n")
    print("Classification report:\n")
    print(classification_report(y_test, preds, digits=4))


if __name__ == "__main__":
    main()
