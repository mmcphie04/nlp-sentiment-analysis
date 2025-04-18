#!/usr/bin/env python3
# baseline_nb.py

import os
import json

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_imdb_json(split_dir: str):
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


def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    print(f"\n=== {name} ===")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc*100:.2f}%\n")

    print("Classification Report:")
    print(classification_report(y_test, preds, digits=4))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, preds)
    print(cm)


def main():
    SPLIT_DIR    = "imdb_splits_90_10"
    MAX_FEATURES = 10_000
    NGRAM_RANGE  = (1, 2)

    # 1) Load data
    X_train, y_train, X_test, y_test = load_imdb_json(SPLIT_DIR)

    # 2) Vectorize
    vect = CountVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        binary=False   # use raw counts
    )
    X_train_counts = vect.fit_transform(X_train)
    X_test_counts  = vect.transform(X_test)

    # 3) MultinomialNB
    mnb = MultinomialNB()
    evaluate_model("MultinomialNB", mnb, X_train_counts, y_train, X_test_counts, y_test)

    # 4) BernoulliNB
    #    binarize counts → presence/absence
    bnb_vect = CountVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        binary=True
    )
    X_train_bin = bnb_vect.fit_transform(X_train)
    X_test_bin  = bnb_vect.transform(X_test)

    bnb = BernoulliNB()
    evaluate_model("BernoulliNB", bnb, X_train_bin, y_train, X_test_bin, y_test)


if __name__ == "__main__":
    main()
