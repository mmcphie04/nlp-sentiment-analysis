#!/usr/bin/env python3
import json
import os
import re
import random
from typing import List, Dict
from datetime import datetime

import pandas as pd
from transformers import AutoTokenizer

# === CONFIG ===
CSV_PATH     = "imdb.csv"
MAX_TOKENS   = 250
MODEL_NAME   = "distilbert-base-uncased"
TRAIN_RATIO  = 0.9
SEED         = 42
OUTPUT_DIR   = "imdb_splits_90_10"
# ==============

def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", " ", text).strip()

def load_and_filter(tokenizer) -> List[Dict]:
    """
    Load all examples from local CSV, clean & filter out any review
    whose tokenized length > MAX_TOKENS. Returns a combined list.
    """
    print(f"Reading CSV from {CSV_PATH}…")
    df = pd.read_csv(CSV_PATH)
    # Map sentiment string → integer label
    df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})

    all_examples: List[Dict] = []
    kept = skipped = 0

    for _, row in df.iterrows():
        raw = str(row["review"])
        txt = clean_text(raw)

        # tokenize *without* adding special tokens
        tokens = tokenizer(txt, add_special_tokens=False)["input_ids"]
        if len(tokens) > MAX_TOKENS:
            skipped += 1
            continue

        all_examples.append({"review": txt, "label": int(row["label"])})
        kept += 1

    print(f"  Kept {kept}, skipped {skipped} (> {MAX_TOKENS} tokens)")
    print(f"Total examples after filtering: {len(all_examples)}")
    return all_examples

def split_and_save(
    examples: List[Dict],
    train_ratio: float = TRAIN_RATIO,
    output_dir: str = OUTPUT_DIR
):
    os.makedirs(output_dir, exist_ok=True)
    random.seed(SEED)
    random.shuffle(examples)

    n_train = int(len(examples) * train_ratio)
    train_exs = examples[:n_train]
    test_exs  = examples[n_train:]

    with open(os.path.join(output_dir, "train.json"), "w") as f:
        json.dump(train_exs, f, indent=2)
    with open(os.path.join(output_dir, "test.json"), "w") as f:
        json.dump(test_exs, f, indent=2)

    print(f"Saved {len(train_exs)} train and {len(test_exs)} test examples to '{output_dir}'")

def main():
    start = datetime.now()
    print(f"Started at {start.strftime('%Y-%m-%d %H:%M:%S')}")

    print(f"Loading tokenizer '{MODEL_NAME}'…")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    examples = load_and_filter(tokenizer)
    split_and_save(examples)

    end = datetime.now()
    elapsed = end - start
    print(f"Finished at {end.strftime('%Y-%m-%d %H:%M:%S')} (took {elapsed})")

if __name__ == "__main__":
    main()
