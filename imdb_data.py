# python3 split_imdb_90_10.py

import json
import os
import re
import random
import requests
import tarfile
from typing import List, Dict, Tuple
from datetime import datetime

from transformers import AutoTokenizer

MAX_TOKENS   = 250
MODEL_NAME  = "distilbert-base-uncased"
TRAIN_RATIO = 0.9
SEED        = 42

def download_imdb_dataset() -> str:
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    dataset_dir = "imdb_data"
    os.makedirs(dataset_dir, exist_ok=True)
    archive_path = os.path.join(dataset_dir, "aclImdb_v1.tar.gz")

    if not os.path.exists(archive_path):
        print("Downloading IMDb dataset...")
        resp = requests.get(url, stream=True)
        total = int(resp.headers.get("content-length", 0))
        dl = 0
        with open(archive_path, "wb") as f:
            for chunk in resp.iter_content(1024):
                f.write(chunk)
                dl += len(chunk)
                if total:
                    pct = dl * 100 / total
                    print(f"\r  {pct:.1f}% downloaded", end="")
        print("\nDownload complete!")

    extracted = os.path.join(dataset_dir, "aclImdb")
    if not os.path.isdir(extracted):
        print("Extracting dataset...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=dataset_dir)
        print("Extraction complete!")

    return extracted

def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", " ", text).strip()

def load_and_filter(tokenizer) -> List[Dict]:
    """
    Load both train+test IMDB splits, clean & filter out any review
    whose tokenized length > MAX_TOKENS.
    Returns a single combined list.
    """
    base = download_imdb_dataset()
    all_examples: List[Dict] = []

    for split in ("train", "test"):
        for sentiment, label in (("pos", 1), ("neg", 0)):
            path = os.path.join(base, split, sentiment)
            if not os.path.isdir(path):
                continue

            print(f"Loading & filtering {split}/{sentiment}...")
            kept = 0
            skipped = 0

            for fn in os.listdir(path):
                if not fn.endswith(".txt"):
                    continue
                full = os.path.join(path, fn)
                with open(full, encoding="utf-8", errors="replace") as f:
                    raw = f.read()
                txt = clean_text(raw)

                # tokenize *without* adding special tokens
                tokens = tokenizer(txt, add_special_tokens=False)["input_ids"]
                if len(tokens) > MAX_TOKENS:
                    skipped += 1
                    continue

                all_examples.append({"review": txt, "label": label})
                kept += 1

            print(f"  Kept {kept}, skipped {skipped} (> {MAX_TOKENS} tokens)")

    print(f"Total examples after filtering: {len(all_examples)}")
    return all_examples

def split_and_save(
    examples: List[Dict],
    train_ratio: float = TRAIN_RATIO,
    output_dir: str = "imdb_splits_90_10"
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
    print(f"Finished at {end.strftime('%Y-%m-%d %H:%M:%S')} (took {end - start})")

if __name__ == "__main__":
    main()
