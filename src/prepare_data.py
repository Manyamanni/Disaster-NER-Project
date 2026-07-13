"""
prepare_data.py

Loads labeled disaster-text data (token + BIO tag format) and prepares
train/validation/test splits saved as JSON, ready for tokenization.

Expected input format (see data/sample_labeled_data.json):
[
  {
    "tokens": ["Flooding", "has", "cut", "off", ...],
    "tags":   ["O", "O", "O", "O", ...]
  },
  ...
]

To use real data instead of the small sample file:
  1. Download CrisisNLP or HumAID datasets (search "CrisisNLP" or "HumAID GitHub").
  2. Write a small converter that turns their raw text + entity spans into
     the tokens/tags format above (see convert_conll_to_json() for a template
     if your source data is already in CoNLL/BIO column format).
  3. Point --input at your converted JSON file.
"""

import json
import argparse
import random
from pathlib import Path

LABELS = [
    "O",
    "B-LOCATION", "I-LOCATION",
    "B-INFRASTRUCTURE", "I-INFRASTRUCTURE",
    "B-CASUALTY", "I-CASUALTY",
    "B-RESOURCE", "I-RESOURCE",
]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}


def convert_conll_to_json(conll_path: str) -> list:
    """
    Template converter if your source data is in classic CoNLL column format:
        token1 TAG1
        token2 TAG2
        <blank line separates sentences>
    Adjust column indices if your source has extra columns (POS tags, etc).
    """
    sentences = []
    tokens, tags = [], []
    with open(conll_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append({"tokens": tokens, "tags": tags})
                    tokens, tags = [], []
                continue
            parts = line.split()
            tokens.append(parts[0])
            tags.append(parts[-1])
    if tokens:
        sentences.append({"tokens": tokens, "tags": tags})
    return sentences


def validate_examples(examples: list) -> list:
    """Drop any malformed examples (mismatched token/tag lengths, unknown labels)."""
    clean = []
    for ex in examples:
        if len(ex["tokens"]) != len(ex["tags"]):
            continue
        if any(tag not in LABEL2ID for tag in ex["tags"]):
            continue
        if len(ex["tokens"]) == 0:
            continue
        clean.append(ex)
    return clean


def split_data(examples: list, train_ratio=0.7, val_ratio=0.15, seed=42):
    random.Random(seed).shuffle(examples)
    n = len(examples)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return (
        examples[:n_train],
        examples[n_train:n_train + n_val],
        examples[n_train + n_val:],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/sample_labeled_data.json",
                         help="Path to labeled data JSON (tokens/tags format)")
    parser.add_argument("--outdir", default="data")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        examples = json.load(f)

    examples = validate_examples(examples)
    print(f"Loaded {len(examples)} valid labeled examples.")

    if len(examples) < 20:
        print(
            "\nWARNING: You're working with a very small dataset (this is expected "
            "if you're still using data/sample_labeled_data.json, which is just a "
            "5-example placeholder to show the expected format).\n"
            "For a real project, download CrisisNLP or HumAID labeled tweet data "
            "and convert it into this tokens/tags format before training.\n"
        )

    train, val, test = split_data(examples)
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    for name, split in [("train", train), ("val", val), ("test", test)]:
        path = outdir / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(split, f, indent=2)
        print(f"Wrote {len(split)} examples to {path}")

    with open(outdir / "label_map.json", "w") as f:
        json.dump({"label2id": LABEL2ID, "id2label": ID2LABEL}, f, indent=2)
    print(f"Wrote label map ({len(LABELS)} labels) to {outdir / 'label_map.json'}")


if __name__ == "__main__":
    main()
