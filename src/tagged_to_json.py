"""
Converts manually labeled tweets into the
tokens/tags BIO-format JSON expected by prepare_data.py

TAGGING SYNTAX:
Wrap each entity span in tags like this, directly in the tweet text:
    Flooding hit <LOCATION>Houston</LOCATION> and damaged
    <INFRASTRUCTURE>the hospital</INFRASTRUCTURE>, 12
    <CASUALTY>people trapped</CASUALTY>, need <RESOURCE>boats</RESOURCE>
"""

import re
import json
import argparse
import pandas as pd

VALID_TAGS = {"LOCATION", "INFRASTRUCTURE", "CASUALTY", "RESOURCE"}
TAG_PATTERN = re.compile(r"<(\w+)>(.*?)</\1>")


def parse_tagged_text(text: str):
    """
    Converts a string with <TAG>...</TAG> spans into (tokens, tags) lists
    in BIO format.
    """
    tokens, tags = [], []
    pos = 0

    for match in TAG_PATTERN.finditer(text):
        tag_name = match.group(1).upper()
        entity_text = match.group(2)

        # Add any plain (untagged) text before this match as "O" tokens
        plain_before = text[pos:match.start()]
        for word in plain_before.split():
            tokens.append(word)
            tags.append("O")

        if tag_name not in VALID_TAGS:
            # Unknown tag: treat its content as plain text instead of failing
            for word in entity_text.split():
                tokens.append(word)
                tags.append("O")
        else:
            entity_words = entity_text.split()
            for i, word in enumerate(entity_words):
                tokens.append(word)
                tags.append(f"B-{tag_name}" if i == 0 else f"I-{tag_name}")

        pos = match.end()

    # Any remaining plain text after the last tag
    for word in text[pos:].split():
        tokens.append(word)
        tags.append("O")

    return tokens, tags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", default="data/to_label.csv")
    parser.add_argument("--outfile", default="data/sample_labeled_data.json")
    parser.add_argument("--text_column", default="tagged_text")
    args = parser.parse_args()

    df = pd.read_csv(args.infile)

    if args.text_column not in df.columns:
        raise ValueError(f"Column '{args.text_column}' not found in {args.infile}")

    examples = []
    skipped = 0
    for _, row in df.iterrows():
        text = row[args.text_column]
        if not isinstance(text, str) or not text.strip():
            skipped += 1
            continue
        tokens, tags = parse_tagged_text(text)
        if not tokens:
            skipped += 1
            continue
        examples.append({"tokens": tokens, "tags": tags})

    with open(args.outfile, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2)

    print(f"Converted {len(examples)} labeled tweets to {args.outfile}")
    print(f"Skipped {skipped} rows (empty or untagged 'tagged_text' cells)")

    # quick sanity check: show label distribution
    from collections import Counter
    tag_counts = Counter(t for ex in examples for t in ex["tags"] if t != "O")
    print("\nEntity tag counts (sanity check):")
    for tag, count in sorted(tag_counts.items()):
        print(f"  {tag}: {count}")


if __name__ == "__main__":
    main()
