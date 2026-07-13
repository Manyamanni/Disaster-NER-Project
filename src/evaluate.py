"""
evaluate.py

Runs the fine-tuned model on the held-out test set, prints entity-level
precision/recall/F1 per label, and prints qualitative error examples
(cases where the model's predicted tags didn't match the gold tags).

This is the "resume-worthy" part of the project — don't skip it. Showing
you measured failure modes, not just accuracy, is what separates a real
project from a tutorial clone.

Usage:
    python src/evaluate.py --model_dir model_output --data_dir data
"""

import json
import argparse
import numpy as np
import torch
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForTokenClassification
from seqeval.metrics import classification_report


def predict_tags(text_tokens, tokenizer, model, id2label, device):
    inputs = tokenizer(text_tokens, is_split_into_words=True, return_tensors="pt",
                        truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()

    word_ids = inputs.word_ids(batch_index=0)
    pred_tags = []
    previous_word_idx = None
    for idx, word_idx in zip(predictions, word_ids):
        if word_idx is None or word_idx == previous_word_idx:
            continue
        pred_tags.append(id2label[idx])
        previous_word_idx = word_idx
    return pred_tags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="model_output")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--max_error_examples", type=int, default=5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir).to(device)
    model.eval()

    id2label = model.config.id2label

    with open(Path(args.data_dir) / "test.json") as f:
        test_data = json.load(f)

    all_true, all_pred = [], []
    error_examples = []

    for ex in test_data:
        tokens, gold_tags = ex["tokens"], ex["tags"]
        pred_tags = predict_tags(tokens, tokenizer, model, id2label, device)

        # Guard against tokenizer edge cases producing mismatched lengths
        if len(pred_tags) != len(gold_tags):
            continue

        all_true.append(gold_tags)
        all_pred.append(pred_tags)

        if pred_tags != gold_tags and len(error_examples) < args.max_error_examples:
            error_examples.append((tokens, gold_tags, pred_tags))

    print("=== Entity-level classification report (test set) ===\n")
    print(classification_report(all_true, all_pred))

    print("\n=== Sample errors (for your write-up / error analysis) ===\n")
    for tokens, gold, pred in error_examples:
        print("Text:      ", " ".join(tokens))
        print("Gold tags: ", gold)
        print("Pred tags: ", pred)
        mismatches = [
            (t, g, p) for t, g, p in zip(tokens, gold, pred) if g != p
        ]
        print("Mismatched tokens:", mismatches)
        print("-" * 60)


if __name__ == "__main__":
    main()
