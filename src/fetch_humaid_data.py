"""
Downloads the HumAID disaster tweet dataset from Hugging Face, filters 
categories most likely to contain the target entities (location,
infra, casualty, resource), and samples tweets for manual labeling.
"""

import argparse
import pandas as pd
from datasets import load_dataset

# Categories most likely to contain LOCATION / INFRASTRUCTURE / CASUALTY / RESOURCE
RELEVANT_CATEGORIES = [
    "infrastructure_and_utility_damage",
    "injured_or_dead_people",
    "requests_or_urgent_needs",
    "displaced_people_and_evacuations",
    "rescue_volunteering_or_donation_effort",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_per_category", type=int, default=40,
                         help="How many tweets to sample per category")
    parser.add_argument("--outfile", default="data/to_label.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Downloading HumAID dataset from Hugging Face...")
    ds = load_dataset("QCRI/HumAID-all", split="train")
    df = ds.to_pandas()

    print(f"Loaded {len(df)} total tweets.")
    print("Category counts:\n", df["class_label"].value_counts())

    filtered = df[df["class_label"].isin(RELEVANT_CATEGORIES)]
    print(f"\n{len(filtered)} tweets in relevant categories.")

    # Light cleaning: drop retweet boilerplate duplicates, very short tweets
    filtered = filtered[filtered["tweet_text"].str.len() > 20]
    filtered = filtered.drop_duplicates(subset="tweet_text")

    sampled = (
        filtered.groupby("class_label", group_keys=False)
        .apply(lambda x: x.sample(min(len(x), args.n_per_category), random_state=args.seed))
        .reset_index(drop=True)
    )

    sampled["tagged_text"] = ""  # empty column for you to fill in during labeling
    sampled = sampled[["tweet_text", "class_label", "tagged_text"]]

    sampled.to_csv(args.outfile, index=False)
    print(f"\nWrote {len(sampled)} tweets to {args.outfile}, ready for labeling.")
    print("Open this CSV (Google Sheets / Excel) and fill in the 'tagged_text' column")
    print("using the tagging syntax described in the labeling guide.")


if __name__ == "__main__":
    main()
