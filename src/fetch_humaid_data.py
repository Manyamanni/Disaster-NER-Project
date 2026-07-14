"""
Downloads the HumAID disaster tweet dataset from Hugging Face, filters to
categories most likely to contain our target entities (location,
infras\, casualty, resource), and samples a manageable number of
tweets for manual labeling.

"""

import argparse
import requests
import pandas as pd

# Categories most likely to contain LOCATION / INFRASTRUCTURE / CASUALTY / RESOURCE
RELEVANT_CATEGORIES = [
    "infrastructure_and_utility_damage",
    "injured_or_dead_people",
    "requests_or_urgent_needs",
    "displaced_people_and_evacuations",
    "rescue_volunteering_or_donation_effort",
]


def fetch_humaid_dataframe(split="train"):
    """
    Downloads HumAID directly as a parquet file via HF's datasets-server API,
    bypassing the `datasets` library's xet-based download path (which has been
    hitting a broken CDN endpoint as of writing this).
    """
    api_url = "https://datasets-server.huggingface.co/parquet?dataset=QCRI/HumAID-all"
    resp = requests.get(api_url, timeout=30)
    resp.raise_for_status()
    parquet_info = resp.json()

    split_entry = next(
        (f for f in parquet_info["parquet_files"] if f["split"] == split), None
    )
    if split_entry is None:
        raise ValueError(f"No parquet file found for split '{split}'")

    parquet_url = split_entry["url"]
    print(f"Downloading parquet file directly from: {parquet_url}")
    df = pd.read_parquet(parquet_url)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_per_category", type=int, default=40,
                         help="How many tweets to sample per category")
    parser.add_argument("--outfile", default="data/to_label.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Downloading HumAID dataset from Hugging Face...")
    df = fetch_humaid_dataframe(split="train")

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
