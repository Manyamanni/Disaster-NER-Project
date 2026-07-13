# Disaster response NER

Fine-tuned transformer model that extracts structured information from
disaster-related text (tweets, news, situation reports) — locations,
damaged infrastructure, casualty mentions, and resource needs — to help
speed up triage during emergency response.

Built on top of Hugging Face's standard token-classification fine-tuning
pipeline (`transformers` + `datasets`), extended with a custom disaster-domain
label schema, an entity-level evaluation harness with error analysis, and an
interactive demo.

## Entity types

| Tag | Meaning | Example |
|---|---|---|
| `LOCATION` | Places affected | "5th Avenue", "Houston" |
| `INFRASTRUCTURE` | Damaged/blocked structures | "hospital", "bridge" |
| `CASUALTY` | People affected/trapped/injured | "12 people trapped" |
| `RESOURCE` | Needs mentioned | "boats", "medical supplies" |

## Project structure

```
disaster-ner-project/
├── data/
│   └── sample_labeled_data.json   # 5-example placeholder — REPLACE with real data
├── src/
│   ├── prepare_data.py            # splits labeled data into train/val/test
│   ├── train.py                   # fine-tunes DistilBERT for token classification
│   ├── evaluate.py                # entity-level F1 + qualitative error analysis
│   └── app.py                     # Streamlit demo
├── notebooks/
│   └── run_on_colab.ipynb         # run the full pipeline on free Colab GPU
└── requirements.txt
```

## Before you start: get a real dataset

`data/sample_labeled_data.json` has only 5 hand-written examples so you can
see the expected format. **You need to replace this with real labeled data**
before training something resume-worthy. Two solid free options:

- **CrisisNLP** — labeled disaster-related tweets across multiple crisis
  events. Search "CrisisNLP datasets" for their download page.
- **HumAID** — a large labeled disaster tweet dataset from CrisisNLP's team,
  organized by disaster type (floods, earthquakes, hurricanes, wildfires).

These datasets typically come with *classification* labels (e.g. "infrastructure
damage", "requests or urgent needs") rather than token-level BIO tags. You'll
need to do light manual annotation on a subset (100-200 tweets is enough) to
get token-level entity tags — or use the tweet text + classification label as
a starting point and manually mark entity spans within tweets already flagged
as relevant. This annotation step is worth keeping — being able to say
"I built the labeled training set myself" is a real interview talking point.

A fast way to hand-label ~150-200 short tweets: open them in a spreadsheet,
read each one, and mark entity spans. It's more time-effective than it sounds
because tweets are short.

## Running the project

### Option A — Colab (recommended, free GPU)
Open `notebooks/run_on_colab.ipynb` in Google Colab, set the runtime to GPU,
and run the cells in order. See the notebook for details.

### Option B — locally (CPU, slower, fine for testing)
```bash
pip install -r requirements.txt

# 1. Prepare data (swap --input for your real labeled data once ready)
python src/prepare_data.py --input data/sample_labeled_data.json --outdir data

# 2. Train
python src/train.py --epochs 5 --batch_size 16

# 3. Evaluate — prints entity-level precision/recall/F1 + error examples
python src/evaluate.py --model_dir model_output --data_dir data

# 4. Run the interactive demo
streamlit run src/app.py
```

## What to actually spend your time on

Given a 4-5 day timeline, the code here handles the boilerplate. Spend your
time on:

1. **Getting a real, reasonably sized labeled dataset** (Day 1-2) — this is
   the single highest-leverage thing for making the project legitimate.
2. **Training + evaluating** (Day 3) — run `train.py`, then `evaluate.py`,
   and actually read the error examples it prints. Note patterns (e.g. "model
   confuses infrastructure and location when both appear together") — this
   becomes your write-up.
3. **The demo** (Day 4) — get `app.py` working end-to-end; optionally extend
   it with a simple map (geocode extracted `LOCATION` entities with a free
   geocoding API and plot them).
4. **README + write-up** (Day 5) — document your eval numbers, a couple of
   good/bad example outputs, and what you'd improve with more time/data.

## Resume bullet

> Fine-tuned a transformer-based NER model to extract locations,
> infrastructure damage, and resource needs from disaster-related text;
> built an entity-level evaluation harness (precision/recall/F1) and an
> interactive demo, achieving [X]% F1 on held-out data.

Fill in your actual F1 once you've trained on real data — don't guess a
number, and don't be discouraged if it's not huge on a small dataset. A
modest F1 with honest error analysis is more credible in an interview than
an unverifiable high number.

## Honesty note

This scaffold builds on Hugging Face's standard token-classification training
pattern. If asked in an interview, it's completely normal (and expected) to
say you started from a standard fine-tuning pipeline and focused your effort
on the dataset, evaluation, and domain-specific extensions — that's how real
ML engineering work looks. Don't claim the training loop itself as a novel
contribution; claim the dataset curation, error analysis, and product framing,
because that's the part that's actually yours.
