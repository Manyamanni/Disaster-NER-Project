"""
app.py

Interactive Streamlit demo for the disaster-response NER model.
Paste in disaster-related text and see extracted entities highlighted.

Run with:
    streamlit run src/app.py
"""

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

MODEL_DIR = "model_output"

COLORS = {
    "LOCATION": "#B5D4F4",
    "INFRASTRUCTURE": "#F0997B",
    "CASUALTY": "#F09595",
    "RESOURCE": "#97C459",
}


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model


def predict(text, tokenizer, model):
    tokens = text.split()
    inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    preds = torch.argmax(outputs.logits, dim=2)[0].numpy()
    word_ids = inputs.word_ids(batch_index=0)

    id2label = model.config.id2label
    results = []
    previous_word_idx = None
    for idx, word_idx in zip(preds, word_ids):
        if word_idx is None or word_idx == previous_word_idx:
            continue
        results.append((tokens[word_idx], id2label[idx]))
        previous_word_idx = word_idx
    return results


def render_highlighted(results):
    html_parts = []
    for token, tag in results:
        if tag == "O":
            html_parts.append(token)
        else:
            entity_type = tag.split("-")[-1]
            color = COLORS.get(entity_type, "#CCCCCC")
            html_parts.append(
                f'<span style="background-color:{color}; padding:2px 4px; '
                f'border-radius:4px; margin:0 1px;">{token} '
                f'<sub style="font-size:10px;">{entity_type}</sub></span>'
            )
    return " ".join(html_parts)


st.set_page_config(page_title="Disaster Response NER", layout="centered")
st.title("Disaster response entity extraction")
st.write(
    "Paste in a disaster-related report or tweet. The model extracts "
    "locations, damaged infrastructure, casualty mentions, and resource needs."
)

example = (
    "Flooding has cut off access to the hospital on 5th Avenue in Houston, "
    "at least 12 people trapped, urgent need for boats and medical supplies."
)

text_input = st.text_area("Enter text:", value=example, height=100)

if st.button("Extract entities"):
    try:
        tokenizer, model = load_model()
        results = predict(text_input, tokenizer, model)
        st.markdown(render_highlighted(results), unsafe_allow_html=True)

        st.subheader("Extracted entities by type")
        grouped = {}
        current_entity, current_type = [], None
        for token, tag in results:
            if tag.startswith("B-"):
                if current_entity:
                    grouped.setdefault(current_type, []).append(" ".join(current_entity))
                current_entity = [token]
                current_type = tag[2:]
            elif tag.startswith("I-") and current_type == tag[2:]:
                current_entity.append(token)
            else:
                if current_entity:
                    grouped.setdefault(current_type, []).append(" ".join(current_entity))
                current_entity, current_type = [], None
        if current_entity:
            grouped.setdefault(current_type, []).append(" ".join(current_entity))

        for entity_type, mentions in grouped.items():
            st.write(f"**{entity_type}**: {', '.join(mentions)}")

    except OSError:
        st.error(
            "Model not found. Train the model first by running "
            "`python src/train.py` (see README), then re-run this demo."
        )
