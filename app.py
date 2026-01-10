import streamlit as st
import pandas as pd
import joblib
import re
import numpy as np

# Load bundle (prep + model)
MODEL_PATH = "best_shopee_sentiment_model.joblib"
bundle = joblib.load(MODEL_PATH)
prep = bundle["prep"]
model = bundle["model"]

URL_RE = re.compile(r"http\S+|www\.\S+")
WS_RE = re.compile(r"\s+")
EMOJI_RE = re.compile(r"[\U00010000-\U0010ffff]", flags=re.UNICODE)

def clean_text(t: str) -> str:
    t = (t or "").lower()
    t = URL_RE.sub(" ", t)
    t = t.replace("\n", " ").replace("\r", " ")
    t = WS_RE.sub(" ", t).strip()
    return t

def build_row(text: str, thumbs_up: int, lang: str):
    clean = clean_text(text)
    return {
        "clean_content": clean,
        "thumbsUpCount": int(thumbs_up),
        "text_len": len(clean),
        "word_count": len(clean.split()) if clean else 0,
        "exclamation_count": (text or "").count("!"),
        "question_count": (text or "").count("?"),
        "emoji_count": len(EMOJI_RE.findall(text or "")),
        "has_reply": 0,
        "review_hour": 0,
        "review_dow": 0,
        "lang": lang
    }

st.title("Shopee Review Sentiment Predictor")

text = st.text_area("Paste a review:", height=140)
thumbs = st.number_input("thumbsUpCount (optional)", min_value=0, value=0, step=1)
lang = st.selectbox("Language", ["EN", "ID"])

if st.button("Predict"):
    X_new = pd.DataFrame([build_row(text, thumbs, lang)])
    X_vec = prep.transform(X_new)      # ✅ important line
    pred = model.predict(X_vec)[0]     # ✅ important line
    st.success(f"Predicted sentiment: **{pred}**")
