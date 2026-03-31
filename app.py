"""
Amazon Review Analyzer — Streamlit App
Predicts Buy / Caution / Don't Buy based on user-provided reviews.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from model.predict import predict_reviews, get_explanation

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Amazon Review Analyzer",
    page_icon="🛒",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .verdict-buy       { background:#d4edda; color:#155724; border-radius:8px; padding:16px 24px; font-size:22px; font-weight:600; text-align:center; }
    .verdict-caution   { background:#fff3cd; color:#856404; border-radius:8px; padding:16px 24px; font-size:22px; font-weight:600; text-align:center; }
    .verdict-dontbuy   { background:#f8d7da; color:#721c24; border-radius:8px; padding:16px 24px; font-size:22px; font-weight:600; text-align:center; }
    .metric-card       { background:#f8f9fa; border-radius:8px; padding:12px 16px; margin:4px 0; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🛒 Amazon Review Analyzer")
st.markdown("Paste one or more Amazon product reviews below and get an AI-powered **Buy / Caution / Don't Buy** verdict.")
st.divider()

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = "model/artifacts/pipeline.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("⚠️ Model not found. Run `python model/train.py` first to train and save the model.")
        return None
    return joblib.load(MODEL_PATH)

model = load_model()

# ── Input section ─────────────────────────────────────────────────────────────
st.subheader("📝 Enter Reviews")

col1, col2 = st.columns([3, 1])
with col1:
    reviews_input = st.text_area(
        "Paste reviews (one per line, or a single block of text):",
        height=200,
        placeholder="Example:\nThe product arrived damaged and the quality is terrible. Complete waste of money.\nAmazing product! Works exactly as described. Highly recommend to everyone.\nDecent product but the price is a bit high for what you get.",
    )
with col2:
    st.markdown("**Tips:**")
    st.markdown("- Paste multiple reviews separated by new lines")
    st.markdown("- Include at least 3–5 reviews for best accuracy")
    st.markdown("- Longer reviews give more reliable results")

avg_rating = st.slider("Average star rating of the product (if known):", 1.0, 5.0, 3.5, 0.1)
num_reviews = st.number_input("Total number of reviews on the product:", min_value=1, value=100, step=10)

st.divider()

# ── Analyze button ────────────────────────────────────────────────────────────
if st.button("🔍 Analyze Reviews", use_container_width=True, type="primary"):
    if not reviews_input.strip():
        st.warning("Please paste at least one review before analyzing.")
    elif model is None:
        st.error("Model not loaded. Please train the model first.")
    else:
        with st.spinner("Analyzing reviews..."):
            reviews_list = [r.strip() for r in reviews_input.strip().split("\n") if r.strip()]
            result = predict_reviews(model, reviews_list, avg_rating, num_reviews)

        # ── Verdict banner ────────────────────────────────────────────────────
        verdict = result["verdict"]
        confidence = result["confidence"]

        if verdict == "Buy":
            st.markdown(f'<div class="verdict-buy">✅ Verdict: BUY — {confidence:.0%} confidence</div>', unsafe_allow_html=True)
        elif verdict == "Caution":
            st.markdown(f'<div class="verdict-caution">⚠️ Verdict: CAUTION — {confidence:.0%} confidence</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="verdict-dontbuy">❌ Verdict: DON\'T BUY — {confidence:.0%} confidence</div>', unsafe_allow_html=True)

        st.markdown("")

        # ── Confidence bars ───────────────────────────────────────────────────
        st.subheader("Confidence Breakdown")
        probs = result["probabilities"]
        for label, prob in probs.items():
            icon = {"Buy": "✅", "Caution": "⚠️", "Don't Buy": "❌"}[label]
            st.markdown(f"**{icon} {label}**")
            st.progress(prob, text=f"{prob:.1%}")

        st.divider()

        # ── Metrics ───────────────────────────────────────────────────────────
        st.subheader("📊 Review Analytics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Reviews analyzed", len(reviews_list))
        m2.metric("Avg sentiment", f"{result['avg_sentiment']:+.2f}")
        m3.metric("Positive reviews", f"{result['pct_positive']:.0%}")
        m4.metric("Avg star rating", f"{avg_rating:.1f} ⭐")

        st.divider()

        # ── Explanation ───────────────────────────────────────────────────────
        st.subheader("💡 Why this verdict?")
        explanation = get_explanation(result)
        for point in explanation:
            st.markdown(f"- {point}")

        # ── Per-review breakdown ──────────────────────────────────────────────
        with st.expander("See per-review sentiment breakdown"):
            df = pd.DataFrame({
                "Review (truncated)": [r[:120] + "..." if len(r) > 120 else r for r in reviews_list],
                "Sentiment score": result["individual_sentiments"],
                "Label": result["individual_labels"],
            })
            st.dataframe(df, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center;color:gray;font-size:12px;'>Built for Fundamentals of AI & ML · BYOP Project</div>",
    unsafe_allow_html=True,
)
