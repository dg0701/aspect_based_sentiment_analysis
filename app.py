# paste your streamlit code here

import streamlit as st

# IMPORT ONLY PIPELINE (no logic here)
from inference_absa import get_predictions   # use your exact function name

st.set_page_config(page_title="ABSA Dashboard", layout="wide")

st.title("📊 Aspect-Based Sentiment Analysis")

# -------- INPUT -------- #
review_text = st.text_area(
    "Review Input",
    placeholder="Type a Hinglish or English review..."
)

# -------- BUTTON -------- #
if st.button("Analyze"):
    if review_text:

        # 🔴 DIRECT PIPELINE CALL (NO EXTRA LOGIC)
        pos, neg = get_predictions(review_text.lower())

        # -------- DISPLAY -------- #
        col1, col2 = st.columns(2)

        with col1:
            st.success("### Positive Aspects")
            for p in pos:
                st.write(f"✅ {p}")

        with col2:
            st.error("### Negative Aspects")
            for n in neg:
                st.write(f"❌ {n}")
