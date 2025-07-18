import streamlit as st
import pickle
import numpy as np

# Load the sentiment pipeline
with open("sentiment_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

# Set page config
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="auto",
)

# --- Custom CSS ---
st.markdown("""
    <style>
    .positive {color: green; font-weight: bold;}
    .negative {color: red; font-weight: bold;}
    .confidence-bar {height: 20px; background-color: #e0e0e0; border-radius: 10px; overflow: hidden;}
    .confidence-fill {height: 100%; background-color: #4CAF50; text-align: right; padding-right: 5px; color: white;}
    </style>
""", unsafe_allow_html=True)

# --- App Title ---
st.title("💬 Sentiment Analyzer")
st.caption("Analyze customer feedback and predict sentiment using XGBoost and TF-IDF")

# --- Input ---
user_input = st.text_area("📝 Enter your review:", height=150)

# --- Predict Button ---
if st.button("🔎 Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        prediction = pipeline.predict([user_input])[0]
        proba = pipeline.predict_proba([user_input])[0][1]
        confidence = round(proba * 100, 2)

        # --- Sentiment Display ---
        if prediction == 1:
            st.markdown(f"### ✅ Sentiment: <span class='positive'>Positive</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"### ❌ Sentiment: <span class='negative'>Negative</span>", unsafe_allow_html=True)

        # --- Confidence Bar ---
        st.markdown("#### 📊 Confidence (Positive):")
        st.markdown(f"""
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {confidence}%;">{confidence:.1f}%</div>
            </div>
        """, unsafe_allow_html=True)

        # --- Confidence Hint ---
        if confidence > 90:
            st.success("Very confident prediction ✅")
        elif confidence > 70:
            st.info("Moderate confidence ℹ️")
        else:
            st.warning("Low confidence — result may not be reliable ⚠️")

# --- Footer ---
st.markdown("---")
st.markdown("""
**About**  
This app uses a trained XGBoost classifier with TF-IDF features to predict whether a review is positive or negative.  
Built with ❤️ using [Streamlit](https://streamlit.io)
""")
