import streamlit as st
import pickle
import numpy as np
import time

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="AI Spam Mail Detector",
    page_icon="📧",
    layout="wide"          # ⭐ FIX: allows full desktop width
)

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    try:
        with open("spam_classifier_pipeline.pkl", "rb") as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        st.error("Model file not found. Ensure 'spam_classifier_pipeline.pkl' is present.")
        return None

model = load_model()

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, .stApp {
    font-family: 'Inter', sans-serif !important;
    color: #1f2937;
}

/* ⭐ STREAMLIT RESPONSIVE WIDTH FIX */
.block-container {
    padding-top: 2rem !important;
    max-width: 1200px !important;   /* expanded width */
    margin: auto !important;
}

/* Background */
.stApp {
    background: radial-gradient(circle at 20% 20%, #e9edff, #f7f9fc, #edf1f7);
}

/* MAIN CARD (Hero Glass Box) */
.main-box {
    background: rgba(255, 255, 255, 0.60);
    border-radius: 24px;
    padding: 3rem 3.2rem;
    margin-top: 20px;
    box-shadow: 0 18px 45px rgba(0,0,0,0.08);
    backdrop-filter: blur(25px);
    border: 1px solid rgba(255, 255, 255, 0.4);
    transition: all 0.35s ease;
    width: 100%;
}

/* Title modern */
.title {
    font-size: 3rem;            /* bigger title */
    font-weight: 800;
    text-align: center;
    margin-top: -90px;
    background: linear-gradient(135deg, #4338ca, #6d28d9, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #64748b;
    font-size: 1.1rem;
    margin-top: -6px;
    margin-bottom: 2.2rem;
}

/* Responsive behavior */
@media (max-width: 768px) {
    .main-box {
        padding: 1.8rem !important;
    }

    .title {
        font-size: 2.2rem !important;
    }
}

/* About Box */
.about-box {
    background: rgba(255,255,255,0.75);
    padding: 1.5rem;
    border-radius: 16px;
    border: 1px solid rgba(99,102,241,0.3);
    box-shadow: 0 4px 16px rgba(99,102,241,0.1);
    color: #1f2937; 
}

textarea {
    border-radius: 14px !important;
    border: 2px solid #d6d8e6 !important;
    background: #fafbff !important;
    color: #1f2937;
}

.stButton > button {
    width: 100%;
    border-radius: 14px;
    background: linear-gradient(135deg,#6366f1,#7c3aed);
    border: none;
    padding: 0.75rem 1rem;
    color: white;
    font-weight: 600;
}

.result {
    padding: 1.4rem;
    border-radius: 16px;
    text-align: center;
    font-size: 1.3rem;
    font-weight: 700;
    margin-top: 1.3rem;
    color: #1f2937;
}

.result-spam {
    background: #ffe5e5;
    color: #b91c1c;
    border: 2px solid #fca5a5;
}

.result-ham {
    background: #e7ffe8;
    color: #166534;
    border: 2px solid #86efac;
}

</style>
""", unsafe_allow_html=True)

# ------------------ MAIN LAYOUT ------------------

st.markdown("<div class='main-box'>", unsafe_allow_html=True)

# Header
st.markdown('<div class="title">📧 AI Spam Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Paste an email below & let AI classify it</div>', unsafe_allow_html=True)

# About Section
st.markdown("""
<div class="about-box">
    <h4>📘 About the Model</h4>
    <p><b>Spam vs Ham Classifier</b></p>
    <p><b>🚨 SPAM:</b> Phishing, scams, promotional links, malicious content.</p>
    <p><b>✅ HAM:</b> Safe & legitimate emails.</p>
</div>
""", unsafe_allow_html=True)

st.write("")

# Email Input
if "email_text" not in st.session_state:
    st.session_state.email_text = ""

st.session_state.email_text = st.text_area(
    "Enter Email Text:",
    value=st.session_state.email_text,
    height=180,
    placeholder="Paste the email content you want to analyze..."
)

# Load Example Buttons
c1, c2 = st.columns(2)
with c1:
    if st.button("📨 Load Example Spam"):
        st.session_state.email_text = (
            "Congratulations! You have WON $5000! Claim now: http://bit.ly/claim-fast"
        )
with c2:
    if st.button("📝 Load Example Ham"):
        st.session_state.email_text = (
            "Hi David, reminder for our meeting tomorrow at 10 AM. Let me know if timing works."
        )

st.write("")

# Analyze Button
if st.button("🔍 Analyze Email"):
    email = st.session_state.email_text.strip()

    if not email:
        st.warning("Please enter some email content.")
    else:
        with st.spinner("Analyzing with AI..."):
            time.sleep(1.2)

            if model:
                prediction = model.predict([email])[0]
                proba = model.predict_proba([email])[0]
            else:
                prediction = 1 if any(k in email.lower() for k in ["win", "prize", "claim"]) else 0
                proba = [0.20, 0.80] if prediction else [0.90, 0.10]

        # Result Output
        if prediction == 1:
            st.markdown('<div class="result result-spam">🚨 SPAM DETECTED</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result result-ham">✅ HAM — SAFE EMAIL</div>', unsafe_allow_html=True)

        # Confidence Levels
        st.subheader("📊 Confidence Levels")
        spam_prob = proba[1] * 100
        ham_prob = proba[0] * 100

        c3, c4 = st.columns(2)
        with c3:
            st.metric("Spam Probability", f"{spam_prob:.1f}%")
            st.progress(spam_prob / 100)
        with c4:
            st.metric("Ham Probability", f"{ham_prob:.1f}%")
            st.progress(ham_prob / 100)

st.markdown("</div>", unsafe_allow_html=True)
