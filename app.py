import streamlit as st
import pickle
import numpy as np
import time

# -----------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------
st.set_page_config(
    page_title="AI Spam Mail Detector",
    page_icon="📧",
    layout="wide"
)

# -----------------------------------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------------------------------
if 'email_text' not in st.session_state:
    st.session_state.email_text = ""

# -----------------------------------------------------------
# LOAD MODEL SAFELY
# -----------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        with open("spam_classifier_pipeline.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("❌ Model file 'spam_classifier_pipeline.pkl' not found. Please ensure it's in the same directory.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()

# -----------------------------------------------------------
# SAFE PROBA FUNCTION
# -----------------------------------------------------------
def get_probabilities(model, text):
    """
    Always returns: [ham_prob, spam_prob]
    Works even if model has no predict_proba().
    """
    if model is None:
        return [0.1, 0.9]  # default fallback

    try:
        # If model supports predict_proba
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([text])[0]
            return proba

        # If model lacks predict_proba -> try decision_function
        if hasattr(model, "decision_function"):
            score = model.decision_function([text])[0]
            # convert score → probability using sigmoid
            spam_prob = 1 / (1 + np.exp(-score))
            ham_prob = 1 - spam_prob
            return [ham_prob, spam_prob]

        # Final fallback
        pred = model.predict([text])[0]
        return [0.1, 0.9] if pred else [0.9, 0.1]
    except Exception as e:
        st.error(f"Error calculating probabilities: {e}")
        return [0.5, 0.5]


# -----------------------------------------------------------
# CUSTOM CSS
# -----------------------------------------------------------
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Global Styles */
html, body, .stApp { 
    font-family: 'Inter', sans-serif !important; 
    color: #000000 !important; 
}

.block-container { 
    padding-top: 2rem !important; 
    max-width: 1200px !important; 
    margin: auto !important; 
}

.stApp { 
    background: #f0f4f8;
}

/* Main Container */
.main-box { 
    background: #ffffff; 
    border-radius: 24px; 
    padding: 3rem 3.2rem; 
    margin-top: 20px; 
    box-shadow: 0 10px 40px rgba(0,0,0,0.1);
    border: 1px solid #e2e8f0;
}

/* Title */
.title { 
    font-size: 3rem; 
    font-weight: 800; 
    text-align: center; 
    margin-top: -90px;
    color: #1e40af;
    margin-bottom: 10px; 
}

/* Subtitle */
.subtitle { 
    text-align: center; 
    color: #334155; 
    font-size: 1.15rem; 
    margin-top: -6px; 
    margin-bottom: 2.2rem;
    font-weight: 500;
}

/* About Box */
.about-box { 
    background: #f8fafc;
    padding: 1.5rem; 
    border-radius: 16px;
    border: 2px solid #cbd5e1;
    color: #0f172a;
    margin-bottom: 1.5rem;
}

.about-box h4 {
    color: #0f172a !important;
    font-weight: 700 !important;
    margin-bottom: 0.5rem;
    font-size: 1.1rem;
}

.about-box p {
    color: #1e293b !important;
    margin: 0.5rem 0;
    font-size: 0.95rem;
    line-height: 1.6;
}

.about-box b {
    color: #000000 !important;
    font-weight: 700;
}

/* Result Boxes */
.result { 
    padding: 1.8rem; 
    border-radius: 16px; 
    text-align: center;
    font-size: 1.5rem; 
    font-weight: 800; 
    margin-top: 1.5rem;
    box-shadow: 0 4px 15px rgba(0,0,0,0.15);
}

.result-spam { 
    background: #fef2f2;
    color: #7f1d1d; 
    border: 3px solid #dc2626;
}

.result-ham { 
    background: #f0fdf4;
    color: #14532d; 
    border: 3px solid #16a34a;
}

/* Streamlit Elements */
.stTextArea label {
    color: #0f172a !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
}

.stTextArea textarea {
    border: 2px solid #94a3b8 !important;
    border-radius: 12px !important;
    color: #000000 !important;
    background: #ffffff !important;
    font-size: 1rem !important;
}

.stTextArea textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.2) !important;
}

.stTextArea textarea::placeholder {
    color: #64748b !important;
}

/* Buttons */
.stButton button {
    border-radius: 12px !important;
    font-weight: 700 !important;
    border: 2px solid transparent !important;
    transition: all 0.3s ease !important;
    color: #000000 !important;
}

.stButton button[kind="primary"] {
    background: #2563eb !important;
    color: #ffffff !important;
    font-size: 1.05rem !important;
}

.stButton button[kind="primary"]:hover {
    background: #1d4ed8 !important;
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(37,99,235,0.4) !important;
}

.stButton button[kind="secondary"] {
    background: #f1f5f9 !important;
    color: #0f172a !important;
    border: 2px solid #cbd5e1 !important;
}

.stButton button[kind="secondary"]:hover {
    background: #e2e8f0 !important;
    border-color: #94a3b8 !important;
}

/* Metrics */
.stMetric {
    background: #f8fafc;
    padding: 1rem;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
}

.stMetric label {
    color: #334155 !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
}

.stMetric [data-testid="stMetricValue"] {
    color: #0f172a !important;
    font-weight: 800 !important;
    font-size: 1.8rem !important;
}

/* Progress bars */
.stProgress > div > div {
    background: #2563eb !important;
}

/* Subheaders */
h2, h3 {
    color: #0f172a !important;
    font-weight: 800 !important;
}

/* Warning/Error boxes */
.stAlert {
    background: #ffffff !important;
    color: #0f172a !important;
    border-radius: 12px !important;
    border: 2px solid #cbd5e1 !important;
}

/* Spinner */
.stSpinner > div {
    border-top-color: #2563eb !important;
}

/* All text elements */
p, span, div {
    color: #0f172a !important;
}
</style>""", unsafe_allow_html=True)

# -----------------------------------------------------------
# MAIN UI
# -----------------------------------------------------------
st.markdown("<div class='main-box'>", unsafe_allow_html=True)
st.markdown("<div class='title'>📧 AI Spam Detection System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Paste an email below & let AI classify it</div>", unsafe_allow_html=True)

# About
st.markdown("""
<div class="about-box">
    <h4>📘 About the Model</h4>
    <p><b>Spam vs Ham Classifier</b></p>
    <p><b>🚨 SPAM:</b> Fraud, phishing, unwanted promotional content.</p>
    <p><b>✅ HAM:</b> Safe & legitimate emails.</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# EXAMPLE BUTTONS
# -----------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    if st.button("📨 Load Example Spam"):
        st.session_state.email_text = "Congratulations! You have WON $5000! Claim now: http://bit.ly/claim-fast"
        st.rerun()
with col2:
    if st.button("📝 Load Example Ham"):
        st.session_state.email_text = "Hi David, reminder for our meeting tomorrow at 10 AM."
        st.rerun()

# -----------------------------------------------------------
# TEXT INPUT
# -----------------------------------------------------------
email_text = st.text_area(
    "Enter Email Text:",
    value=st.session_state.email_text,
    height=180,
    placeholder="Paste the email content you want to analyze...",
    key="text_input"
)

# Update session state when text area changes
if email_text != st.session_state.email_text:
    st.session_state.email_text = email_text

# -----------------------------------------------------------
# ANALYZE BUTTON
# -----------------------------------------------------------
if st.button("🔍 Analyze Email", type="primary"):
    if not email_text.strip():
        st.warning("⚠️ Please enter email content.")
    elif model is None:
        st.error("❌ Model not loaded. Cannot perform analysis.")
    else:
        with st.spinner("Analyzing..."):
            time.sleep(1)

            try:
                # Prediction
                prediction = model.predict([email_text])[0]

                # Probabilities (handles all models safely)
                proba = get_probabilities(model, email_text)
                ham_prob, spam_prob = proba

                # ---------------------------------------------------
                # OUTPUT SECTION
                # ---------------------------------------------------
                if prediction == 1:
                    st.markdown('<div class="result result-spam">🚨 SPAM DETECTED</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result result-ham">✅ HAM — SAFE EMAIL</div>', unsafe_allow_html=True)

                # Confidence
                st.subheader("📊 Confidence Levels")

                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Spam Probability", f"{spam_prob*100:.1f}%")
                    st.progress(spam_prob)
                with c2:
                    st.metric("Ham Probability", f"{ham_prob*100:.1f}%")
                    st.progress(ham_prob)

            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")
                st.info("Please check if the model file is compatible and properly trained.")

st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------
# FOOTER
# -----------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #334155; font-size: 0.9rem; font-weight: 600; padding: 1rem 0;'>"
    "Built with Streamlit 🎈 | Powered by Machine Learning"
    "</div>",
    unsafe_allow_html=True
)