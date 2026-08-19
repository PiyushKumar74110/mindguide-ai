import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")

sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, SRC_DIR)

import streamlit as st
import pandas as pd
import pickle

from preprocessing.preprocess import load_and_clean
from features.feature_engineering import prepare_features
from model.predict import predict_test
from decision.decision_engine import decide_action
from generation.response_generator import generate_response


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="MindGuide AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background-color: #f7f8fc;
    }

    /* Header */
    .main-header {
        padding: 25px 0 10px 0;
    }

    .main-title {
        font-size: 42px;
        font-weight: 700;
        color: #202124;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 17px;
        color: #6b7280;
        margin-top: 5px;
    }

    /* Cards */
    .card {
        background: white;
        padding: 22px;
        border-radius: 15px;
        box-shadow: 0 3px 12px rgba(0,0,0,0.06);
        border: 1px solid #eeeeee;
        height: 100%;
    }

    .card-title {
        font-size: 14px;
        color: #6b7280;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .card-value {
        font-size: 28px;
        font-weight: 700;
        color: #202124;
        margin-top: 8px;
    }

    .recommendation-card {
        background: white;
        padding: 28px;
        border-radius: 15px;
        box-shadow: 0 3px 12px rgba(0,0,0,0.06);
        border-left: 5px solid #6366f1;
    }

    .guidance-card {
        background: #eef2ff;
        padding: 25px;
        border-radius: 15px;
        margin-top: 15px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 45px;
        font-weight: 600;
    }

    /* Text area */
    textarea {
        border-radius: 12px !important;
    }

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
<div class="main-header">
    <div class="main-title">🧠 MindGuide AI</div>
    <div class="subtitle">
        Understand your state → Make better decisions → Get personalized guidance
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")


# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

@st.cache_resource
def load_models():

    state_model = pickle.load(
        open("outputs/models/state_model.pkl", "rb")
    )

    int_model = pickle.load(
        open("outputs/models/int_model.pkl", "rb")
    )

    tfidf = pickle.load(
        open("outputs/models/tfidf.pkl", "rb")
    )

    le = pickle.load(
        open("outputs/models/label_encoder.pkl", "rb")
    )

    return state_model, int_model, tfidf, le


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown("## ⚙️ Your Context")

    st.markdown("### Current State")

    stress_level = st.slider(
        "Stress Level",
        1, 5, 3,
        help="1 = Very Low, 5 = Very High"
    )

    energy_level = st.slider(
        "Energy Level",
        1, 5, 3,
        help="1 = Very Low, 5 = Very High"
    )

    sleep_hours = st.slider(
        "Sleep Hours",
        0.0, 12.0, 7.0,
        step=0.5
    )

    st.markdown("### Environment")

    time_of_day = st.selectbox(
        "Time of Day",
        ["morning", "afternoon", "evening", "night"]
    )

    ambience_type = st.selectbox(
        "Ambience",
        ["forest", "ocean", "rain", "mountain", "cafe"]
    )

    st.markdown("### Reflection")

    reflection_quality = st.selectbox(
        "Reflection Quality",
        ["low", "medium", "high"]
    )

    previous_day_mood = st.selectbox(
        "Previous Mood",
        ["happy", "sad", "neutral", "unknown"]
    )

    duration_min = st.number_input(
        "Duration (minutes)",
        min_value=1,
        max_value=60,
        value=15
    )


# --------------------------------------------------
# JOURNAL
# --------------------------------------------------

st.markdown("## ✍️ How are you feeling?")

st.markdown(
    "Write freely about your thoughts, feelings, or what happened today."
)

journal_text = st.text_area(
    "Journal",
    placeholder="Example: I had a busy day and feel a little stressed about tomorrow...",
    height=180,
    label_visibility="collapsed"
)

st.markdown("")


# --------------------------------------------------
# SUBMIT
# --------------------------------------------------

submit = st.button(
    "✨ Analyze & Get Recommendation",
    type="primary"
)


# --------------------------------------------------
# PREDICTION FLOW
# --------------------------------------------------

if submit:

    if not journal_text.strip():

        st.warning(
            "Please write something in your journal before getting a recommendation."
        )

        st.stop()

    with st.spinner("Understanding your current state..."):

        # Load models
        state_model, int_model, tfidf, le = load_models()

        # Create input dataframe
        input_df = pd.DataFrame({
            "journal_text": [journal_text],
            "stress_level": [stress_level],
            "energy_level": [energy_level],
            "sleep_hours": [sleep_hours],
            "time_of_day": [time_of_day],
            "reflection_quality": [reflection_quality],
            "duration_min": [duration_min],
            "previous_day_mood": [previous_day_mood],
            "ambience_type": [ambience_type]
        })

        # Load training data
        df_train = pd.read_csv(
            "data/sample_data/sample_data.csv"
        )

        # Prepare features
        X_dummy, X_input, _, meta_cols = prepare_features(
            df_train.copy(),
            input_df.copy()
        )

        # Prediction
        state_preds, intensity_preds, confidence, uncertain_flag = predict_test(
            state_model,
            int_model,
            X_input,
            le
        )

        pred_state = state_preds[0]

        pred_intensity = float(
            intensity_preds[0]
        )

        conf = float(
            confidence[0]
        )

        uncertain = int(
            uncertain_flag[0]
        )

        # Decision
        time_map = {
            "morning": 0,
            "afternoon": 1,
            "evening": 2,
            "night": 3
        }

        action, timing = decide_action(
            pred_state,
            pred_intensity,
            stress_level,
            energy_level,
            time_map[time_of_day]
        )

        # Response
        row = {
            "predicted_state": pred_state,
            "predicted_intensity": pred_intensity,
            "recommended_action": action,
            "recommended_time": timing,
            "uncertain_flag": uncertain
        }

        message = generate_response(row)


    # --------------------------------------------------
    # RESULTS
    # --------------------------------------------------

    st.markdown("---")

    st.markdown("## 📊 Your MindGuide Summary")

    col1, col2, col3 = st.columns(3)

    # Emotional State
    with col1:

        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">Emotional State</div>
                <div class="card-value">🧠 {pred_state}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Intensity
    with col2:

        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">Intensity</div>
                <div class="card-value">{pred_intensity:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Confidence
    with col3:

        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">Confidence</div>
                <div class="card-value">{conf * 100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("")


    # --------------------------------------------------
    # CONFIDENCE BAR
    # --------------------------------------------------

    st.markdown("### Model Confidence")

    st.progress(
        min(max(conf, 0.0), 1.0)
    )

    if uncertain == 1:

        st.warning(
            "⚠️ The model is not highly confident about this prediction. "
            "Consider reflecting more about your current feelings."
        )


    # --------------------------------------------------
    # RECOMMENDATION
    # --------------------------------------------------

    st.markdown("## 🎯 Personalized Recommendation")

    st.markdown(
        f"""
        <div class="recommendation-card">

            <h3>Recommended Action</h3>

            <p style="font-size:20px;">
                <b>{action}</b>
            </p>

            <hr>

            <p>
                <b>⏰ Recommended Time:</b> {timing}
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------
    # GUIDANCE
    # --------------------------------------------------

    st.markdown("## 💡 MindGuide")

    st.markdown(
        f"""
        <div class="guidance-card">

            <h4>Personalized Guidance</h4>

            <p style="font-size:17px; line-height:1.6;">
                {message}
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------
    # INPUT SUMMARY
    # --------------------------------------------------

    with st.expander("🔍 View your input summary"):

        summary_col1, summary_col2 = st.columns(2)

        with summary_col1:

            st.write(
                f"**Stress:** {stress_level}/5"
            )

            st.write(
                f"**Energy:** {energy_level}/5"
            )

            st.write(
                f"**Sleep:** {sleep_hours} hours"
            )

            st.write(
                f"**Previous Mood:** {previous_day_mood}"
            )

        with summary_col2:

            st.write(
                f"**Time:** {time_of_day}"
            )

            st.write(
                f"**Reflection Quality:** {reflection_quality}"
            )

            st.write(
                f"**Duration:** {duration_min} min"
            )

            st.write(
                f"**Ambience:** {ambience_type}"
            )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.caption(
    "MindGuide AI • Understand → Decide → Guide"
)