import sys
import os

# Fix import path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import pandas as pd
import numpy as np
import pickle

from src.preprocessing.preprocess import load_and_clean
from src.features.feature_engineering import prepare_features
from src.models.predict import predict_test
from src.decision.decision_engine import decide_action
from src.generation.response_generator import generate_response



# Page Config
st.set_page_config(page_title="MindGuide AI", layout="wide")

st.title("MindGuide AI")
st.markdown("Understand → Decide → Guide")


# Load Models
@st.cache_resource
def load_models():
    state_model = pickle.load(open("outputs/models/state_model.pkl", "rb"))
    int_model = pickle.load(open("outputs/models/int_model.pkl", "rb"))
    tfidf = pickle.load(open("outputs/models/tfidf.pkl", "rb"))
    le = pickle.load(open("outputs/models/label_encoder.pkl", "rb"))
    return state_model, int_model, tfidf, le



# UI Inputs
st.markdown("## Journal")
journal_text = st.text_area("Write your thoughts...", height=150)

st.markdown("## Context")

col1, col2, col3 = st.columns(3)

with col1:
    stress_level = st.slider("Stress Level", 1, 5, 3)
    energy_level = st.slider("Energy Level", 1, 5, 3)
    sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 7.0, step=0.5)

with col2:
    time_of_day = st.selectbox("Time of Day", ["morning","afternoon","evening","night"])
    reflection_quality = st.selectbox("Reflection Quality", ["low","medium","high"])
    duration_min = st.number_input("Duration (min)", 1, 60, 15)

with col3:
    previous_day_mood = st.selectbox("Previous Mood", ["happy","sad","neutral","unknown"])
    ambience_type = st.selectbox("Ambience", ["forest","ocean","rain","mountain","cafe"])

submit = st.button("Get Recommendation")


# Prediction Flow
if submit:

    state_model, int_model, tfidf, le = load_models()

    # Create input dataframe
    input_df = pd.DataFrame({
        'journal_text':[journal_text],
        'stress_level':[stress_level],
        'energy_level':[energy_level],
        'sleep_hours':[sleep_hours],
        'time_of_day':[time_of_day],
        'reflection_quality':[reflection_quality],
        'duration_min':[duration_min],
        'previous_day_mood':[previous_day_mood],
        'ambience_type':[ambience_type],
        'duration_min': [duration_min]
    })

    # Load training data for feature alignment
    df_train = pd.read_csv("data/sample_data/sample_data.csv")

    # Prepare features
    X_dummy, X_input, _, meta_cols = prepare_features(df_train.copy(), input_df.copy())

    # Predict
    state_preds, intensity_preds, confidence, uncertain_flag = predict_test(
        state_model, int_model, X_input, le
    )

    pred_state = state_preds[0]
    pred_intensity = float(intensity_preds[0])
    conf = float(confidence[0])
    uncertain = int(uncertain_flag[0])

    # Decision
    time_map = {"morning":0,"afternoon":1,"evening":2,"night":3}

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

    
    
    st.markdown("---")

    st.markdown("## 🧠 Emotional State")
    st.success(f"{pred_state} (Intensity: {pred_intensity:.2f})")

    st.markdown("## 📊 Confidence")
    st.progress(conf)

    if uncertain == 1:
        st.warning("⚠️ Low confidence — reflect more.")

    st.markdown("## 🧭 Recommendation")
    st.write(f"**Action:** {action}")
    st.write(f"**When:** {timing}")

    st.markdown("## 💬 Guidance")
    st.info(message)
