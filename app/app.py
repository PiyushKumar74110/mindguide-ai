import streamlit as st
import pandas as pd
import numpy as np
import pickle

from src.preprocessing.preprocess import load_and_clean
from src.features.feature_engineering import prepare_features
from src.models.predict import predict_test
from src.decision.decision_engine import decision_action
from src.generation.response_generator import generate_response
from sklearn.preprocessing import LabelEncoder


# Page Config
st.set_page_config(page_title="AI Emotion System", layout="wide")

st.title("AI Emotion-Based Recommendation System")
st.markdown("Understand --> Decide --> Guide")

# Emotion Colors
def get_color(state):
    colors = {
        "stressed": "#ff4d4d",
        "anxious": "#ff944d",
        "sad": "#4da6ff",
        "calm": "#4dff88",
        "relaxed": "#66ffcc"
    }
    return colors.get(state, "#cccccc")


# Input Section
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


# Prediction
if submit:

    # Load models
    state_model = pickle.load(open("outputs/models/state_model.pkl", "rb"))
    int_model = pickle.load(open("outputs/models/int_model.pkl", "rb"))

    # Load training data for encoding consistency
    train_df = pd.read_csv("data/raw/dataset.csv")

    le = LabelEncoder()
    le.fit(train_df['emotional_state'])

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
        'ambience_type':[ambience_type]
    })

    
    # Preprocess + Feature Engineering
    df_clean, _ = load_and_clean("data/raw/dataset.csv", "data/raw/dataset.csv")
    _, X_input, tfidf, meta_cols = prepare_features(df_clean, input_df)

    
    # Predict
    state_label, intensity_pred, confidence, uncertain_flag = predict_test(
        state_model, int_model, X_input, le
    )

    pred_state = state_label[0]
    pred_intensity = float(intensity_pred[0])
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

    # Generate Response
    row = {
        "predicted_state": pred_state,
        "predicted_intensity": pred_intensity,
        "recommended_action": action,
        "recommended_time": timing,
        "uncertain_flag": uncertain
    }

    message = generate_response(row)

    # Display Output
    st.markdown("---")
    st.markdown("## Your State")

    color = get_color(pred_state)

    st.markdown(
        f"""
        <div style="padding:20px;border-radius:10px;background-color:{color};color:white;">
            <h2>{pred_state.upper()}</h2>
            <p>Intensity: {pred_intensity:.1f}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Confidence
    st.markdown("## Confidence")
    st.progress(conf)

    if uncertain == 1:
        st.warning("Low confidence — please reflect on how you feel.")

    # Recommendation
    st.markdown("## Recommendation")

    colA, colB = st.columns(2)
    with colA:
        st.metric("Action", action)
    with colB:
        st.metric("When", timing)

    # Message
    st.markdown("## Guidance")
    st.success(message)
