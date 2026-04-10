import numpy as np
from src.confidence.confidence import compute_confidence

def predict_test(state_model, int_model, X_test, le):

    state_probs = state_model.predict_proba(X_test)
    state_preds = state_model.predict(X_test)
    state_labels = le.inverse_transform(state_preds)

    intensity_preds = int_model.predict(X_test)

    confidence, uncertain_flag = compute_confidence(state_probs, intensity_preds)

    return state_labels, intensity_preds, confidence, uncertain_flag
