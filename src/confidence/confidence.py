import numpy as np

def compute_confidence(probs, intensity):

    max_prob = np.max(probs, axis=1)

    # Normalize intensity (1–5 → 0–1)
    intensity_norm = intensity / 5.0

    confidence = 0.7 * max_prob + 0.3 * intensity_norm

    uncertain_flag = (confidence < 0.6).astype(int)

    return confidence, uncertain_flag
