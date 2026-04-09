from src.confidence.confidence import (
    get_classification_confidence,
    get_regression_confidence,
    combine_confidence,
    get_uncertainity_flag
)

def predict_test(state_model, int_model, X_test, le):

    state_pred = state_model.predict(X_test)
    state_proba = state_model.predict_proba(X_test)
    intensity_pred = int_model.predict(X_test)

    class_conf = get_classification_confidence(state_proba)
    reg_conf = get_regression_confidence(intensity_pred)

    combined_conf = combine_confidence(class_conf, reg_conf)
    uncertain_flag = get_uncertainity_flag(combined_conf)

    state_labels = le.inverse.transform(state_pred)

    return state_labels, intensity_pred, combined_conf, uncertain_flag
