import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

from src.preprocessing.preprocess import load_and_clean
from src.features.feature_engineering import prepare_features
from src.models.train_state_model import train_state_model
from src.models.train_intensity_model import train_intensity_model
from src.models.predict import predict_test


def run_pipeline(train_path, test_path, output_path):

    print("Loading and preprocessing data...")
    df, test_df = load_and_clean(train_path, test_path)

    print("Feature engineering...")
    X, X_test, tfidf, meta_cols = prepare_features(df, test_df)

    print("Preparing targets...")
    le = LabelEncoder()
    y_state = le.fit_transform(df['emotional_state'])
    y_intensity = df['intensity']

    print("Training models...")
    state_model = train_state_model(X, y_state)
    int_model = train_intensity_model(X, y_intensity)

    print("Saving models...")
    pickle.dump(state_model, open("outputs/models/state_model.pkl", "wb"))
    pickle.dump(int_model, open("outputs/models/int_model.pkl", "wb"))
    pickle.dump(tfidf, open("outputs/models/tfidf.pkl", "wb"))
    pickle.dump(le, open("outputs/models/label_encoder.pkl", "wb"))

    print("Predicting on test data...")
    state_preds, intensity_preds, confidence, uncertain_flag = predict_test(
        state_model, int_model, X_test, le
    )

    print("Saving predictions...")
    results = pd.DataFrame({
        "predicted_state": state_preds,
        "predicted_intensity": intensity_preds,
        "confidence": confidence,
        "uncertain_flag": uncertain_flag
    })

    results.to_csv(output_path, index=False)

    print("Sample predictions:")
    print(results.head())

    print("Pipeline successfull!")
