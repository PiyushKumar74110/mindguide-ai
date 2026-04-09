from src.preprocessing.preprocess import load_and_clean
from src.features.feature_engineering import prepare_features
from src.models.train_state_model import train_state_model
from src.models.train_intensity_model import train_intensity_model
from src.models.predict import predict_test
from src.decision.decision_engine import decide_action
from src.generation.response_generator import generate_response

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import pandas as pd
import pickle

def run_pipeline(train_path, test_path, output_path):

    df, test_df = load_and_clean(train_path, test_path)

    X, X_test, tfidf, meta_cols = prepare_features(df, test_df)

    le = LabelEncoder()
    y_state = le.fit_transform(df['emotional_state'])
    y_intensity = df['intensity']

    X_train, _, y_state_train, _ = train_test_split(X, y_state, test_size=0.2, random_state=42)
    _, _, y_int_train, _ = train_test_split(X, y_intensity, test_size=0.2, random_state=42)

    state_model = train_state_model(X_train, y_state_train)
    int_model = train_intensity_model(X_train, y_int_train)

    # Save models
    pickle.dump(state_model, open("outputs/models/state_model.pkl","wb"))
    pickle.dump(int_model, open("outputs/models/int_model.pkl","wb"))

    state_labels, int_pred, conf, uncertain = predict_test(state_model, int_model, X_test, le)

    test_df['predicted_state'] = state_labels
    test_df['predicted_intensity'] = int_pred
    test_df['confidence'] = conf
    test_df['uncertain_flag'] = uncertain

    actions, timings = [], []
    for _, row in test_df.iterrows():
        act, t = decide_action(row['predicted_state'], row['predicted_intensity'],
                               row['stress_level'], row['energy_level'], row['time_of_day'])
        actions.append(act)
        timings.append(t)

    test_df['recommended_action'] = actions
    test_df['recommended_time'] = timings

    test_df['recommendation'] = test_df.apply(generate_response, axis=1)

    test_df.to_csv(output_path, index=False)
