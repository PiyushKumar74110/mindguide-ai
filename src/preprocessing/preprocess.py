import pandas as pd

def load_and_clean(train_path, test_path):
    
    df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Handle missing values
    df['sleep_hours'] = df['sleep_hours'].fillna(df['sleep_hours'].median())
    test_df['sleep_hours'] = test_df['sleep_hours'].fillna(df['sleep_hours'].median())

    df['previous_day_mood'] = df['previous_day_mood'].fillna("unknown")
    test_df['previous_day_mood'] = test_df['previous_day_mood'].fillna("unknown")

    # Drop noisy column
    if 'face_emotion_hint' in df.columns:
        df = df.drop(columns=['face_emotion_hint'])
    if 'face_emotion_hint' in test_df.columns:
        test_df = test_df.drop(columns=['face_emotion_hint'])

    return df, test_df
