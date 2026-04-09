import pandas as pd
def load_and_clean(train_path, test_path):
    df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    df['sleep_hours'] = df['sleep_hours'].fillna(df['sleep_hours'].median())
    df['previous_day_mood'] = df['previous_day_mood'].fillna("unknown")
    df = df.drop(columns=['face_emotion_hint'])

    test_df['sleep_hours'] = test_df['sleep_hours'].fillna(test_df['sleep_hours'].median())
    test_df['previous_day_mood'] = test_df['previous_day_mood'].fillna("unknown")
    test_df = test_df.drop(columns=['face_emotion_hint'])

    df['time_of_day'] = df['time_of_day'].replace({"early_morning":"morning"})

    test_df['time_of_day'] = test_df['time_of_day'].replace({"early_morning":"morning"})

    time_map = {
        "morning":0,
        "afternoon":1,
        "evening":2,
        "night":3
    }

    quality_map = {
        "low":0,
        "medium":1,
        "high":2
    }

    for dataset in [df,test_df]:
        dataset['time_of_day'] = dataset['time_of_day'].map(time_map)
        dataset['reflection_quality'] = dataset['reflection_quality'].map(quality_map)
        dataset['previous_day_mood'] = dataset['previous_day_mood'].astype('category').cat.codes

    df = pd.get_dummies(df,columns=['ambience_type'])
    test_df = pd.get_dummies(test_df, columns=['ambience_type'])

    return df, test_df




