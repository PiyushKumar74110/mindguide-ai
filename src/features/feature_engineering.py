import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def prepare_features(df, test_df):

    
    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1,2), stop_words='english')

    X_text = tfidf.fit_transform(df['journal_text']).toarray()
    X_text_test = tfidf.transform(test_df['journal_text']).toarray()

    
    df['duration_min'] = df['duration_min'].astype(float)
    test_df['duration_min'] = test_df['duration_min'].astype(float)

    df['time_of_day'] = df['time_of_day'].replace({'early_morning': 'morning'})
    test_df['time_of_day'] = test_df['time_of_day'].replace({'early_morning': 'morning'})

    time_map = {"morning":0, "afternoon":1, "evening":2, "night":3}
    df['time_of_day'] = df['time_of_day'].map(time_map)
    test_df['time_of_day'] = test_df['time_of_day'].map(time_map)

    quality_map = {"low":0, "medium":1, "high":2}
    df['reflection_quality'] = df['reflection_quality'].map(quality_map)
    test_df['reflection_quality'] = test_df['reflection_quality'].map(quality_map)

    
    all_moods = pd.concat([df['previous_day_mood'], test_df['previous_day_mood']])
    mood_map = {v:i for i,v in enumerate(all_moods.unique())}

    df['previous_day_mood'] = df['previous_day_mood'].map(mood_map)
    test_df['previous_day_mood'] = test_df['previous_day_mood'].map(mood_map)

    
    df = pd.get_dummies(df, columns=['ambience_type'])
    test_df = pd.get_dummies(test_df, columns=['ambience_type'])

    
    df['text_length'] = df['journal_text'].apply(len)
    test_df['text_length'] = test_df['journal_text'].apply(len)

    
    meta_cols = [
        'stress_level',
        'energy_level',
        'sleep_hours',
        'time_of_day',
        'previous_day_mood',
        'reflection_quality',
        'duration_min',   
        'text_length'
    ]

    meta_cols += [col for col in df.columns if col.startswith('ambience_type_')]

    
    for col in meta_cols:
        if col not in test_df.columns:
            test_df[col] = 0

    
    test_df = test_df.reindex(columns=df.columns, fill_value=0)

    
    X_meta = df[meta_cols].values
    X_meta_test = test_df[meta_cols].values

    X = np.hstack([X_text, X_meta])
    X_test = np.hstack([X_text_test, X_meta_test])

    return X, X_test, tfidf, meta_cols
