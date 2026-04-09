import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def prepare_features(df, test_df):
    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1,2), stop_words='english')
    X_text = tfidf.fit_transform(df['journal_text'].toarray())
    X_text_test = tfidf.fit_transform(test_df['journal_text']).toarray()

    meta_cols = ['stress_level', 'energy_level', 'sleep_hours', 'time_of_day', 'previous_day_mood', 'reflection_quality', 'duration_min']
    meta_cols+=[col for col in df.columns if col.startswith('ambience_type_')]

    df['text_length'] = df['journal_text'].apply(len)
    test_df['text_length'] = test_df['journal_text'].apply(len)
    meta_cols.append('text_length')

    for col in meta_cols:
        if col not in test_df.columns:
            test_df[col] = 0

    X_meta = df[meta_cols].values
    X_meta_test = test_df[meta_cols].values

    X = np.hstack([X_text, X_meta])
    X_test = np.hstack([X_text_test, X_meta_test])

    return X, X_test, tfidf, meta_cols


