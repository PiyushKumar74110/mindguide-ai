from xgboost import XGBClassifier

def train_state_model(X_train, y_train):
    model = XGBClassifier(
        max_depth=4,
        n_estimators=150,
        learning_rate = 0.1,
        use_label_encoder = False,
        eval_METRIC = 'mlogloss',
        random_state = 42
    )

    model.fit(X_train, y_train)
    return model
