from xgboost import XGBClassifier

def train_state_model(X, y):
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        # use_label_encoder=False,
        eval_metric='mlogloss'
    )
    model.fit(X, y)
    return model
