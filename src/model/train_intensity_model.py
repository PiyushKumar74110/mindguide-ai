from xgboost import XGBRegressor

def train_intensity_model(X_train, y_train):
    model = XGBRegressor(
        max_depth=4,
        n_estimators=150,
        learning_rate=0.1,
        random_state=42
    )

    model.fit(X_train,y_train)
    return model
    
