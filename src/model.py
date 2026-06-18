# src/model.py
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score


def train_model(model_name, X_train, y_train):
    """Instantiate and train a model based on its name."""
    if model_name == "XGBoost":
        model = XGBRegressor(
            n_estimators=100, learning_rate=0.1, random_state=42)
    elif model_name == "RandomForest":
        model = RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1)
    elif model_name == "LightGBM":
        model = LGBMRegressor(n_estimators=100, random_state=42)
    else:
        raise ValueError(
            "Please choose 'XGBoost', 'RandomForest', or 'LightGBM'")

    # Train the model
    model.fit(X_train, y_train)
    return model


def get_predictions_and_score(model, X_test, y_test):
    """Predict energy usage and calculate the R-squared score."""
    predictions = model.predict(X_test)
    score = r2_score(y_test, predictions)
    return predictions, score
