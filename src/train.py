# src/train.py
from sklearn.model_selection import train_test_split
# or whatever model you're using
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

import src.config as config
from src.preprocess import load_and_clean_data, add_new_features


def get_predictions_and_score(model, X_test, y_test):
    """Predict energy usage and calculate the R-squared score."""
    predictions = model.predict(X_test)
    score = r2_score(y_test, predictions)
    return predictions, score


def main():
    # 1. Load and clean
    df = load_and_clean_data()

    # 2. Feature engineering
    df = add_new_features(df)

    # 3. Select features (X) and target (y)
    X = df[config.FEATURES]
    y = df[config.TARGET]

    # 4. Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5. Train model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # 6. Evaluate
    predictions, score = get_predictions_and_score(model, X_test, y_test)
    print("R² score:", score)


if __name__ == "__main__":
    main()
