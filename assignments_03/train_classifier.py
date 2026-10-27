"""Train and save a classifier for daily running weather."""
import json
import platform
from pathlib import Path

import joblib
import pandas as pd
import requests
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
FEATURES = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
CITY = "Charlotte, NC"
LABEL_RULE = "7 <= high_C <= 26; low_C >= 0; precipitation_mm < 3; wind_kmh < 30"


def fetch_weather():
    # Task 1: I chose Charlotte, NC, using all of 2023 and the API's Celsius,
    # millimeter, and km/h units so the assignment's thresholds apply directly.
    response = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
        "latitude": 35.23, "longitude": -80.84,
        "start_date": "2023-01-01", "end_date": "2023-12-31",
        "daily": FEATURES, "timezone": "America/New_York",
        "temperature_unit": "celsius", "precipitation_unit": "mm", "wind_speed_unit": "kmh",
    }, timeout=60)
    response.raise_for_status()
    df = pd.DataFrame(response.json()["daily"])
    df["date"] = pd.to_datetime(df.pop("time"))
    if df[FEATURES].isna().any().any():
        raise ValueError("Weather data contains missing feature values.")
    return df


def label_weather(df):
    # Task 2: I kept all four thresholds exactly as assigned.
    return (df["temperature_2m_max"].between(7, 26)
            & df["temperature_2m_min"].ge(0)
            & df["precipitation_sum"].lt(3)
            & df["wind_speed_10m_max"].lt(30)).astype(int)


def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    scores = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
    }
    print(scores)
    print(classification_report(y_test, predictions, labels=[0, 1], target_names=["skip", "good"], zero_division=0))
    return scores


def main():
    df = fetch_weather()
    print("Weather shape:", df.shape)
    print(df.head().to_string(index=False))
    df["good_for_running"] = label_weather(df)
    print("Class counts (0=skip, 1=good):", df["good_for_running"].value_counts().sort_index().to_dict())
    print(f"Good fraction: {df['good_for_running'].mean():.4f}")
    # There were 139 good days out of 365 (38.08%). That seems plausible
    # for Charlotte: hot summers and rainy days rule out plenty of runs.
    # This is a moderate imbalance; I still check good-class precision and recall
    # because predicting skip every time would already get 61.92% accuracy.
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df["good_for_running"], test_size=0.2, random_state=42,
        stratify=df["good_for_running"],
    )

    # Task 3: Cross-validation sees training rows only, including when scaling.
    cv_scores = {}
    for k in [1, 3, 5, 7, 9, 11, 15, 21]:
        candidate = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=k))])
        cv_scores[k] = cross_val_score(candidate, X_train, y_train, cv=5).mean()
        print(f"k={k}: CV accuracy={cv_scores[k]:.4f}")
    best_k = max(cv_scores, key=cv_scores.get)
    print("Chosen k:", best_k)
    knn = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=best_k))])
    knn.fit(X_train, y_train)
    print("KNN test results:")
    knn_scores = evaluate(knn, X_test, y_test)

    # Task 4
    logreg = Pipeline([("scaler", StandardScaler()), ("logreg", LogisticRegression(max_iter=1000))])
    logreg.fit(X_train, y_train)
    print("Logistic regression test results:")
    logreg_scores = evaluate(logreg, X_test, y_test)
    for feature, coefficient in zip(FEATURES, logreg.named_steps["logreg"].coef_[0]):
        print(f"{feature}: {coefficient:.4f}")
    # Rain (-2.4916) and higher maximum temperature (-0.9343) push toward
    # skip. The weak positive wind coefficient (0.3198) surprised me: it is a
    # learned association conditional on other features, not proof wind helps.
    # A linear boundary cannot fully capture these threshold-based labels.

    # Task 5
    # KNN had higher accuracy: 0.9726 versus logistic regression at 0.8767.
    # I weigh precision heavily: a false positive sends someone out on a bad
    # day; a false negative only misses a good opportunity. KNN precision was
    # 0.9333, compared with 0.8276 for logistic regression, on this test split.
    # I chose KNN (k=3). It keeps training rows, so the model is larger and
    # prediction slower than logistic regression, but 292 rows are small.
    # Logistic regression has clearer coefficients; KNN explains a decision
    # through nearby days. Both offer probabilities, though KNN vote shares
    # are coarse and are not calibrated guarantees. Its better test results
    # make KNN a reasonable choice here; one split is not a safety guarantee.
    chosen = knn
    chosen_scores = knn_scores

    # Task 6: Save the entire pipeline so the scaler travels with the classifier.
    models = BASE / "models"
    models.mkdir(exist_ok=True)
    joblib.dump(chosen, models / "weather_classifier.pkl")
    metadata = {
        "python_version": platform.python_version(), "sklearn_version": sklearn.__version__,
        "features": FEATURES, "label_rule": LABEL_RULE, "city": CITY,
        "date_range": ["2023-01-01", "2023-12-31"],
        "model_type": type(chosen.steps[-1][1]).__name__,
        "test_accuracy": chosen_scores["accuracy"], "test_f1": chosen_scores["f1"],
    }
    (models / "weather_classifier_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Saved models/weather_classifier.pkl and models/weather_classifier_metadata.json")


if __name__ == "__main__":
    main()
