import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
)
import joblib

df = pd.read_csv("weather_classification.csv")
numeric = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
X = df[numeric]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

from pathlib import Path

Path("models").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)

# --- Preprocessing ---
# Q1: I fit on training data so test values cannot influence the learned scale.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Scaled training means:", X_train_scaled.mean(axis=0))

# Q2: The four season categories become four indicator columns, one per category.
encoder = OneHotEncoder(sparse_output=False)
seasons_encoded = encoder.fit_transform(df[["season"]])
print("Encoded shape:", seasons_encoded.shape)
print("Season columns:", encoder.get_feature_names_out())

# Q3: The pipeline learns preprocessing on the training set and applies the same
# transformations before every prediction, so I do not have to repeat those steps.
preprocessor = ColumnTransformer([
    ("numeric", StandardScaler(), numeric),
    ("season", OneHotEncoder(handle_unknown="ignore"), ["season"]),
])
season_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("logreg", LogisticRegression(max_iter=1000)),
])
season_features = numeric + ["season"]
season_pipeline.fit(df.loc[X_train.index, season_features], y_train)
print("Season pipeline accuracy:", season_pipeline.score(df.loc[X_test.index, season_features], y_test))

# --- KNN ---
# Q1
knn_pipeline = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=5))])
knn_pipeline.fit(X_train, y_train)
print("Scaled KNN accuracy:", knn_pipeline.score(X_test, y_test))
print(classification_report(y_test, knn_pipeline.predict(X_test), zero_division=0))

# Q2
cv_scores = {}
for k in [1, 3, 5, 7, 9, 11, 15, 21]:
    candidate = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=k))])
    cv_scores[k] = cross_val_score(candidate, X_train, y_train, cv=5).mean()
    print(f"k={k}: CV accuracy={cv_scores[k]:.4f}")
print("Chosen k:", max(cv_scores, key=cv_scores.get))
# I would choose k=11: its mean training CV accuracy was highest, at 0.9417.

# Q3
raw_knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
print("Raw KNN accuracy:", raw_knn.score(X_test, y_test))
print("Scaled KNN accuracy:", knn_pipeline.score(X_test, y_test))
# Scaling helped: accuracy rose from 0.9250 to 0.9583. KNN uses distances,
# so a large numeric range should not give one weather variable extra weight.

# --- Logistic Regression ---
# Q1
pipeline = Pipeline([("scaler", StandardScaler()), ("logreg", LogisticRegression(max_iter=1000))])
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
for name, metric in [("Accuracy", accuracy_score), ("Precision", precision_score),
                     ("Recall", recall_score), ("F1", f1_score)]:
    print(f"Logistic {name}: {metric(y_test, predictions):.4f}")

# Q2
for feature, coefficient in zip(numeric, pipeline.named_steps["logreg"].coef_[0]):
    print(f"{feature}: {coefficient:.4f}")
# Wind (-2.4438) and precipitation (-1.8891) push hardest toward skip.
# That makes sense to me: a windy, wet day is less appealing for a run.
# These coefficients compare one-standard-deviation changes, holding others fixed.

# Q3: Near 0.5, the model has little preference between good and skip.
good_index = list(pipeline.classes_).index(1)
print("First five probabilities:", pipeline.predict_proba(X_test.iloc[:5])[:, good_index].round(2))

# --- Evaluation ---
# Q1
matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
print("Confusion matrix:", matrix)
ConfusionMatrixDisplay(matrix, display_labels=["skip", "good"]).plot()
plt.tight_layout()
plt.savefig("outputs/logreg_confusion_matrix.png")
plt.close()
# There were 9 false positives and 7 false negatives. I would worry more
# about recommending a bad day than missing one good day, so precision matters.

# --- joblib ---
# Q1: Without the scaler, raw inputs would be interpreted as standardized values;
# that would change the decision scores and make the predictions unreliable.
joblib.dump(pipeline, "models/warmup_model.pkl")
loaded = joblib.load("models/warmup_model.pkl")
assert (loaded.predict(X_test) == pipeline.predict(X_test)).all()
print("Predictions match.")
