"""Load the saved running-weather pipeline and predict four example days."""
import json
from pathlib import Path
import joblib
import pandas as pd

BASE = Path(__file__).resolve().parent


def main():
    try:
        metadata = json.loads((BASE / "models/weather_classifier_metadata.json").read_text())
        pipeline = joblib.load(BASE / "models/weather_classifier.pkl")
    except FileNotFoundError as error:
        raise SystemExit("Missing saved model or metadata. Run train_classifier.py first.") from error
    print("City:", metadata["city"])
    print("Features:", metadata["features"])
    print("Test accuracy:", metadata["test_accuracy"])
    days = pd.DataFrame([
        [18, 9, 0, 10],
        [35, 27, 20, 45],
        [26, 15, 2.9, 29],
        [6, -2, 0, 12],
    ], columns=metadata["features"], index=["mild and dry", "hot and stormy", "borderline", "cold"])
    labels = pipeline.predict(days)
    good_index = list(pipeline.classes_).index(1)
    probabilities = pipeline.predict_proba(days)[:, good_index]
    for (name, inputs), label, probability in zip(days.iterrows(), labels, probabilities):
        print(f"{name}: {inputs.to_dict()} -> {'good' if label else 'skip'}, P(good)={probability:.4f}")


if __name__ == "__main__":
    main()

# Task 8
# My borderline day (26 C high, 15 C low, 2.9 mm rain, 29 km/h wind)
# got P(good)=0.3333. Two of its three neighbors voted skip, so I would call
# that limited confidence rather than a firm warning. It actually passes the
# label rule, which shows that this learned model can disagree with the rule.
# At 0.52 I would show an uncertain result and let the runner review the
# weather instead of giving a confident recommendation. With this k=3 model,
# votes yield 0, 1/3, 2/3, or 1, so 0.52 is a hypothetical future-model case.
# Running this before training would fail because the model and metadata do not
# exist. The error above tells the person which script to run first.
# In Week 10, I would replace these fixed example days with the daily weather
# records passed into the cloud pipeline, keeping the metadata's column order.
