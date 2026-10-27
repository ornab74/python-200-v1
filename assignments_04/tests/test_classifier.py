"""Test the component using a tiny pipeline built independently of Week 3."""
import joblib
import pandas as pd
import pytest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from weather_model import Prediction, WeatherClassifier

FEATURES = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
GOOD = dict(zip(FEATURES, [18, 9, 0, 10]))
BAD = dict(zip(FEATURES, [-5, -10, 20, 50]))


@pytest.fixture
def classifier(tmp_path):
    days = pd.DataFrame([
        [18, 9, 0, 10], [22, 12, 1, 15],
        [-5, -10, 20, 50], [2, -4, 12, 40],
    ], columns=FEATURES)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=1)),
    ])
    pipeline.fit(days, [1, 1, 0, 0])
    path = tmp_path / "tiny_weather.pkl"
    joblib.dump(pipeline, path)
    return WeatherClassifier(path)


def test_good_day(classifier):
    assert classifier.predict([GOOD])[0].label == "good"


def test_bad_day(classifier):
    assert classifier.predict([BAD])[0].label == "skip"


@pytest.mark.parametrize("day", [GOOD, BAD])
def test_probability_range(classifier, day):
    prediction = classifier.predict([day])[0]
    assert 0.0 <= prediction.probability <= 1.0


def test_batch_preserves_order(classifier):
    # Different key order must not change how the model reads each feature.
    reversed_good = dict(reversed(list(GOOD.items())))
    results = classifier.predict([reversed_good, BAD, GOOD])
    assert len(results) == 3
    assert all(isinstance(item, Prediction) for item in results)
    assert [item.label for item in results] == ["good", "skip", "good"]
    assert results[0].probability == results[2].probability


def test_empty_batch(classifier):
    assert classifier.predict([]) == []


def test_missing_feature(classifier):
    incomplete = {key: value for key, value in GOOD.items() if key != "precipitation_sum"}
    with pytest.raises(ValueError, match="precipitation_sum"):
        classifier.predict([GOOD, incomplete])


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Weather model file not found"):
        WeatherClassifier(tmp_path / "missing.pkl")


# I temporarily made every result say "skip". test_good_day failed because
# it expected "good". I restored the label conversion before the final test run.
