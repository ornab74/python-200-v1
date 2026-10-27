"""A reusable interface for a saved running-weather classifier."""
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

FEATURES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]


@dataclass
class Prediction:
    """A good/skip label and the probability that a day is good for running."""

    label: str
    probability: float


class WeatherClassifier:
    """Load a saved pipeline once and reuse it for batches of weather records.

    Raises:
        FileNotFoundError: If the saved pipeline does not exist.
    """

    def __init__(self, model_path: str | Path) -> None:
        """Load the pipeline, or explain where the missing model was expected.

        Raises:
            FileNotFoundError: If model_path is not an existing file.
        """
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Weather model file not found: {path}. Copy the saved Week 3 model first."
            )
        self.pipeline = joblib.load(path)

    def predict(self, days: list[dict[str, float]]) -> list[Prediction]:
        """Return predictions in input order using the four required features.

        Raises:
            ValueError: If a day is missing any required weather features.
        """
        if not days:
            return []
        for index, day in enumerate(days):
            missing = [feature for feature in FEATURES if feature not in day]
            if missing:
                raise ValueError(f"Day {index} is missing required features: {', '.join(missing)}")
        frame = pd.DataFrame(days)[FEATURES]
        labels = self.pipeline.predict(frame)
        probabilities = self.pipeline.predict_proba(frame)[:, 1]
        return [
            Prediction(label="good" if label == 1 else "skip", probability=float(probability))
            for label, probability in zip(labels, probabilities)
        ]
