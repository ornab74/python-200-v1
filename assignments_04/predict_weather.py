"""Use the saved Week 3 model through the Week 4 component."""
from pathlib import Path
from weather_model import WeatherClassifier


def main() -> None:
    """Print running-weather predictions for four hypothetical days."""
    classifier = WeatherClassifier(Path(__file__).resolve().parent / "models/weather_classifier.pkl")
    days = [
        {"temperature_2m_max": 18, "temperature_2m_min": 9, "precipitation_sum": 0, "wind_speed_10m_max": 10},
        {"temperature_2m_max": -5, "temperature_2m_min": -10, "precipitation_sum": 20, "wind_speed_10m_max": 50},
        {"temperature_2m_max": 26, "temperature_2m_min": 15, "precipitation_sum": 2.9, "wind_speed_10m_max": 29},
        {"temperature_2m_max": 35, "temperature_2m_min": 27, "precipitation_sum": 20, "wind_speed_10m_max": 45},
    ]
    for day, prediction in zip(days, classifier.predict(days)):
        print(f"{day} -> {prediction.label}; probability of good: {prediction.probability:.4f}")


# Without this guard, importing the script would load the model and print example
# predictions immediately, even if the other script only wanted to reuse a helper.
if __name__ == "__main__":
    main()

# Reflection
# 1. A caller might put wind first and high temperature last. With positional
# values, 10 km/h could land in the temperature column while 18 C lands in the
# wind column. Selecting named columns in a fixed order prevents that mix-up.
# 2. The fixture makes and saves its own small model, so tests work without my
# Week 3 file and do not change just because I retrain on a different city.
# The expected good and bad days are clear, making failures easier to explain.
# 3. The component already accepts batches and loads the pipeline once, so a
# database pipeline can reuse one classifier across many calls. Before production,
# I would add checks for nonnumeric, missing, and infinite values so bad records
# produce useful errors before they reach scikit-learn.
