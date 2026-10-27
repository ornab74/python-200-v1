"""Week 4 warmups: classes, dataclasses, and error checks."""
import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest


# Q1
class ModelInfo:
    """Read and keep a model's metadata for repeated access."""

    def __init__(self, path: str | Path) -> None:
        with open(path, encoding="utf-8") as metadata:
            self.data = json.load(metadata)

    def feature_count(self) -> int:
        """Return the number of input features."""
        return len(self.data["features"])


# I load the JSON once so repeated feature_count calls do not reopen and parse it.

# Q2
@dataclass
class DayForecast:
    """A short forecast for one day."""

    date: str
    high_c: float
    will_rain: bool

    def summary(self) -> str:
        """Describe the temperature and whether rain is expected."""
        rain = "rain expected" if self.will_rain else "no rain expected"
        return f"{self.date}: high {self.high_c:.1f}C, {rain}"


# @dataclass generated __init__, so I did not write the field assignments myself.

# Q3
def parse_temperature(text: str) -> float:
    """Convert numeric text to a temperature, with a useful error for bad text."""
    try:
        return float(text)
    except ValueError as error:
        raise ValueError(f"Invalid temperature: {text!r}") from error


def test_parse_temperature_valid():
    assert parse_temperature("24.1") == pytest.approx(24.1)


def test_parse_temperature_bad_input_raises():
    with pytest.raises(ValueError, match="temperature"):
        parse_temperature("hot")


# Without match, the test could pass for an unrelated ValueError with an unhelpful
# message. I want to check that the error actually explains the bad temperature.

if __name__ == "__main__":
    with TemporaryDirectory() as folder:
        path = Path(folder) / "example_metadata.json"
        path.write_text(json.dumps({"features": ["high_c", "rain_mm"]}))
        info = ModelInfo(path)
        print("Feature count:", info.feature_count())
    print(DayForecast("2023-06-15", 24.1, True).summary())
    print(DayForecast("2023-06-16", 20.5, False).summary())
    print("Parsed temperature:", parse_temperature("24.1"))

