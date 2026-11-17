"""Week 7 project: a World Happiness data-analysis agent."""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import pearsonr
from smolagents import CodeAgent, OpenAIServerModel, tool


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY was not found in .env")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "happiness_project"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)
df = None

COLUMN_NAMES = {
    "Country": "country",
    "Regional indicator": "region",
    "Happiness score": "happiness_score",
    "Ladder score": "happiness_score",
    "GDP per capita": "gdp_per_capita",
}


# --- Task 1: Define Tools ---
@tool
def load_happiness_data() -> dict:
    """Load and merge all yearly World Happiness CSV files.

    Returns:
        A dictionary containing the merged shape and column names. Returns an error
        dictionary if no yearly files are available.
    """
    global df
    frames = []
    for path in sorted(DATA_DIR.glob("world_happiness_*.csv")):
        year_df = pd.read_csv(path, sep=";", decimal=",")
        year_df = year_df.rename(columns=COLUMN_NAMES)
        year_df["year"] = int(path.stem.split("_")[-1])
        frames.append(year_df)
    if not frames:
        return {"error": f"No happiness files found in {DATA_DIR}"}
    df = pd.concat(frames, ignore_index=True)
    required = {"country", "region", "happiness_score", "gdp_per_capita", "year"}
    missing = sorted(required - set(df.columns))
    if missing:
        df = None
        return {"error": f"Required columns are missing: {missing}"}
    return {"shape": list(df.shape), "columns": df.columns.tolist()}


@tool
def summarize_column(column: str) -> dict:
    """Return descriptive statistics for one loaded column.

    Args:
        column: Name of the column to summarize.

    Returns:
        Statistics from pandas describe or an error dictionary.
    """
    if df is None:
        return {"error": "No data is loaded. Call load_happiness_data first."}
    if column not in df.columns:
        return {"error": f"Column not found: {column}"}
    return df[column].describe().to_dict()


@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute Pearson correlation and p-value for two numeric columns.

    Args:
        col1: First numeric column name.
        col2: Second numeric column name.

    Returns:
        Column names with Pearson r and p-value rounded to four decimals or an error.
    """
    if df is None:
        return {"error": "No data is loaded. Call load_happiness_data first."}
    missing = [column for column in (col1, col2) if column not in df.columns]
    if missing:
        return {"error": f"Columns not found: {missing}"}
    clean = df[[col1, col2]].dropna()
    try:
        coefficient, p_value = pearsonr(clean[col1], clean[col2])
    except (TypeError, ValueError) as error:
        return {"error": str(error)}
    return {
        "col1": col1,
        "col2": col2,
        "pearson_r": round(float(coefficient), 4),
        "p_value": round(float(p_value), 4),
    }


@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> list[dict] | dict:
    """Return the top countries for one column and year.

    Args:
        column: Numeric column used to rank countries.
        year: Dataset year to filter.
        n: Number of countries to return.

    Returns:
        A list of country and value dictionaries, or an error dictionary.
    """
    if df is None:
        return {"error": "No data is loaded. Call load_happiness_data first."}
    if column not in df.columns:
        return {"error": f"Column not found: {column}"}
    if n < 1:
        return {"error": "n must be at least 1"}
    filtered = df[df["year"] == year]
    if filtered.empty:
        return {"error": f"No data found for year {year}"}
    return filtered.nlargest(n, column)[["country", column]].to_dict("records")


# --- Task 2: Build the Agent ---
model = OpenAIServerModel(api_key=api_key, model_id="gpt-4o-mini")
SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.
Use the available tools for loading data, summarizing columns, computing correlations,
and ranking countries. Write Python directly only when the tools are not sufficient.
Be concise, verify results, and save requested plots to the exact requested path.
"""

agent = CodeAgent(
    tools=[load_happiness_data, summarize_column, compute_correlation, get_top_n_countries],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=["pandas", "matplotlib", "matplotlib.pyplot", "scipy.stats"],
    max_steps=8,
)


def main():
    """Run the five guided queries and two original questions."""
    # Task 3: Guided Queries
    queries = [
        "Load the happiness data and tell me its shape and column names.",
        "Summarize the happiness_score column.",
        "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
        "Show me the top 5 happiest countries in 2020.",
        (
            "Plot happiness_score over the years as a line chart, with one line per region. "
            f"Save the plot to {OUTPUTS_DIR / 'happiness_by_region.png'}."
        ),
    ]
    for query in queries:
        print(f"\n--- Query: {query} ---")
        response = agent.run(query, reset=False, additional_args={"df": df})
        print(response)

    expected_plot = OUTPUTS_DIR / "happiness_by_region.png"
    print("Required plot exists:", expected_plot.exists())

    # Task 4: My Questions
    my_query_1 = "Which five countries had the highest gdp_per_capita in 2024?"
    response_1 = agent.run(my_query_1, reset=False, additional_args={"df": df})
    print("\nMy query 1 response:", response_1)
    # Actual behavior: the agent generated pandas ranking code for this query.

    my_query_2 = (
        "Use Python to calculate the average happiness_score by region for 2024. "
        "Report the three highest regional averages."
    )
    response_2 = agent.run(my_query_2, reset=False, additional_args={"df": df})
    print("\nMy query 2 response:", response_2)
    # Actual behavior: the agent generated pandas groupby code because no provided tool
    # computes grouped regional means.


if __name__ == "__main__":
    main()


# --- Reflection ---
#
# 1. Query 3 used compute_correlation and returned Pearson r = 0.6313 with p = 0.0.
# This is a moderately strong positive relationship and is statistically significant.
#
# 2. The regional line-chart query was the most surprising. The agent's first generated
# code block had a parsing error. It recovered on its next attempt, wrote the groupby and
# matplotlib code, and saved the required plot successfully.
#
# 3. A useful additional tool would validate dataset quality. It would report missing values,
# duplicates, unexpected types, and outliers by year. That would help answer whether an
# analysis is trustworthy before the agent interprets correlations or rankings.
