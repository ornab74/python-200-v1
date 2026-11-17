"""Week 7 warmups: tool schemas, ReAct agents, and smolagents."""

import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from scipy.stats import pearsonr
from smolagents import CodeAgent, OpenAIServerModel, ToolCallingAgent, tool


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY was not found in .env")
client = OpenAI(api_key=api_key)
BASE_DIR = Path(__file__).resolve().parent
RESOURCES_DIR = BASE_DIR / "resources"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


# --- Lesson 02: Tool Definitions and the ReAct Loop ---
# Q1
def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert a Celsius temperature to Fahrenheit and return it as a formatted string."""
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}°C is {fahrenheit}°F"


celsius_to_fahrenheit_schema = {
    "type": "function",
    "function": {
        "name": "celsius_to_fahrenheit",
        "description": "Convert a Celsius temperature to Fahrenheit.",
        "parameters": {
            "type": "object",
            "properties": {
                "celsius": {
                    "type": "number",
                    "description": "Temperature in degrees Celsius.",
                }
            },
            "required": ["celsius"],
        },
    },
}


def get_current_time() -> str:
    """Return the current local time as a formatted string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


time_schema = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Return the current local time as a string.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

tools = [time_schema]


def run_agent(user_prompt: str) -> str:
    """Run a minimal ReAct-style agent for one prompt."""
    messages = [
        {
            "role": "system",
            "content": "Use an available tool when it is needed. Otherwise answer directly.",
        },
        {"role": "user", "content": user_prompt},
    ]
    first_response = client.chat.completions.create(
        model="gpt-4o-mini", messages=messages, tools=tools, tool_choice="auto"
    )
    first_message = first_response.choices[0].message
    messages.append(first_message)

    if not first_message.tool_calls:
        print("Tool called: none")
        return first_message.content or ""

    for tool_call in first_message.tool_calls:
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments or "{}")
        if function_name == "get_current_time":
            result = get_current_time()
        elif function_name == "celsius_to_fahrenheit":
            result = celsius_to_fahrenheit(**arguments)
        else:
            result = f"Error: unknown tool {function_name}"
        print("Tool called:", function_name)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": result,
            }
        )

    final_response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    return final_response.choices[0].message.content or ""


def run_lesson_02_questions():
    """Run Q1 through Q3."""
    print("Q1 0°C:", celsius_to_fahrenheit(0))
    print("Q1 100°C:", celsius_to_fahrenheit(100))
    print("Q1 -40°C:", celsius_to_fahrenheit(-40))

    # Q2 prediction: the Celsius request will not trigger get_current_time because that
    # tool has no temperature capability. The model should answer directly in one API
    # call. If it unexpectedly requests the time tool, a second API call will be needed.
    result = run_agent("Convert 100 degrees Celsius to Fahrenheit")
    print("Q2 result:", result)
    # The prediction was correct: no tool was called, so the model answered in one API
    # call using its own knowledge.

    # Q3: extend the agent with the temperature tool.
    tools.append(celsius_to_fahrenheit_schema)
    response_a = run_agent("What is 37 degrees Celsius in Fahrenheit?")
    print("Response A:", response_a)
    # The temperature tool was called because the request asked for a conversion that
    # matched its schema.
    response_b = run_agent("What is the boiling point of water in plain English?")
    print("Response B:", response_b)
    # No tool was called. The model treated the boiling point as general knowledge and
    # answered directly without needing a calculation.


# --- Lesson 03: Multi-Tool Agent ---
class CsvManager:
    """Manage one active CSV and expose safe analysis operations."""

    def __init__(self, resources_dir: Path):
        self.resources_dir = resources_dir
        self.df = None
        self.csv_name = None

    def _normalize_csv_name(self, filename: str) -> str:
        return filename if filename.lower().endswith(".csv") else filename + ".csv"

    def _ensure_loaded(self):
        if self.df is None:
            return {"error": "No CSV is loaded."}
        return None

    def list_csv_files(self):
        return {"files": sorted(path.name for path in self.resources_dir.glob("*.csv"))}

    def load_csv(self, filename: str):
        filename = self._normalize_csv_name(filename)
        path = self.resources_dir / filename
        if not path.exists():
            return {"error": f"Could not find {filename}."}
        self.df = pd.read_csv(path)
        self.csv_name = filename
        return {"shape": list(self.df.shape), "columns": self.df.columns.tolist()}

    def get_columns(self):
        error = self._ensure_loaded()
        return error or self.df.columns.tolist()

    def summarize_columns(self, columns=None):
        error = self._ensure_loaded()
        if error:
            return error
        data = self.df if columns is None else self.df[columns]
        return data.describe(include="all").round(3).to_dict()

    def describe_column(self, column: str):
        error = self._ensure_loaded()
        if error:
            return error
        if column not in self.df.columns:
            return {"error": f"Column not found: {column}"}
        return self.df[column].describe().round(4).to_dict()

    def plot_data(self, y: str, x: str = None, plot_type: str = "line"):
        error = self._ensure_loaded()
        if error:
            return error
        if y not in self.df.columns or (x and x not in self.df.columns):
            return {"error": "Requested column was not found."}
        ax = self.df.plot(x=x, y=y, kind=plot_type)
        output = OUTPUTS_DIR / f"{plot_type}_{x or 'index'}_{y}.png"
        ax.figure.savefig(output, bbox_inches="tight")
        plt.close(ax.figure)
        return {"saved_to": str(output)}

    def compute_correlation(self, col1: str, col2: str):
        """Compute Pearson correlation and p-value for two loaded columns."""
        error = self._ensure_loaded()
        if error:
            return error
        missing = [column for column in (col1, col2) if column not in self.df.columns]
        if missing:
            return {"error": f"Columns not found: {missing}"}
        clean = self.df[[col1, col2]].dropna()
        coefficient, p_value = pearsonr(clean[col1], clean[col2])
        return {
            "col1": col1,
            "col2": col2,
            "pearson_r": round(float(coefficient), 4),
            "p_value": round(float(p_value), 4),
        }


csv_manager = CsvManager(RESOURCES_DIR)
node_tools = {
    "list_csv_files": csv_manager.list_csv_files,
    "load_csv": csv_manager.load_csv,
    "get_columns": csv_manager.get_columns,
    "summarize_columns": csv_manager.summarize_columns,
    "describe_column": csv_manager.describe_column,
    "plot_data": csv_manager.plot_data,
    "compute_correlation": csv_manager.compute_correlation,
}


def schema(name, description, properties=None, required=None):
    function = {"name": name, "description": description}
    if properties is not None:
        function["parameters"] = {
            "type": "object",
            "properties": properties,
            "required": required or [],
        }
    return {"type": "function", "function": function}


tools_schema = [
    schema("list_csv_files", "List CSV files."),
    schema("load_csv", "Load a CSV.", {"filename": {"type": "string"}}, ["filename"]),
    schema("get_columns", "Get loaded CSV columns."),
    schema("summarize_columns", "Summarize columns.", {"columns": {"type": "array", "items": {"type": "string"}}}),
    schema("describe_column", "Describe one column.", {"column": {"type": "string"}}, ["column"]),
    schema("plot_data", "Plot loaded data.", {"y": {"type": "string"}, "x": {"type": "string"}, "plot_type": {"type": "string", "enum": ["line", "scatter"]}}, ["y"]),
    schema("compute_correlation", "Compute Pearson correlation and p-value.", {"col1": {"type": "string"}, "col2": {"type": "string"}}, ["col1", "col2"]),
]

SYSTEM_PROMPT = (
    "You are a small data assistant for CSV files stored in resources/. "
    "Use available tools for data work and do not guess. Load a CSV before analysis. "
    "Keep answers short and student-friendly."
)


def run_agent_cycle(messages, user_text, max_tool_rounds=5):
    """Run one complete ReAct cycle while preserving message history."""
    messages.append({"role": "user", "content": user_text})
    for _ in range(max_tool_rounds):
        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, tools=tools_schema, tool_choice="auto"
        )
        message = response.choices[0].message
        assistant_message = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            assistant_message["tool_calls"] = [
                tool_call.model_dump() for tool_call in message.tool_calls
            ]
        messages.append(assistant_message)
        if not message.tool_calls:
            return message.content or ""
        for tool_call in message.tool_calls:
            name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments or "{}")
            result = node_tools[name](**arguments) if name in node_tools else {"error": f"Unknown tool: {name}"}
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": name,
                    "content": json.dumps(result, default=str),
                }
            )
    return "Tool-round limit reached."


# --- Lesson 04: smolagents ---
@tool
def list_csv_files() -> dict:
    """List CSV files available for analysis.

    Returns:
        A dictionary containing the available filenames.
    """
    return csv_manager.list_csv_files()


@tool
def load_csv(filename: str) -> dict:
    """Load a CSV file as the active dataset.

    Args:
        filename: Name of the CSV file to load.

    Returns:
        Dataset shape and column names or an error.
    """
    return csv_manager.load_csv(filename)


@tool
def get_columns() -> list:
    """Get column names for the active CSV.

    Returns:
        A list of column names or an error dictionary.
    """
    return csv_manager.get_columns()


@tool
def summarize_columns(columns: list = None) -> dict:
    """Summarize selected columns or all columns.

    Args:
        columns: Optional list of column names.

    Returns:
        A dictionary of descriptive statistics.
    """
    return csv_manager.summarize_columns(columns)


@tool
def describe_column(column: str) -> dict:
    """Describe one column in the active CSV.

    Args:
        column: Column name to describe.

    Returns:
        A dictionary of descriptive statistics or an error.
    """
    return csv_manager.describe_column(column)


@tool
def plot_data(y: str, x: str = None, plot_type: str = "line") -> dict:
    """Plot columns from the active CSV.

    Args:
        y: Y-axis column name.
        x: Optional X-axis column name.
        plot_type: Either line or scatter.

    Returns:
        The saved plot path or an error.
    """
    return csv_manager.plot_data(y, x, plot_type)


@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute Pearson correlation between two active CSV columns.

    Args:
        col1: First numeric column.
        col2: Second numeric column.

    Returns:
        Column names with Pearson r and p-value or an error.
    """
    return csv_manager.compute_correlation(col1, col2)


TOOLS = [list_csv_files, load_csv, get_columns, summarize_columns, describe_column, plot_data, compute_correlation]


def run_lesson_03_and_04_questions():
    """Run Q4 through Q8."""
    # Q5
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    result = run_agent_cycle(
        messages,
        "Load bike_commute.csv and compute the correlation between avg_traffic_density and avg_speed_kmh.",
    )
    print("Q5 final response:", result)

    # Q6 roles: system gives behavior rules; user supplies the task; assistant records
    # reasoning, tool requests, and final answers; tool contains observed function output.
    print("Q6 full messages:")
    print(json.dumps(messages, indent=2, default=str))

    # Q7
    print("Q7 generated description:", compute_correlation.description)
    # smolagents creates tool metadata from the name, type hints, and docstring. Unlike
    # the manual schema, it removes most JSON boilerplate. The developer must still give
    # precise types and a complete docstring with meaningful Args and Returns sections.

    model = OpenAIServerModel(api_key=api_key, model_id="gpt-4o-mini")
    tool_agent = ToolCallingAgent(tools=TOOLS, model=model, instructions=SYSTEM_PROMPT, max_steps=8)
    code_agent = CodeAgent(
        tools=TOOLS,
        model=model,
        instructions=(
            SYSTEM_PROMPT
            + " Write Python when a requested plot style cannot be produced by a tool. "
            + "The active DataFrame is csv_manager.df."
        ),
        additional_authorized_imports=["pandas", "matplotlib", "matplotlib.pyplot", "numpy"],
        max_steps=8,
    )
    # Q8
    prompt = "Load bike_commute.csv. Plot avg_heart_rate vs duration_min as a scatter plot with green dots."
    response_tool = tool_agent.run(prompt)
    response_code = code_agent.run(prompt, additional_args={"csv_manager": csv_manager})
    print("Q8 ToolCallingAgent response:", response_tool)
    print("Q8 CodeAgent response:", response_code)
    # Both agents called the fixed plot_data tool and created the same default-color plot.
    # Neither agent changed the dots to green. The ToolCallingAgent was limited by the
    # tool schema. The CodeAgent was capable of writing custom matplotlib code but chose
    # not to do so. This shows that code agents offer more flexibility but do not guarantee
    # that the model will use it. Tool agents remain a better fit for stable, bounded work.


def main():
    run_lesson_02_questions()
    run_lesson_03_and_04_questions()


if __name__ == "__main__":
    main()


# Q9
# A ToolCallingAgent is better for a customer-support workflow that may only look up an
# order, issue a predefined refund, or open a ticket. The small allowlist makes the task
# predictable and limits side effects. A CodeAgent introduces the distinct risk of running
# generated code. That code could access unintended files, consume excessive resources, or
# perform an unsafe operation beyond the fixed capabilities of a ToolCallingAgent.
