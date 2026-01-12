

import os
import json
import pandas as pd
from dotenv import load_dotenv
from typing import Dict, List, Any

from agents import Agent, function_tool

load_dotenv()

@function_tool
def extract_dataset_profile(csv_path: str) -> str:
    try:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file '{csv_path}' not found.")

        df = pd.read_csv(csv_path)

        if df.empty:
            raise ValueError(f"CSV file '{csv_path}' is empty.")

        columns: List[str] = df.columns.tolist()
        if not columns:
            raise ValueError("No columns found in dataset.")

        lowered_cols = [col.lower() for col in columns]
        target_keywords = ['approved', 'label', 'target', 'class', 'decision']
        target = next((col for col in columns if any(key in col.lower() for key in target_keywords)), columns[-1])

        protected: List[str] = []
        for col in columns:
            unique_vals = df[col].nunique(dropna=True)
            if unique_vals <= 10 and df[col].dtype == object:
                protected.append(col)
            elif unique_vals <= 10 and df[col].dtype.name == "category":
                protected.append(col)
            elif "gender" in col.lower() or "race" in col.lower() or "age" in col.lower():
                protected.append(col)

        result = {
            "columns": columns,
            "shape": list(df.shape),  # make it JSON serializable
            "protected": list(set(protected)),
            "target": target
        }
        return json.dumps(result)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "columns": [],
            "shape": [0, 0],
            "protected": [],
            "target": None
        })

data_agent = Agent(
    name="Data Agent",
    instructions="""
    You are a dataset analyzer. Analyze the structure of a CSV dataset,
    identify protected attributes (like gender, race, age) and the target column for fairness assessment.
    Use heuristics such as column names, types, and uniqueness.
    """,
    tools=[extract_dataset_profile],
)



'''
import os
import json
import pandas as pd
from dotenv import load_dotenv
from typing import List

from agents import Agent, function_tool

load_dotenv()

@function_tool
def extract_dataset_profile(csv_path: str) -> str:
    try:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file '{csv_path}' not found.")

        df = pd.read_csv(csv_path)

        if df.empty:
            raise ValueError(f"CSV file '{csv_path}' is empty.")

        columns: List[str] = df.columns.tolist()
        if not columns:
            raise ValueError("No columns found in dataset.")

        target_keywords = ['approved', 'label', 'target', 'class', 'decision']
        target = next((col for col in columns if any(key in col.lower() for key in target_keywords)), columns[-1])

        protected: List[str] = []
        for col in columns:
            unique_vals = df[col].nunique(dropna=True)
            if unique_vals <= 10 and df[col].dtype == object:
                protected.append(col)
            elif unique_vals <= 10 and df[col].dtype.name == "category":
                protected.append(col)
            elif "gender" in col.lower() or "race" in col.lower() or "age" in col.lower():
                protected.append(col)

        result = {
            "columns": columns,
            "shape": list(df.shape),
            "protected": list(set(protected)),
            "target": target
        }
        return json.dumps(result)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "columns": [],
            "shape": [0, 0],
            "protected": [],
            "target": None
        })

data_agent = Agent(
    name="Data Agent",
    instructions="""
    You are a dataset analyzer. Your job is to examine the CSV structure, identify column names,
    detect protected attributes (like gender, race, age), and determine the target column based on keywords.
    
    You MUST return your response strictly as a JSON object containing the following keys:
    - columns (list of column names)
    - shape (list of 2 integers)
    - protected (list of protected attribute column names)
    - target (the target column for fairness analysis)
    """,
    tools=[extract_dataset_profile],
)

'''
'''
import os
import json
import pandas as pd
from dotenv import load_dotenv
from typing import List

from agents import Agent, function_tool

load_dotenv()

@function_tool
def extract_dataset_profile(csv_path: str) -> str:
    try:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file '{csv_path}' not found.")

        df = pd.read_csv(csv_path)

        if df.empty:
            raise ValueError(f"CSV file '{csv_path}' is empty.")

        columns: List[str] = df.columns.tolist()
        if not columns:
            raise ValueError("No columns found in dataset.")

        # Detect target column
        target_keywords = ['approved', 'label', 'target', 'class', 'decision']
        target = next(
            (col for col in columns if any(key in col.lower() for key in target_keywords)),
            None
        )
        if not target:
            # Fallback: pick last column, but only if it's not clearly a protected attribute
            last_col = columns[-1]
            if not any(keyword in last_col.lower() for keyword in ["gender", "race", "age"]):
                target = last_col
            else:
                target = columns[0]  # fallback to first column if last looks protected

        # Detect protected attributes
        protected: List[str] = []
        for col in columns:
            unique_vals = df[col].nunique(dropna=True)
            col_lower = col.lower()

            if unique_vals <= 10:  # likely categorical
                protected.append(col)
            elif df[col].dtype == object or df[col].dtype.name == "category":
                protected.append(col)
            elif any(keyword in col_lower for keyword in ["gender", "race", "age"]):
                protected.append(col)

        result = {
            "columns": columns,
            "shape": list(df.shape),
            "protected": list(set(protected)),
            "target": target
        }
        return json.dumps(result)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "columns": [],
            "shape": [0, 0],
            "protected": [],
            "target": None
        })

data_agent = Agent(
    name="Data Agent",
    instructions="""
    You are a dataset analyzer. Your job is to examine the CSV structure, identify column names,
    detect protected attributes (like gender, race, age, employment status, etc.), and determine the target column based on keywords.
    
    You MUST return your response strictly as a JSON object containing the following keys:
    - columns (list of column names)
    - shape (list of 2 integers)
    - protected (list of protected attribute column names)
    - target (the target column for fairness analysis)
    """,
    tools=[extract_dataset_profile],
)

'''

