

import os
import json
import pandas as pd
from dotenv import load_dotenv
from typing import Dict, Any

from agents import Agent, function_tool

load_dotenv()

@function_tool
def calculate_rate_difference(csv_path: str, target: str, protected: str) -> str:
    try:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file '{csv_path}' not found.")

        df = pd.read_csv(csv_path)

        if df.empty:
            raise ValueError("Dataset is empty.")

        if protected not in df.columns or target not in df.columns:
            raise ValueError(f"Columns '{protected}' or '{target}' not found in dataset.")

        if df[protected].nunique() != 2:
            raise ValueError("Protected attribute must have exactly two unique groups for rate difference.")

        df_clean = df[[protected, target]].dropna()
        group_stats = df_clean.groupby(protected)[target].mean().to_dict()
        groups = list(group_stats.keys())
        group_a, group_b = groups[0], groups[1]

        rate_a = round(group_stats[group_a], 3)
        rate_b = round(group_stats[group_b], 3)
        rate_diff = round(abs(rate_a - rate_b), 3)

        return json.dumps({
            "protected_attribute": protected,
            "group_a": group_a,
            "group_b": group_b,
            "rate_a": rate_a,
            "rate_b": rate_b,
            "rate_difference": rate_diff
        })

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "protected_attribute": protected,
            "rate_difference": None
        })

bias_agent = Agent(
    name="Bias Agent",
    instructions="""
    You are a fairness auditor. Your job is to assess rate difference bias between two groups 
    defined by a protected attribute in a dataset. The goal is to check for outcome disparities 
    in a binary target variable (e.g., loan approval). Use group-wise means to compute the disparity.
    """,
    tools=[calculate_rate_difference],
)


'''

import os
import json
import pandas as pd
from dotenv import load_dotenv
from typing import Dict

from agents import Agent, function_tool

load_dotenv()

@function_tool
def calculate_rate_difference(csv_path: str, target: str, protected: str) -> str:
    try:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file '{csv_path}' not found.")

        df = pd.read_csv(csv_path)

        if df.empty:
            raise ValueError("Dataset is empty.")

        if protected not in df.columns or target not in df.columns:
            raise ValueError(f"Columns '{protected}' or '{target}' not found in dataset.")

        df_clean = df[[protected, target]].dropna()

        if df_clean[protected].nunique() != 2:
            raise ValueError("Protected attribute must have exactly two unique groups for rate difference.")

        group_stats = df_clean.groupby(protected)[target].mean().to_dict()
        groups = list(group_stats.keys())
        group_a, group_b = groups[0], groups[1]

        rate_a = round(group_stats[group_a], 3)
        rate_b = round(group_stats[group_b], 3)
        rate_diff = round(abs(rate_a - rate_b), 3)

        result = {
            "protected_attribute": protected,
            "group_a": group_a,
            "group_b": group_b,
            "rate_a": rate_a,
            "rate_b": rate_b,
            "rate_difference": rate_diff
        }

        return json.dumps(result)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "protected_attribute": protected,
            "rate_difference": None
        })

bias_agent = Agent(
    name="Bias Agent",
    instructions="""
    You are a fairness auditor. Your job is to assess *rate difference bias* between two groups 
    in a protected attribute of a dataset, based on a binary target variable.
    
    You MUST return output as a JSON with these keys:
    - protected_attribute
    - group_a
    - group_b
    - rate_a
    - rate_b
    - rate_difference

    Avoid any plain text explanations. Output only valid JSON.
    """,
    tools=[calculate_rate_difference],
)

'''
'''
import os
import json
import pandas as pd
from dotenv import load_dotenv
from typing import Dict

from agents import Agent, function_tool

load_dotenv()

@function_tool
def calculate_rate_difference(csv_path: str, target: str, protected: str) -> str:
    try:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file '{csv_path}' not found.")

        df = pd.read_csv(csv_path)

        if df.empty:
            raise ValueError("Dataset is empty.")

        if protected not in df.columns or target not in df.columns:
            raise ValueError(f"Columns '{protected}' or '{target}' not found in dataset.")

        df_clean = df[[protected, target]].dropna()

        # 🔹 Ensure target is numeric (important for categorical targets stored as strings like 'Yes/No')
        if not pd.api.types.is_numeric_dtype(df_clean[target]):
            df_clean[target] = df_clean[target].astype("category").cat.codes

        unique_groups = df_clean[protected].unique()
        if len(unique_groups) < 2:
            raise ValueError("Protected attribute must have at least two unique groups for comparison.")

        group_stats = df_clean.groupby(protected)[target].mean().to_dict()

        # 🔹 Take only the first two groups for now (consistent with rate difference definition)
        group_a, group_b = list(group_stats.keys())[:2]

        rate_a = round(group_stats[group_a], 3)
        rate_b = round(group_stats[group_b], 3)
        rate_diff = round(abs(rate_a - rate_b), 3)

        result = {
            "protected_attribute": protected,
            "group_a": str(group_a),
            "group_b": str(group_b),
            "rate_a": rate_a,
            "rate_b": rate_b,
            "rate_difference": rate_diff
        }

        return json.dumps(result)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "protected_attribute": protected,
            "rate_difference": None
        })

bias_agent = Agent(
    name="Bias Agent",
    instructions="""
    You are a fairness auditor. Your job is to assess *rate difference bias* between two groups 
    in a protected attribute of a dataset, based on a binary target variable.
    
    You MUST return output as a JSON with these keys:
    - protected_attribute
    - group_a
    - group_b
    - rate_a
    - rate_b
    - rate_difference

    Avoid any plain text explanations. Output only valid JSON.
    """,
    tools=[calculate_rate_difference],
)

'''