

import os
import json
import pandas as pd
from dotenv import load_dotenv
from typing import Dict, Any

from agents import Agent, function_tool

load_dotenv()

@function_tool
def check_inclusion_bias(csv_path: str, protected: str) -> str:
    try:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file '{csv_path}' not found.")

        df = pd.read_csv(csv_path)

        if df.empty:
            raise ValueError("Dataset is empty.")

        if protected not in df.columns:
            raise ValueError(f"Protected attribute '{protected}' not found in dataset.")

        group_counts = df[protected].value_counts(dropna=True).to_dict()
        total = sum(group_counts.values())

        if total == 0 or len(group_counts) < 2:
            raise ValueError("Not enough valid groups for comparison.")

        inclusion_ratio = {
            group: round((count / total) * 100, 2)
            for group, count in group_counts.items()
        }

        min_group = min(inclusion_ratio, key=inclusion_ratio.get)
        max_group = max(inclusion_ratio, key=inclusion_ratio.get)
        gap_percent = round(inclusion_ratio[max_group] - inclusion_ratio[min_group], 2)

        return json.dumps({
            "protected_attribute": protected,
            "group_distribution_percent": inclusion_ratio,
            "least_represented_group": min_group,
            "most_represented_group": max_group,
            "distribution_gap_percent": gap_percent
        })

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "protected_attribute": protected,
            "distribution_gap_percent": None
        })

inclusion_agent = Agent(
    name="Inclusion Agent",
    instructions="""
    You are a fairness evaluator. Your job is to analyze the dataset and assess whether 
    underrepresentation (inclusion bias) exists in a protected attribute.
    Calculate each group's share of the dataset and identify disparities.
    """,
    tools=[check_inclusion_bias],
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
def check_inclusion_bias(csv_path: str, protected: str) -> str:
    try:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file '{csv_path}' not found.")

        df = pd.read_csv(csv_path)

        if df.empty:
            raise ValueError("Dataset is empty.")

        if protected not in df.columns:
            raise ValueError(f"Protected attribute '{protected}' not found in dataset.")

        group_counts = df[protected].value_counts(dropna=True).to_dict()
        total = sum(group_counts.values())

        if total == 0 or len(group_counts) < 2:
            raise ValueError("Not enough valid groups for inclusion bias comparison.")

        inclusion_ratio = {
            str(group): round((count / total) * 100, 2)
            for group, count in group_counts.items()
        }

        min_group = min(inclusion_ratio, key=inclusion_ratio.get)
        max_group = max(inclusion_ratio, key=inclusion_ratio.get)
        gap_percent = round(inclusion_ratio[max_group] - inclusion_ratio[min_group], 2)

        result = {
            "protected_attribute": protected,
            "group_distribution_percent": inclusion_ratio,
            "least_represented_group": min_group,
            "most_represented_group": max_group,
            "distribution_gap_percent": gap_percent
        }

        return json.dumps(result)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "protected_attribute": protected,
            "distribution_gap_percent": None
        })

inclusion_agent = Agent(
    name="Inclusion Agent",
    instructions="""
    You are a fairness evaluator. Your task is to analyze a protected attribute in a dataset and detect
    potential inclusion bias. You must output only valid JSON with the following keys:
    
    - protected_attribute
    - group_distribution_percent
    - least_represented_group
    - most_represented_group
    - distribution_gap_percent

    Do not output plain text or summaries. JSON only.
    """,
    tools=[check_inclusion_bias],
)

'''
'''
# custom_agents/inclusion_agent.py
import os
import json
import pandas as pd
from dotenv import load_dotenv
from typing import Optional, Union

from agents import Agent, function_tool

load_dotenv()

@function_tool
def check_inclusion_bias(
    csv_path: str,
    protected: str,
    bins: Optional[Union[int, str]] = 4,
    method: str = "quantile"
) -> str:
    """
    Inspect group representation for a protected attribute.
    Returns JSON with keys:
      - protected_attribute
      - group_distribution_percent
      - least_represented_group
      - most_represented_group
      - distribution_gap_percent
    Params:
      - bins: number of bins for numeric columns (default 4). If <=1, no binning.
      - method: "quantile" (default) or "uniform".
    """
    try:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file '{csv_path}' not found.")

        df = pd.read_csv(csv_path)
        if df.empty:
            raise ValueError("Dataset is empty.")

        if protected not in df.columns:
            raise ValueError(f"Protected attribute '{protected}' not found in dataset.")

        # normalize bins param (Runner.run might pass strings)
        try:
            bins_int = int(bins) if bins is not None else 4
        except Exception:
            bins_int = 4

        method = str(method or "quantile").lower()
        if method not in {"quantile", "uniform"}:
            method = "quantile"

        series = df[protected]

        # treat missing values explicitly as a group
        series_filled = series.fillna("Missing")

        # numeric handling: optional binning
        if pd.api.types.is_numeric_dtype(series):
            if bins_int and bins_int > 1 and series.nunique(dropna=True) > 1:
                num_bins = min(bins_int, series.nunique(dropna=True))
                try:
                    if method == "uniform":
                        binned = pd.cut(series, bins=num_bins, duplicates="drop")
                    else:  # quantile
                        binned = pd.qcut(series, q=num_bins, duplicates="drop")
                    # convert to string labels and include Missing if any
                    processed = binned.astype(str).fillna("Missing")
                except Exception:
                    # fallback to string values if binning fails
                    processed = series_filled.astype(str)
            else:
                # no binning requested or not enough unique values
                processed = series_filled.astype(str)
        else:
            # categorical / object: use as is (string normalized)
            processed = series_filled.astype(str)

        group_counts = processed.value_counts(dropna=False).to_dict()
        total = sum(group_counts.values())

        if total == 0 or len(group_counts) < 2:
            raise ValueError("Not enough valid groups for inclusion bias comparison.")

        inclusion_ratio = {
            str(group): round((count / total) * 100, 2)
            for group, count in group_counts.items()
        }

        min_group = min(inclusion_ratio, key=inclusion_ratio.get)
        max_group = max(inclusion_ratio, key=inclusion_ratio.get)
        gap_percent = round(inclusion_ratio[max_group] - inclusion_ratio[min_group], 2)

        result = {
            "protected_attribute": protected,
            "group_distribution_percent": inclusion_ratio,
            "least_represented_group": min_group,
            "most_represented_group": max_group,
            "distribution_gap_percent": gap_percent
        }
        return json.dumps(result)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "protected_attribute": protected,
            "distribution_gap_percent": None
        })


inclusion_agent = Agent(
    name="Inclusion Agent",
    instructions="""
    You are a fairness evaluator. Analyze a protected attribute's representation.
    Output MUST be valid JSON with keys:
      - protected_attribute
      - group_distribution_percent
      - least_represented_group
      - most_represented_group
      - distribution_gap_percent
    """,
    tools=[check_inclusion_bias],
)
'''
