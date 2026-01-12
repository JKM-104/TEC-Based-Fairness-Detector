

import json
import pandas as pd
from agents import Agent, function_tool

@function_tool
def analyze_learning_bias(csv_path: str, target: str, prediction: str, protected: str) -> str:
    try:
        df = pd.read_csv(csv_path)

        if any(col not in df.columns for col in [target, prediction, protected]):
            return json.dumps({"error": "Missing required columns."})

        df = df[[target, prediction, protected]].dropna()

        if df[target].nunique() != 2 or df[prediction].nunique() != 2:
            return json.dumps({"error": "Target and prediction must be binary."})

        # Overall metrics
        TP_total = ((df[target] == 1) & (df[prediction] == 1)).sum()
        TN_total = ((df[target] == 0) & (df[prediction] == 0)).sum()
        FP_total = ((df[target] == 0) & (df[prediction] == 1)).sum()

        overall_accuracy = round((TP_total + TN_total) / len(df), 3)
        overall_fpr = round(FP_total / (FP_total + TN_total + 1e-6), 3)

        groups = df[protected].dropna().unique()
        if len(groups) != 2:
            return json.dumps({"error": "Protected attribute must have exactly two groups."})

        group_stats = {}
        for group in groups:
            sub = df[df[protected] == group]
            TP = ((sub[target] == 1) & (sub[prediction] == 1)).sum()
            TN = ((sub[target] == 0) & (sub[prediction] == 0)).sum()
            FP = ((sub[target] == 0) & (sub[prediction] == 1)).sum()

            acc = round((TP + TN) / len(sub), 3) if len(sub) > 0 else 0
            fpr = round(FP / (FP + TN + 1e-6), 3) if (FP + TN) > 0 else 0

            group_stats[group] = {"accuracy": acc, "fpr": fpr}

        group_a, group_b = groups
        learning_bias = round(abs(group_stats[group_a]["accuracy"] - group_stats[group_b]["accuracy"]), 3)
        fpr_diff = round(abs(group_stats[group_a]["fpr"] - group_stats[group_b]["fpr"]), 3)

        return json.dumps({
            "accuracy": {
                group_a: group_stats[group_a]["accuracy"],
                group_b: group_stats[group_b]["accuracy"]
            },
            "false_positive_rate": {
                group_a: group_stats[group_a]["fpr"],
                group_b: group_stats[group_b]["fpr"]
            },
            "group_deviations": {
                "accuracy_deviation": {
                    group_a: round(abs(group_stats[group_a]["accuracy"] - overall_accuracy), 3),
                    group_b: round(abs(group_stats[group_b]["accuracy"] - overall_accuracy), 3)
                },
                "fpr_deviation": {
                    group_a: round(abs(group_stats[group_a]["fpr"] - overall_fpr), 3),
                    group_b: round(abs(group_stats[group_b]["fpr"] - overall_fpr), 3)
                }
            },
            "overall_metrics": {
                "accuracy": overall_accuracy,
                "fpr": overall_fpr
            },
            "bias_metrics": {
                "learning_bias": learning_bias,
                "fpr_difference": fpr_diff
            }
        })

    except Exception as e:
        return json.dumps({"error": str(e)})

learning_bias_agent = Agent(
    name="Learning Bias Agent",
    instructions="Only use the tool to return clean JSON metrics for accuracy, FPR, and group differences.",
    tools=[analyze_learning_bias],
    model=None
)


'''
import json
import pandas as pd
from agents import Agent, function_tool

@function_tool
def analyze_learning_bias(csv_path: str, target: str, prediction: str, protected: str) -> str:
    try:
        df = pd.read_csv(csv_path)

        # Validate required columns
        if any(col not in df.columns for col in [target, prediction, protected]):
            return json.dumps({"error": "Missing required columns."})

        groups = df[protected].dropna().unique()
        if len(groups) != 2:
            return json.dumps({"error": "Protected attribute must have exactly two groups."})

        group_a, group_b = groups[0], groups[1]
        stats = {}

        for group in [group_a, group_b]:
            subset = df[df[protected] == group]

            TP = ((subset[target] == 1) & (subset[prediction] == 1)).sum()
            TN = ((subset[target] == 0) & (subset[prediction] == 0)).sum()
            FP = ((subset[target] == 0) & (subset[prediction] == 1)).sum()
            FN = ((subset[target] == 1) & (subset[prediction] == 0)).sum()

            total = len(subset)
            accuracy = round((TP + TN) / total, 3) if total > 0 else 0
            fpr = round(FP / (FP + TN + 1e-6), 3) if (FP + TN) > 0 else 0

            stats[group] = {"accuracy": accuracy, "fpr": fpr}

        acc_a, acc_b = stats[group_a]["accuracy"], stats[group_b]["accuracy"]
        fpr_a, fpr_b = stats[group_a]["fpr"], stats[group_b]["fpr"]

        overall_accuracy = round((acc_a + acc_b) / 2, 3)
        overall_fpr = round((fpr_a + fpr_b) / 2, 3)

        result = {
            "protected_attribute": protected,
            "group_a": group_a,
            "group_b": group_b,
            "accuracy": {
                group_a: acc_a,
                group_b: acc_b
            },
            "false_positive_rate": {
                group_a: fpr_a,
                group_b: fpr_b
            },
            "bias_metrics": {
                "learning_bias": round(abs(acc_a - acc_b), 3),
                "fpr_difference": round(abs(fpr_a - fpr_b), 3)
            },
            "overall_metrics": {
                "accuracy": overall_accuracy,
                "false_positive_rate": overall_fpr
            },
            "group_deviations": {
                "accuracy_deviation": {
                    group_a: round(acc_a - overall_accuracy, 3),
                    group_b: round(acc_b - overall_accuracy, 3)
                },
                "fpr_deviation": {
                    group_a: round(fpr_a - overall_fpr, 3),
                    group_b: round(fpr_b - overall_fpr, 3)
                }
            }
        }

        return json.dumps(result)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "protected_attribute": protected
        })

learning_bias_agent = Agent(
    name="Learning Bias Agent",
    instructions="""
    You are a fairness evaluator. Analyze learning bias from model predictions on protected attributes.
    Compute:
    - group-wise accuracy and false positive rate
    - learning_bias = accuracy diff between groups
    - fpr_difference = false positive rate diff
    - overall metrics (accuracy, fpr)
    - deviation of each group from overall
    Return strict JSON with keys:
    group_a, group_b, accuracy, false_positive_rate, bias_metrics, overall_metrics, group_deviations
    """.strip(),
    tools=[analyze_learning_bias],
)

'''

'''
# custom_agents/learning_bias_agent.py
import json
import pandas as pd
from itertools import combinations
from agents import Agent, function_tool

@function_tool
def analyze_learning_bias(csv_path: str, target: str, prediction: str, protected: str) -> str:
    """
    Compute learning bias metrics for multiple groups in a protected attribute.
    Returns JSON with group-wise accuracy, FPR, pairwise learning_bias, pairwise FPR differences,
    overall metrics, and deviations.
    """
    try:
        df = pd.read_csv(csv_path)

        # Validate required columns
        if any(col not in df.columns for col in [target, prediction, protected]):
            return json.dumps({"error": "Missing required columns."})

        groups = df[protected].dropna().unique()
        if len(groups) < 2:
            return json.dumps({"error": "Protected attribute must have at least 2 groups."})

        group_stats = {}
        for group in groups:
            subset = df[df[protected] == group]
            TP = ((subset[target] == 1) & (subset[prediction] == 1)).sum()
            TN = ((subset[target] == 0) & (subset[prediction] == 0)).sum()
            FP = ((subset[target] == 0) & (subset[prediction] == 1)).sum()
            FN = ((subset[target] == 1) & (subset[prediction] == 0)).sum()

            total = len(subset)
            accuracy = round((TP + TN) / total, 3) if total > 0 else None
            fpr = round(FP / (FP + TN + 1e-6), 3) if (FP + TN) > 0 else None

            group_stats[group] = {"accuracy": accuracy, "fpr": fpr}

        # Overall metrics
        valid_accs = [v["accuracy"] for v in group_stats.values() if v["accuracy"] is not None]
        valid_fprs = [v["fpr"] for v in group_stats.values() if v["fpr"] is not None]
        overall_accuracy = round(sum(valid_accs) / len(valid_accs), 3) if valid_accs else None
        overall_fpr = round(sum(valid_fprs) / len(valid_fprs), 3) if valid_fprs else None

        # Pairwise learning bias and FPR differences
        pairwise_bias = []
        for g1, g2 in combinations(groups, 2):
            acc1, acc2 = group_stats[g1]["accuracy"], group_stats[g2]["accuracy"]
            fpr1, fpr2 = group_stats[g1]["fpr"], group_stats[g2]["fpr"]
            pairwise_bias.append({
                "group_1": g1,
                "group_2": g2,
                "learning_bias": round(abs(acc1 - acc2), 3) if acc1 is not None and acc2 is not None else None,
                "fpr_difference": round(abs(fpr1 - fpr2), 3) if fpr1 is not None and fpr2 is not None else None
            })

        # Group deviations
        accuracy_dev = {g: round(group_stats[g]["accuracy"] - overall_accuracy, 3) if group_stats[g]["accuracy"] is not None else None for g in groups}
        fpr_dev = {g: round(group_stats[g]["fpr"] - overall_fpr, 3) if group_stats[g]["fpr"] is not None else None for g in groups}

        result = {
            "protected_attribute": protected,
            "groups": list(groups),
            "group_metrics": group_stats,
            "overall_metrics": {"accuracy": overall_accuracy, "fpr": overall_fpr},
            "pairwise_bias_metrics": pairwise_bias,
            "group_deviations": {"accuracy_deviation": accuracy_dev, "fpr_deviation": fpr_dev}
        }

        return json.dumps(result)

    except Exception as e:
        return json.dumps({"error": str(e), "protected_attribute": protected})

learning_bias_agent = Agent(
    name="Learning Bias Agent",
    instructions="""
    You are a fairness evaluator. Analyze learning bias from model predictions on a protected attribute.
    Compute:
    - Group-wise accuracy and false positive rate (FPR)
    - Pairwise learning bias and FPR differences for all group pairs
    - Overall accuracy and FPR across all groups
    - Deviations of each group from overall
    Return strict JSON with keys:
    protected_attribute, groups, group_metrics, overall_metrics, pairwise_bias_metrics, group_deviations
    """.strip(),
    tools=[analyze_learning_bias],
)

'''