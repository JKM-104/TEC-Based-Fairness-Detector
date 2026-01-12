

import json
from agents import Agent, function_tool

@function_tool
def generate_report(profile_json: str, rate_result_json: str, inclusion_result_json: str,
                    learning_bias_json: str, policy_json: str) -> str:
    try:
        profile = json.loads(profile_json)
        rate = json.loads(rate_result_json)
        inclusion = json.loads(inclusion_result_json)
        learning = json.loads(learning_bias_json)
        policy = json.loads(policy_json)

        report = "# TEC AI Fairness Guidelines: Fairness Report\n\n"
        report += "The following fairness metrics were evaluated based on the provided dataset and model predictions.\n\n"

        # Dataset Profile
        report += "## 📄 Dataset Profile\n"
        if "error" in profile:
            report += f"⚠️ {profile['error']}\n\n"
        else:
            report += f"- Shape: {profile.get('shape')}\n"
            report += f"- Columns: {profile.get('columns')}\n"
            report += f"- Protected Attributes: {profile.get('protected')}\n"
            report += f"- Target Variable: {profile.get('target')}\n\n"

        # Bias Analysis
        report += "## ⚖️ Bias Assessment\n"

        # Rate Bias
        if "error" in rate:
            report += f"⚠️ Rate Bias Error: {rate['error']}\n\n"
        else:
            report += "### 🔹 Rate Difference\n"
            report += (f"- Protected Attribute: **{rate['protected_attribute']}**\n"
                       f"- Group A: {rate['group_a']} (Rate: {rate['rate_a']})\n"
                       f"- Group B: {rate['group_b']} (Rate: {rate['rate_b']})\n"
                       f"- **Rate Difference**: {rate['rate_difference']} (Threshold: {policy['rate_difference_threshold']})\n\n")

        # Inclusion Bias
        if "error" in inclusion:
            report += f"⚠️ Inclusion Bias Error: {inclusion['error']}\n\n"
        else:
            report += "### 🔹 Inclusion Bias\n"
            report += (f"- Protected Attribute: **{inclusion['protected_attribute']}**\n"
                       f"- Most Represented: {inclusion['most_represented_group']}\n"
                       f"- Least Represented: {inclusion['least_represented_group']}\n"
                       f"- **Distribution Gap**: {inclusion['distribution_gap_percent']}% (Threshold: {policy['inclusion_gap_threshold']}%)\n\n")

        # Learning Bias
        if "error" in learning:
            report += f"⚠️ Learning Bias Error: {learning['error']}\n\n"
        else:
            report += "### 🔹 Learning Bias and FPR\n"
            report += f"- **Overall Accuracy**: {learning['overall_metrics']['accuracy']}\n"
            report += f"- **Overall FPR**: {learning['overall_metrics']['fpr']}\n\n"

            report += f"#### Accuracy by Group\n"
            for grp, acc in learning["accuracy"].items():
                dev = learning["group_deviations"]["accuracy_deviation"].get(grp, "N/A")
                report += f"- {grp}: {acc} (Deviation from overall: {dev})\n"
            report += "\n"

            report += f"#### FPR by Group\n"
            for grp, fpr in learning["false_positive_rate"].items():
                dev = learning["group_deviations"]["fpr_deviation"].get(grp, "N/A")
                report += f"- {grp}: {fpr} (Deviation from overall: {dev})\n"
            report += "\n"

            report += f"#### Bias Metrics\n"
            report += f"- **Learning Bias (accuracy difference)**: {learning['bias_metrics']['learning_bias']} (Threshold: {policy['learning_bias_threshold']})\n"
            report += f"- **FPR Difference**: {learning['bias_metrics']['fpr_difference']} (Threshold: {policy['fpr_difference_threshold']})\n\n"

        # Final Recommendations
        report += "## 📌 Conclusion & Action Items\n"

        if "error" in learning or "error" in rate or "error" in inclusion:
            report += "- ⚠️ Some fairness metrics could not be evaluated due to errors above. Please review.\n"
        else:
            issues = []
            if rate["rate_difference"] > policy["rate_difference_threshold"]:
                issues.append("rate difference")
            if inclusion["distribution_gap_percent"] > policy["inclusion_gap_threshold"]:
                issues.append("inclusion gap")
            if learning["bias_metrics"]["learning_bias"] > policy["learning_bias_threshold"]:
                issues.append("learning bias")
            if learning["bias_metrics"]["fpr_difference"] > policy["fpr_difference_threshold"]:
                issues.append("FPR difference")

            if not issues:
                report += "- ✅ The dataset and model appear to meet the fairness thresholds defined in the policy.\n"
            else:
                report += f"- ❗ The following fairness issues exceed defined thresholds: {', '.join(issues)}\n"
                report += "- 🔄 Consider rebalancing the dataset, improving representation, or retraining the model.\n"

        return report

    except Exception as e:
        return f"⚠️ Report generation failed: {str(e)}"

report_agent = Agent(
    name="Report Agent",
    instructions="You are a report writer generating TEC-style Markdown fairness reports using JSON outputs from other agents.",
    tools=[generate_report]
)


'''
import json
from agents import Agent, function_tool

@function_tool
def generate_report(profile_json: str, rate_result_json: str, inclusion_result_json: str,
                    learning_bias_json: str, policy_json: str) -> str:
    try:
        profile = json.loads(profile_json)
        rate = json.loads(rate_result_json)
        inclusion = json.loads(inclusion_result_json)
        learning = json.loads(learning_bias_json)
        policy = json.loads(policy_json)

        report = "# TEC AI Fairness Guidelines: Fairness Report\n"
        report += "Utilization of the following datasets should only be done if the following fairness practices are met:\n\n"

        report += "## Dataset Profile\n"
        if "error" in profile:
            report += f"{profile['error']}\n\n"
        else:
            report += f"- Columns: {profile.get('columns')}\n"
            report += f"- Shape: {profile.get('shape')}\n"
            report += f"- Protected Attributes: {profile.get('protected')}\n"
            report += f"- Target Variable: {profile.get('target')}\n\n"

        report += "## Dataset Bias Analysis\n"

        if "error" in rate:
            report += f"⚠️ Rate Bias Error: {rate['error']}\n\n"
        else:
            report += f"### Rate Difference\n"
            report += (f"Protected Attribute: {rate['protected_attribute']}\n"
                       f"- Group A: {rate['group_a']} (Rate: {rate['rate_a']})\n"
                       f"- Group B: {rate['group_b']} (Rate: {rate['rate_b']})\n"
                       f"- Difference: {rate['rate_difference']} (Threshold: {policy['rate_difference_threshold']})\n\n")

        if "error" in inclusion:
            report += f"⚠️ Inclusion Bias Error: {inclusion['error']}\n\n"
        else:
            report += "### Inclusion/Exclusion Bias\n"
            report += (f"Protected Attribute: {inclusion['protected_attribute']}\n"
                       f"- Most Represented: {inclusion['most_represented_group']}\n"
                       f"- Least Represented: {inclusion['least_represented_group']}\n"
                       f"- Gap: {inclusion['distribution_gap_percent']}% (Threshold: {policy['inclusion_gap_threshold']}%)\n\n")

        if "error" in learning:
            report += f"⚠️ Learning Bias Error: {learning['error']}\n\n"
        else:
            report += "### Learning Bias and False Positive Rate\n"
            report += f"- Learning Bias (accuracy difference between groups): {learning['bias_metrics']['learning_bias']} (Threshold: {policy['learning_bias_threshold']})\n"
            report += f"- FPR Difference Between Groups: {learning['bias_metrics']['fpr_difference']} (Threshold: {policy['fpr_difference_threshold']})\n\n"

            report += "#### Group-wise Metrics\n"
            for group in learning['accuracy']:
                report += f"- {group}:\n"
                report += f"  - Accuracy: {learning['accuracy'][group]}\n"
                report += f"  - False Positive Rate: {learning['false_positive_rate'][group]}\n"
            report += "\n"

        report += "## Action Items\n"
        report += "Following the guidelines below will help ensure fairness:\n"
        report += "- [ ] Validate dataset bias with updated policy thresholds\n"
        report += "- [ ] Improve representation of underrepresented groups\n"
        report += "- [ ] Retrain model if learning bias or FPR difference exceeds thresholds\n"

        return report

    except Exception as e:
        return f"⚠️ Failed to generate report: {str(e)}"


report_agent = Agent(
    name="Report Agent",
    instructions="You are a report writer generating TEC-style Markdown fairness reports from JSON results of other agents.",
    tools=[generate_report],
)

'''

'''
# custom_agents/report_agent.py
import json
from typing import Any, Dict, List, Optional
from agents import Agent, function_tool

def _safe_parse(s: Optional[str]) -> Any:
    """Safely parse JSON string or return error dict."""
    if s is None:
        return None
    if isinstance(s, (dict, list)):
        return s
    try:
        return json.loads(s)
    except Exception:
        return {"__parse_error__": True, "raw_output": str(s)}

def _to_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

@function_tool
def generate_report(
    profile_json: str,
    rate_result_json: str,
    inclusion_result_json: str,
    learning_bias_json: str,
    policy_json: Optional[str] = None,
) -> str:
    """
    Generate TEC-style Markdown fairness report + structured compliance summary.
    Handles multiple groups in protected attributes and all pairwise metrics.
    """
    try:
        profile = _safe_parse(profile_json)
        rate = _safe_parse(rate_result_json)
        inclusion = _safe_parse(inclusion_result_json)
        learning = _safe_parse(learning_bias_json)
        policy = _safe_parse(policy_json) if policy_json else {}

        # --- Thresholds ---
        rate_thr = _to_float(policy.get("rate_difference_threshold"), 0.05)
        inclusion_thr = _to_float(policy.get("inclusion_gap_threshold"), 10.0)
        learning_thr = _to_float(policy.get("learning_bias_threshold"), 0.05)
        fpr_thr = _to_float(policy.get("fpr_difference_threshold"), 0.05)
        overall_acc_min = _to_float(policy.get("overall_accuracy_minimum"), 0.6)
        overall_fpr_max = _to_float(policy.get("overall_fpr_maximum"), 0.3)
        max_acc_dev = _to_float(policy.get("max_group_accuracy_deviation"), 0.05)
        max_fpr_dev = _to_float(policy.get("max_group_fpr_deviation"), 0.05)

        md_lines: List[str] = []
        md_lines.append("# TEC AI Fairness Guidelines: Fairness Report\n")
        md_lines.append("Utilization of the following datasets should only be done if the following fairness practices are met:\n")

        # --- Dataset Profile ---
        md_lines.append("## Dataset Profile\n")
        if not profile or profile.get("__parse_error__"):
            err = profile.get("raw_output") if isinstance(profile, dict) else "No profile output."
            md_lines.append(f"⚠️ Dataset profile error: {err}\n")
        else:
            md_lines.append(f"- Columns: {profile.get('columns')}\n")
            md_lines.append(f"- Shape: {profile.get('shape')}\n")
            md_lines.append(f"- Protected Attributes: {profile.get('protected')}\n")
            md_lines.append(f"- Target Variable: {profile.get('target')}\n")

        compliance: Dict[str, Any] = {}
        failures: List[str] = []

        # --- Rate Difference ---
        md_lines.append("\n## Dataset Bias Analysis\n")
        md_lines.append("### Rate Difference\n")
        if not rate or rate.get("__parse_error__"):
            md_lines.append(f"⚠️ Rate bias error: {(rate.get('raw_output') if isinstance(rate, dict) else 'No rate output')}\n")
            compliance["rate"] = {"passed": False, "note": "no_valid_rate_output"}
        else:
            rate_items = rate if isinstance(rate, list) else [rate]
            all_rate_pass = True
            for it in rate_items:
                ga, gb, rd = it.get("group_a"), it.get("group_b"), _to_float(it.get("rate_difference"))
                passed = (rd is not None) and (abs(rd) <= rate_thr)
                status = "PASS" if passed else "FAIL"
                if not passed:
                    failures.append(f"rate_diff_{ga}_vs_{gb}")
                    all_rate_pass = False
                md_lines.append(f"- {ga} vs {gb}: rate difference = {rd} (threshold = {rate_thr}) -> {status}\n")
            compliance["rate"] = {"passed": all_rate_pass, "results": rate_items}

        # --- Inclusion / Exclusion ---
        md_lines.append("\n### Inclusion/Exclusion Bias\n")
        if not inclusion or inclusion.get("__parse_error__"):
            md_lines.append(f"⚠️ Inclusion bias error: {(inclusion.get('raw_output') if isinstance(inclusion, dict) else 'No inclusion output')}\n")
            compliance["inclusion"] = {"passed": False, "note": "no_valid_inclusion_output"}
        else:
            dist_gap = _to_float(inclusion.get("distribution_gap_percent"))
            pa = inclusion.get("protected_attribute")
            mr, lr = inclusion.get("most_represented_group"), inclusion.get("least_represented_group")
            md_lines.append(f"Protected Attribute: {pa}\n")
            md_lines.append(f"- Most Represented: {mr}\n")
            md_lines.append(f"- Least Represented: {lr}\n")
            if dist_gap is not None:
                passed = dist_gap <= inclusion_thr
                status = "PASS" if passed else "FAIL"
                if not passed:
                    failures.append("inclusion_gap")
                md_lines.append(f"- Distribution gap: {dist_gap}% (threshold = {inclusion_thr}%) -> {status}\n")
                compliance["inclusion"] = {"passed": passed, "value": dist_gap, "threshold": inclusion_thr}

        # --- Learning Bias & Pairwise Metrics ---
        md_lines.append("\n### Learning Bias & FPR\n")
        if not learning or learning.get("__parse_error__"):
            md_lines.append(f"⚠️ Learning bias error: {(learning.get('raw_output') if isinstance(learning, dict) else 'No learning output')}\n")
            compliance["learning"] = {"passed": False, "note": "no_valid_learning_output"}
        else:
            pairwise = learning.get("pairwise_bias_metrics", [])
            group_metrics = learning.get("group_metrics", {})
            overall = learning.get("overall_metrics", {})
            group_devs = learning.get("group_deviations", {})

            # Render pairwise results
            pairwise_pass_all = True
            for p in pairwise:
                g1, g2 = p.get("group_1"), p.get("group_2")
                lb, fprd = _to_float(p.get("learning_bias")), _to_float(p.get("fpr_difference"))
                lb_pass = lb is not None and abs(lb) <= learning_thr
                fpr_pass = fprd is not None and abs(fprd) <= fpr_thr
                if not (lb_pass and fpr_pass):
                    pairwise_pass_all = False
                    failures.append(f"learning_{g1}_vs_{g2}")
                md_lines.append(f"- {g1} vs {g2}: learning_bias={lb} (thr={learning_thr}) -> {'PASS' if lb_pass else 'FAIL'}, fpr_diff={fprd} (thr={fpr_thr}) -> {'PASS' if fpr_pass else 'FAIL'}\n")
            compliance["learning_pairwise"] = {"passed": pairwise_pass_all, "pairs": pairwise}

            # Overall metrics
            overall_acc, overall_fpr = _to_float(overall.get("accuracy")), _to_float(overall.get("fpr"))
            acc_ok = overall_acc is None or overall_acc >= overall_acc_min
            fpr_ok = overall_fpr is None or overall_fpr <= overall_fpr_max
            if not acc_ok:
                failures.append("overall_accuracy")
            if not fpr_ok:
                failures.append("overall_fpr")
            compliance["learning_overall"] = {"overall_accuracy_pass": acc_ok, "overall_fpr_pass": fpr_ok, "overall_accuracy": overall_acc, "overall_fpr": overall_fpr}

            # Group deviations
            dev_acc_max = max((abs(_to_float(v)) for v in group_devs.get("accuracy_deviation", {}).values()), default=0)
            dev_fpr_max = max((abs(_to_float(v)) for v in group_devs.get("fpr_deviation", {}).values()), default=0)
            devs_ok = dev_acc_max <= max_acc_dev and dev_fpr_max <= max_fpr_dev
            if not devs_ok:
                failures.append("group_deviations")
            compliance["group_deviations"] = {"max_accuracy_deviation": dev_acc_max, "max_fpr_deviation": dev_fpr_max, "passed": devs_ok}

        # --- Overall pass ---
        overall_pass = not failures
        compliance["overall_pass"] = overall_pass
        compliance["failed_items"] = failures

        # --- Markdown summary ---
        md_lines.append("\n## Action Items & Summary\n")
        if overall_pass:
            md_lines.append("- ✅ **Overall: Passed** — all metrics within thresholds.\n")
        else:
            md_lines.append("- ❌ **Overall: Failed** — see failed items below.\n")
            if failures:
                md_lines.append("\n### Failed Checks\n")
                for f in failures:
                    md_lines.append(f"- {f}\n")

        md_lines.append("\n### Compliance Summary (JSON)\n")
        md_lines.append("```\n" + json.dumps(compliance, indent=2) + "\n```\n")

        return json.dumps({"report": "".join(md_lines), "compliance": compliance})

    except Exception as e:
        return json.dumps({"error": f"Failed to generate report: {str(e)}"})

report_agent = Agent(
    name="Report Agent",
    instructions="""
    You are a report writer generating TEC-style Markdown fairness reports from JSON results of other agents.
    Handles multiple groups in protected attributes, all pairwise learning bias, FPR differences, overall metrics, and deviations.
    Output JSON with keys 'report' (markdown) and 'compliance' (structured summary).
    """.strip(),
    tools=[generate_report],
)

'''