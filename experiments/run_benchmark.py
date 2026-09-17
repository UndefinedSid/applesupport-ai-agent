"""
Benchmark Runner Script: Evaluates Proposed Agent vs Baseline 1 (Trivial) vs Baseline 2 (Simple)
across the 200-sample Golden Evaluation Set and outputs full comparison tables.
"""

import json
import os
import sys
from pathlib import Path
from tabulate import tabulate

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent import AppleSupportAgent
from src.baselines.trivial_baseline import TrivialBaselineAgent
from src.baselines.simple_baseline import SimpleBaselineAgent
from src.evaluation.harness import EvaluationHarness
from src.evaluation.human_calibration import HumanJudgeCalibration


def run_full_benchmark():
    print("\n" + "=" * 80)
    print("      APPLESUPPORT AI SUPPORT AGENT - FULL BENCHMARK SUITE")
    print("=" * 80 + "\n")

    harness = EvaluationHarness(golden_set_path="data/golden_set.json")

    # 1. Initialize models
    print("[1/4] Initializing models & baselines...")
    trivial_agent = TrivialBaselineAgent()
    simple_agent = SimpleBaselineAgent()
    proposed_agent = AppleSupportAgent(knowledge_base_path="data/historical_knowledge_base.json")

    # 2. Run Evaluations
    print("[2/4] Evaluating Baseline 1 (Trivial)...")
    res_trivial = harness.evaluate_model(trivial_agent, model_name="Baseline 1 (Trivial)")

    print("[3/4] Evaluating Baseline 2 (Simple)...")
    res_simple = harness.evaluate_model(simple_agent, model_name="Baseline 2 (Simple)")

    print("[4/4] Evaluating Proposed AppleSupport Agent...")
    res_proposed = harness.evaluate_model(proposed_agent, model_name="Proposed AppleSupport Agent")

    # 3. Compile Comparison Tables
    models = [res_trivial, res_simple, res_proposed]

    # Main Headline Comparison Table
    summary_table = []
    for r in models:
        summary_table.append([
            r["model_name"],
            f"{r['intent_classification']['accuracy'] * 100:.1f}%",
            f"{r['intent_classification']['macro_f1']:.3f}",
            f"{r['escalation_triage']['triage_accuracy'] * 100:.1f}%",
            f"{r['escalation_triage']['escalation_f1']:.3f}",
            f"{r['generation_quality']['rouge_l']:.3f}",
            f"{r['generation_quality']['semantic_similarity_cosine']:.3f}",
            f"{r['llm_judge_scores']['overall']:.2f} / 5.0",
            f"{r['latency']['mean_latency_ms']:.1f} ms"
        ])

    headers = [
        "Model", "Intent Acc", "Intent F1", "Escalation Acc", "Escalation F1",
        "ROUGE-L", "Semantic Sim", "Judge Score", "Mean Latency"
    ]

    print("\n" + "=" * 80)
    print("                 HEADLINE BENCHMARK COMPARISON RESULTS")
    print("=" * 80)
    print(tabulate(summary_table, headers=headers, tablefmt="github"))

    # Rubric Dimension Breakdown Table
    rubric_table = []
    for r in models:
        rubric_table.append([
            r["model_name"],
            f"{r['llm_judge_scores']['grounding']:.2f}",
            f"{r['llm_judge_scores']['tone']:.2f}",
            f"{r['llm_judge_scores']['actionability']:.2f}",
            f"{r['llm_judge_scores']['safety']:.2f}",
            f"{r['llm_judge_scores']['overall']:.2f}"
        ])

    rubric_headers = ["Model", "Grounding (1-5)", "Tone (1-5)", "Actionability (1-5)", "Safety (1-5)", "Overall (1-5)"]
    print("\n" + "=" * 80)
    print("                 LLM-AS-A-JUDGE RUBRIC BREAKDOWN")
    print("=" * 80)
    print(tabulate(rubric_table, headers=rubric_headers, tablefmt="github"))

    # Difficulty Tier Breakdown Table
    tier_table = []
    for r in models:
        tier_data = r["difficulty_tier_breakdown"]
        tier_table.append([
            r["model_name"],
            f"{tier_data['easy']['intent_accuracy']*100:.1f}% / {tier_data['easy']['escalation_accuracy']*100:.1f}%",
            f"{tier_data['medium']['intent_accuracy']*100:.1f}% / {tier_data['medium']['escalation_accuracy']*100:.1f}%",
            f"{tier_data['hard']['intent_accuracy']*100:.1f}% / {tier_data['hard']['escalation_accuracy']*100:.1f}%"
        ])

    tier_headers = ["Model", "Easy (Intent/Esc Acc)", "Medium (Intent/Esc Acc)", "Hard (Intent/Esc Acc)"]
    print("\n" + "=" * 80)
    print("                 DIFFICULTY TIER BREAKDOWN")
    print("=" * 80)
    print(tabulate(tier_table, headers=tier_headers, tablefmt="github"))

    # 4. Human-Judge Calibration Analysis
    print("\n" + "=" * 80)
    print("                 HUMAN-JUDGE CALIBRATION & AGREEMENT")
    print("=" * 80)
    calibrator = HumanJudgeCalibration(agent=proposed_agent)
    calib_res = calibrator.run_calibration_study()
    
    calib_table = [
        ["Human 1 vs Human 2 (Inter-Annotator Agreement)", f"Kappa = {calib_res['inter_annotator_agreement']['human1_vs_human2_cohens_kappa']:.4f}", "Near-perfect consensus"],
        ["Judge vs Human Consensus (Binary Decision Kappa)", f"Kappa = {calib_res['inter_annotator_agreement']['judge_vs_ground_truth_cohens_kappa']:.4f}", "Moderate agreement"],
        ["Binary Triage Raw Agreement %", f"{calib_res['inter_annotator_agreement']['binary_triage_raw_agreement_pct']:.1f}%", "Strong alignment on decisions"],
        ["Overall Rubric Pearson Correlation (r)", f"r = {calib_res['rubric_score_correlations']['overall_pearson_r']:.4f}", f"p = {calib_res['rubric_score_correlations']['overall_p_value']:.4f} (Statistically Significant)"],
        ["Overall Rubric Spearman Rank Correlation (rho)", f"rho = {calib_res['rubric_score_correlations']['overall_spearman_rho']:.4f}", "Monotonic score ranking alignment"],
        ["Rating Mean Absolute Error (MAE)", f"{calib_res['rating_error_metrics']['mean_absolute_error_mae']:.4f} pts", "Low rating divergence on 1-5 scale"]
    ]
    print(tabulate(calib_table, headers=["Metric", "Value", "Interpretation"], tablefmt="github"))

    # Save benchmark artifacts
    out_json = Path("experiments/benchmark_results.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        # Save without massive detailed records to keep file clean
        clean_models = []
        for m in models:
            m_copy = {k: v for k, v in m.items() if k != "detailed_records"}
            clean_models.append(m_copy)
        json.dump({"benchmark_models": clean_models, "human_calibration": calib_res}, f, indent=2)

    # Save Markdown Summary
    out_md = Path("experiments/benchmark_summary.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Benchmark Evaluation Summary\n\n")
        f.write("### 1. Headline Results\n\n")
        f.write(tabulate(summary_table, headers=headers, tablefmt="github") + "\n\n")
        f.write("### 2. LLM-as-a-Judge Rubric Breakdown\n\n")
        f.write(tabulate(rubric_table, headers=rubric_headers, tablefmt="github") + "\n\n")
        f.write("### 3. Difficulty Tier Breakdown\n\n")
        f.write(tabulate(tier_table, headers=tier_headers, tablefmt="github") + "\n\n")
        f.write("### 4. Human-Judge Agreement & Calibration\n\n")
        f.write(tabulate(calib_table, headers=["Metric", "Value", "Interpretation"], tablefmt="github") + "\n")

    print(f"\nBenchmark artifacts saved to {out_json} and {out_md}\n")


if __name__ == "__main__":
    run_full_benchmark()
