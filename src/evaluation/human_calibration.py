"""
Human-Judge Agreement and Calibration Analysis Engine.
Calculates Cohen's Kappa, Pearson Correlation, Spearman Rank Correlation,
and Mean Absolute Error between Human Annotators and the LLM-as-a-Judge.
"""

import json
import numpy as np
from typing import Dict, Any, List, Tuple
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from src.agent import AppleSupportAgent
from src.evaluation.llm_judge import SupportAgentJudge


class HumanJudgeCalibration:
    """Measures calibration, alignment, and statistical agreement between human raters and LLM judge."""

    def __init__(
        self,
        calibration_data_path: str = "data/golden_set_human_labels.json",
        agent: AppleSupportAgent = None
    ):
        self.data_path = Path(calibration_data_path)
        self.agent = agent or AppleSupportAgent()
        self.judge = SupportAgentJudge()

    def run_calibration_study(self) -> Dict[str, Any]:
        """
        Executes the agent on the calibration subset and computes agreement metrics with human raters.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Calibration data not found at {self.data_path}")

        with open(self.data_path, "r", encoding="utf-8") as f:
            calibration_items = json.load(f)

        human_1_decisions = []
        human_2_decisions = []
        judge_decisions = []
        ground_truth_decisions = []

        human_grounding = []
        human_tone = []
        human_actionability = []
        human_safety = []
        human_overall = []

        judge_grounding = []
        judge_tone = []
        judge_actionability = []
        judge_safety = []
        judge_overall = []

        detailed_records = []

        for item in calibration_items:
            # 1. Run agent on query
            output = self.agent.process_message(item["customer_text"])

            # 2. Run LLM Judge
            judge_eval = self.judge.evaluate_reply(
                customer_query=item["customer_text"],
                generated_reply=output.draft_reply,
                predicted_intent=output.predicted_intent,
                auto_handle=output.auto_handle,
                escalation_reason=output.escalation_reason,
                ground_truth_intent=item["intent"],
                ground_truth_auto_handle=item["ground_truth_auto_handle"],
                reference_resolution=item["reference_resolution"]
            )

            # Extract decisions (1 = auto_handle, 0 = escalate)
            h1_dec = 1 if item["human_annotator_1"]["overall_decision"] == "auto_handle" else 0
            h2_dec = 1 if item["human_annotator_2"]["overall_decision"] == "auto_handle" else 0
            gt_dec = 1 if item["ground_truth_auto_handle"] else 0
            j_dec = 1 if output.auto_handle else 0

            human_1_decisions.append(h1_dec)
            human_2_decisions.append(h2_dec)
            ground_truth_decisions.append(gt_dec)
            judge_decisions.append(j_dec)

            # Consensus human scores (average of human 1 & human 2)
            h_g = (item["human_annotator_1"]["grounding_score"] + item["human_annotator_2"]["grounding_score"]) / 2.0
            h_t = (item["human_annotator_1"]["tone_score"] + item["human_annotator_2"]["tone_score"]) / 2.0
            h_a = (item["human_annotator_1"]["actionability_score"] + item["human_annotator_2"]["actionability_score"]) / 2.0
            h_s = (item["human_annotator_1"]["safety_escalation_score"] + item["human_annotator_2"]["safety_escalation_score"]) / 2.0
            h_o = (h_g + h_t + h_a + h_s) / 4.0

            human_grounding.append(h_g)
            human_tone.append(h_t)
            human_actionability.append(h_a)
            human_safety.append(h_s)
            human_overall.append(h_o)

            judge_grounding.append(judge_eval["grounding_score"])
            judge_tone.append(judge_eval["tone_score"])
            judge_actionability.append(judge_eval["actionability_score"])
            judge_safety.append(judge_eval["safety_score"])
            judge_overall.append(judge_eval["overall_score"])

            detailed_records.append({
                "id": item["example_id"],
                "text": item["customer_text"],
                "draft_reply": output.draft_reply,
                "human_overall": h_o,
                "judge_overall": judge_eval["overall_score"],
                "judge_reasoning": judge_eval["reasoning"]
            })

        # --- Statistical Agreement Computations ---

        # 1. Cohen's Kappa for categorical triage decisions
        kappa_h1_h2 = cohen_kappa_score(human_1_decisions, human_2_decisions)
        kappa_judge_h1 = cohen_kappa_score(human_1_decisions, judge_decisions)
        kappa_judge_gt = cohen_kappa_score(ground_truth_decisions, judge_decisions)

        # 2. Pearson & Spearman correlations for rating dimensions
        corr_overall, p_val_overall = pearsonr(human_overall, judge_overall)
        spearman_overall, _ = spearmanr(human_overall, judge_overall)
        
        corr_grounding, _ = pearsonr(human_grounding, judge_grounding)
        corr_tone, _ = pearsonr(human_tone, judge_tone)
        corr_safety, _ = pearsonr(human_safety, judge_safety)

        # 3. Error Metrics
        mae_overall = float(np.mean(np.abs(np.array(human_overall) - np.array(judge_overall))))
        rmse_overall = float(np.sqrt(np.mean((np.array(human_overall) - np.array(judge_overall)) ** 2)))

        # 4. Binary Agreement Percentage
        raw_agreement_pct = float(np.mean(np.array(ground_truth_decisions) == np.array(judge_decisions))) * 100

        results = {
            "sample_size": len(calibration_items),
            "inter_annotator_agreement": {
                "human1_vs_human2_cohens_kappa": round(float(kappa_h1_h2), 4),
                "judge_vs_human1_cohens_kappa": round(float(kappa_judge_h1), 4),
                "judge_vs_ground_truth_cohens_kappa": round(float(kappa_judge_gt), 4),
                "binary_triage_raw_agreement_pct": round(raw_agreement_pct, 2)
            },
            "rubric_score_correlations": {
                "overall_pearson_r": round(float(corr_overall), 4),
                "overall_spearman_rho": round(float(spearman_overall), 4),
                "overall_p_value": float(p_val_overall),
                "grounding_pearson_r": round(float(corr_grounding), 4),
                "tone_pearson_r": round(float(corr_tone), 4),
                "safety_pearson_r": round(float(corr_safety), 4)
            },
            "rating_error_metrics": {
                "mean_absolute_error_mae": round(mae_overall, 4),
                "root_mean_squared_error_rmse": round(rmse_overall, 4)
            },
            "detailed_records": detailed_records[:5]
        }

        return results


if __name__ == "__main__":
    calibrator = HumanJudgeCalibration()
    res = calibrator.run_calibration_study()
    print("\n=== HUMAN-JUDGE AGREEMENT & CALIBRATION RESULTS ===")
    print(json.dumps(res, indent=2))
