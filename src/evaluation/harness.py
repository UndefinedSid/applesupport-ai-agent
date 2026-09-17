"""
Unified Evaluation Harness for AppleSupport AI Support Agent and Baselines.
Evaluates end-to-end performance across the 200-sample Golden Evaluation Set.
"""

import time
import json
import logging
from typing import Dict, Any, List, Union
from pathlib import Path
import numpy as np
import pandas as pd

from src.evaluation.metrics import EvaluationMetrics
from src.evaluation.llm_judge import SupportAgentJudge
from src.agent import AppleSupportAgent
from src.baselines.trivial_baseline import TrivialBaselineAgent
from src.baselines.simple_baseline import SimpleBaselineAgent

logger = logging.getLogger(__name__)


class EvaluationHarness:
    """Runs automated evaluation benchmarks on the golden test set."""

    def __init__(self, golden_set_path: str = "data/golden_set.json"):
        self.golden_set_path = Path(golden_set_path)
        if not self.golden_set_path.exists():
            raise FileNotFoundError(f"Golden evaluation set not found at {self.golden_set_path}")
        
        with open(self.golden_set_path, "r", encoding="utf-8") as f:
            self.golden_data = json.load(f)

        self.judge = SupportAgentJudge()

    def evaluate_model(
        self,
        model_instance: Union[AppleSupportAgent, TrivialBaselineAgent, SimpleBaselineAgent],
        model_name: str = "Proposed Agent"
    ) -> Dict[str, Any]:
        """
        Runs the specified model on all golden set examples and computes full diagnostic metrics.
        """
        logger.info(f"Running evaluation benchmark for '{model_name}' on {len(self.golden_data)} golden examples...")

        y_true_intent = []
        y_pred_intent = []
        y_true_auto = []
        y_pred_auto = []
        
        candidate_replies = []
        reference_resolutions = []
        latencies = []
        detailed_records = []

        for item in self.golden_data:
            start_time = time.perf_counter()
            out = model_instance.process_message(item["text"])
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            latencies.append(elapsed_ms)

            # Extract predictions
            pred_intent = out.predicted_intent if hasattr(out, "predicted_intent") else out["predicted_intent"]
            auto_h = out.auto_handle if hasattr(out, "auto_handle") else out["auto_handle"]
            draft_r = out.draft_reply if hasattr(out, "draft_reply") else out["draft_reply"]
            reason = out.escalation_reason if hasattr(out, "escalation_reason") else out.get("escalation_reason", "none")

            y_true_intent.append(item["intent"])
            y_pred_intent.append(pred_intent)
            y_true_auto.append(item["auto_handle"])
            y_pred_auto.append(auto_h)
            candidate_replies.append(draft_r)
            reference_resolutions.append(item["ref"])

            detailed_records.append({
                "id": item["id"],
                "customer_text": item["text"],
                "ground_truth_intent": item["intent"],
                "predicted_intent": pred_intent,
                "ground_truth_auto_handle": item["auto_handle"],
                "auto_handle": auto_h,
                "escalation_reason": reason,
                "draft_reply": draft_r,
                "reference_resolution": item["ref"],
                "difficulty_tier": item["tier"],
                "latency_ms": round(elapsed_ms, 2)
            })

        # 1. Intent Classification Metrics
        clf_metrics = EvaluationMetrics.compute_classification_metrics(y_true_intent, y_pred_intent)

        # 2. Escalation / Triage Metrics
        esc_metrics = EvaluationMetrics.compute_escalation_metrics(y_true_auto, y_pred_auto)

        # 3. Text Generation & Grounding NLP Metrics
        nlp_metrics = EvaluationMetrics.compute_bleu_and_rouge(candidate_replies, reference_resolutions)
        sem_sim = EvaluationMetrics.compute_semantic_similarity(candidate_replies, reference_resolutions)

        # 4. Compliance & Length Metrics
        comp_metrics = EvaluationMetrics.compute_compliance_metrics(candidate_replies, [not a for a in y_pred_auto])

        # 5. LLM-as-a-Judge Evaluation Rubric
        judge_batch_res = self.judge.evaluate_batch(detailed_records)

        # 6. Latency summary
        latency_summary = {
            "mean_latency_ms": round(float(np.mean(latencies)), 2),
            "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
            "throughput_qps": round(1000.0 / float(np.mean(latencies)), 1)
        }

        # Difficulty Tier Breakdown
        tier_results = {}
        for tier in ["easy", "medium", "hard"]:
            tier_indices = [i for i, item in enumerate(self.golden_data) if item["tier"] == tier]
            if tier_indices:
                sub_true_intent = [y_true_intent[i] for i in tier_indices]
                sub_pred_intent = [y_pred_intent[i] for i in tier_indices]
                sub_true_auto = [y_true_auto[i] for i in tier_indices]
                sub_pred_auto = [y_pred_auto[i] for i in tier_indices]
                
                tier_acc = float(np.mean(np.array(sub_true_intent) == np.array(sub_pred_intent)))
                tier_esc_acc = float(np.mean(np.array(sub_true_auto) == np.array(sub_pred_auto)))
                tier_results[tier] = {
                    "count": len(tier_indices),
                    "intent_accuracy": round(tier_acc, 4),
                    "escalation_accuracy": round(tier_esc_acc, 4)
                }

        results = {
            "model_name": model_name,
            "sample_count": len(self.golden_data),
            "intent_classification": clf_metrics,
            "escalation_triage": esc_metrics,
            "generation_quality": {
                **nlp_metrics,
                "semantic_similarity_cosine": sem_sim
            },
            "compliance": comp_metrics,
            "llm_judge_scores": {
                "grounding": judge_batch_res["mean_grounding_score"],
                "tone": judge_batch_res["mean_tone_score"],
                "actionability": judge_batch_res["mean_actionability_score"],
                "safety": judge_batch_res["mean_safety_score"],
                "overall": judge_batch_res["mean_overall_score"]
            },
            "latency": latency_summary,
            "difficulty_tier_breakdown": tier_results,
            "detailed_records": detailed_records
        }

        logger.info(f"Evaluation for '{model_name}' complete. Overall Judge Score: {judge_batch_res['mean_overall_score']}/5.0")
        return results
