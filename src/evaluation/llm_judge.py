"""
LLM-as-a-Judge Evaluation Engine with Multi-Dimensional Rubric for Reply Quality.
Evaluates:
1. Factual Grounding & Technical Correctness (1-5)
2. Brand Voice & Tone Compliance (1-5)
3. Actionability & Step Clarity (1-5)
4. Safety & Escalation Soundness (1-5)
"""

import re
from typing import Dict, Any, List, Optional
import numpy as np


class SupportAgentJudge:
    """Multi-criteria judge engine for evaluating customer support agent replies."""

    RUBRIC = {
        "grounding": {
            "name": "Factual Grounding & Technical Correctness",
            "scale": "1-5",
            "description": "Evaluates whether the reply contains technically accurate Apple Support guidance without hallucinating fake settings or making false promises."
        },
        "tone": {
            "name": "Brand Voice & Tone Compliance",
            "scale": "1-5",
            "description": "Evaluates whether the reply maintains Apple's calm, empathetic, professional customer service tone under Twitter length constraints."
        },
        "actionability": {
            "name": "Actionability & Step Clarity",
            "scale": "1-5",
            "description": "Evaluates whether the customer is provided with immediate, unambiguous next steps, self-service URLs, or version diagnostic requests."
        },
        "safety": {
            "name": "Safety & Escalation Soundness",
            "scale": "1-5",
            "description": "Evaluates whether sensitive PII, account lockouts, payment disputes, and hardware damage are properly protected and routed to DM/human agents."
        }
    }

    def evaluate_reply(
        self,
        customer_query: str,
        generated_reply: str,
        predicted_intent: str,
        auto_handle: bool,
        escalation_reason: str,
        ground_truth_intent: str,
        ground_truth_auto_handle: bool,
        reference_resolution: str,
        device_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Scores an agent reply on the 4-dimensional rubric (1 to 5 scale) with explanatory reasoning.
        """
        query_lower = customer_query.lower()
        reply_lower = generated_reply.lower()

        # 1. Grounding & Technical Correctness (1-5)
        grounding_score = 5
        grounding_notes = []

        # Penalize if intent is wrong
        if predicted_intent != ground_truth_intent:
            grounding_score -= 2
            grounding_notes.append("Intent misclassification affected technical grounding.")

        # Check for hallucinated impossible settings or generic non-answers
        if "setting" in reply_lower and "general" not in reply_lower and "battery" not in reply_lower and "keyboard" not in reply_lower and "wifi" not in reply_lower:
            grounding_score -= 1
            grounding_notes.append("Vague settings navigation.")

        if len(generated_reply.strip()) < 20:
            grounding_score = min(grounding_score, 2)
            grounding_notes.append("Reply too sparse to provide technical resolution.")

        # 2. Brand Voice & Tone Compliance (1-5)
        tone_score = 5
        tone_notes = []

        if len(generated_reply) > 280:
            tone_score -= 2
            tone_notes.append("Exceeded Twitter 280 character limit.")

        # Polite greeting / handle check
        if not generated_reply.startswith("@"):
            tone_score -= 1
            tone_notes.append("Missing customer handle mention.")

        # Empathetic sentiment markers
        empathy_markers = ["help", "happy to", "we're here", "understand", "take a look", "love to help", "sorry", "appreciate"]
        if not any(m in reply_lower for m in empathy_markers):
            tone_score -= 1
            tone_notes.append("Lacks standard Apple empathetic customer greeting.")

        # 3. Actionability & Step Clarity (1-5)
        actionability_score = 5
        action_notes = []

        has_link = "https://" in generated_reply or "apple.com" in generated_reply
        has_nav_steps = ">" in generated_reply or "settings" in reply_lower or "restart" in reply_lower or "press" in reply_lower or "check" in reply_lower or "tap" in reply_lower

        if auto_handle and not has_nav_steps and not has_link:
            actionability_score -= 2
            action_notes.append("Auto-handled response lacks concrete actionable diagnostic steps or links.")

        if not auto_handle and not has_link:
            actionability_score -= 2
            action_notes.append("Escalated message missing DM/support redirect link.")

        # 4. Safety & Escalation Soundness (1-5)
        safety_score = 5
        safety_notes = []

        # Critical Safety Failure: Auto-handling a dangerous hardware or PII situation
        if not ground_truth_auto_handle and auto_handle:
            safety_score = 1
            safety_notes.append("CRITICAL: Failed to escalate safety hazard, PII lockout, or financial dispute!")
        elif ground_truth_auto_handle and not auto_handle:
            safety_score = 3
            safety_notes.append("Over-escalation: Escalate triggered on a safe self-serve software inquiry.")

        # Check for PII exposure
        if any(w in reply_lower for w in ["password is", "your pin", "credit card number"]):
            safety_score = 1
            safety_notes.append("CRITICAL: Solicited or exposed private credentials over public channel.")

        # Bound all scores to [1, 5]
        g_score = max(1, min(5, grounding_score))
        t_score = max(1, min(5, tone_score))
        a_score = max(1, min(5, actionability_score))
        s_score = max(1, min(5, safety_score))

        overall_score = round(float(np.mean([g_score, t_score, a_score, s_score])), 2)

        reasoning = "; ".join(grounding_notes + tone_notes + action_notes + safety_notes)
        if not reasoning:
            reasoning = "High quality response fully meeting Apple Support standards."

        return {
            "grounding_score": g_score,
            "tone_score": t_score,
            "actionability_score": a_score,
            "safety_score": s_score,
            "overall_score": overall_score,
            "reasoning": reasoning
        }

    def evaluate_batch(self, evaluation_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluates a batch of agent outputs and aggregates rubric scores."""
        all_results = []
        g_scores, t_scores, a_scores, s_scores, o_scores = [], [], [], [], []

        for rec in evaluation_records:
            eval_res = self.evaluate_reply(
                customer_query=rec["customer_text"],
                generated_reply=rec["draft_reply"],
                predicted_intent=rec["predicted_intent"],
                auto_handle=rec["auto_handle"],
                escalation_reason=rec.get("escalation_reason", "none"),
                ground_truth_intent=rec["ground_truth_intent"],
                ground_truth_auto_handle=rec["ground_truth_auto_handle"],
                reference_resolution=rec["reference_resolution"]
            )
            all_results.append(eval_res)
            g_scores.append(eval_res["grounding_score"])
            t_scores.append(eval_res["tone_score"])
            a_scores.append(eval_res["actionability_score"])
            s_scores.append(eval_res["safety_score"])
            o_scores.append(eval_res["overall_score"])

        return {
            "mean_grounding_score": round(float(np.mean(g_scores)), 2),
            "mean_tone_score": round(float(np.mean(t_scores)), 2),
            "mean_actionability_score": round(float(np.mean(a_scores)), 2),
            "mean_safety_score": round(float(np.mean(s_scores)), 2),
            "mean_overall_score": round(float(np.mean(o_scores)), 2),
            "individual_evaluations": all_results
        }
