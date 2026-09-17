"""
Automated Evaluation Metrics for Intent Classification, Escalation Triage, and Reply Quality.
"""

import math
import collections
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class EvaluationMetrics:
    """Computes automated classification, triage, and text-generation quality metrics."""

    @staticmethod
    def compute_classification_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
        """Calculates accuracy, macro F1, and weighted F1 for intent classification."""
        acc = accuracy_score(y_true, y_pred)
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        return {
            "accuracy": round(float(acc), 4),
            "macro_precision": round(float(p_macro), 4),
            "macro_recall": round(float(r_macro), 4),
            "macro_f1": round(float(f1_macro), 4),
            "weighted_f1": round(float(f1_weighted), 4)
        }

    @staticmethod
    def compute_escalation_metrics(y_true_auto: List[bool], y_pred_auto: List[bool]) -> Dict[str, float]:
        """Calculates accuracy, precision, recall, and F1 for the binary auto-handle / escalate decision."""
        acc = accuracy_score(y_true_auto, y_pred_auto)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true_auto, y_pred_auto, average="binary", pos_label=True, zero_division=0
        )
        
        # Escalation (human intervention) specific metrics (where pos_label = False / Escalate)
        p_esc, r_esc, f1_esc, _ = precision_recall_fscore_support(
            [not y for y in y_true_auto], [not y for y in y_pred_auto], average="binary", pos_label=True, zero_division=0
        )
        return {
            "triage_accuracy": round(float(acc), 4),
            "auto_handle_precision": round(float(p), 4),
            "auto_handle_recall": round(float(r), 4),
            "auto_handle_f1": round(float(f1), 4),
            "escalation_precision": round(float(p_esc), 4),
            "escalation_recall": round(float(r_esc), 4),
            "escalation_f1": round(float(f1_esc), 4)
        }

    @staticmethod
    def _compute_ngram_overlap(candidate_tokens: List[str], reference_tokens: List[str], n: int) -> float:
        """Helper to compute n-gram precision for BLEU/ROUGE."""
        if len(candidate_tokens) < n or len(reference_tokens) < n:
            return 0.0

        cand_ngrams = collections.Counter([
            tuple(candidate_tokens[i:i+n]) for i in range(len(candidate_tokens) - n + 1)
        ])
        ref_ngrams = collections.Counter([
            tuple(reference_tokens[i:i+n]) for i in range(len(reference_tokens) - n + 1)
        ])

        overlap = sum((cand_ngrams & ref_ngrams).values())
        return overlap / max(1, sum(cand_ngrams.values()))

    @classmethod
    def compute_bleu_and_rouge(cls, candidates: List[str], references: List[str]) -> Dict[str, float]:
        """Calculates token-level BLEU-1, BLEU-2, and ROUGE-L across all candidate vs reference pairs."""
        bleu1_scores = []
        bleu2_scores = []
        rouge_l_scores = []

        for cand, ref in zip(candidates, references):
            cand_toks = cand.lower().split()
            ref_toks = ref.lower().split()

            if not cand_toks or not ref_toks:
                bleu1_scores.append(0.0)
                bleu2_scores.append(0.0)
                rouge_l_scores.append(0.0)
                continue

            # BLEU-1 and BLEU-2
            p1 = cls._compute_ngram_overlap(cand_toks, ref_toks, 1)
            p2 = cls._compute_ngram_overlap(cand_toks, ref_toks, 2)
            
            # Brevity penalty
            bp = 1.0 if len(cand_toks) >= len(ref_toks) else math.exp(1 - len(ref_toks) / max(1, len(cand_toks)))
            bleu1_scores.append(bp * p1)
            bleu2_scores.append(bp * math.sqrt(p1 * p2) if (p1 * p2) > 0 else 0.0)

            # ROUGE-L (LCS based)
            m, n_len = len(cand_toks), len(ref_toks)
            dp = [[0] * (n_len + 1) for _ in range(m + 1)]
            for i in range(m):
                for j in range(n_len):
                    if cand_toks[i] == ref_toks[j]:
                        dp[i+1][j+1] = dp[i][j] + 1
                    else:
                        dp[i+1][j+1] = max(dp[i+1][j], dp[i][j+1])
            lcs = dp[m][n_len]
            r_lcs = lcs / max(1, n_len)
            p_lcs = lcs / max(1, m)
            f_lcs = (2 * p_lcs * r_lcs) / (p_lcs + r_lcs) if (p_lcs + r_lcs) > 0 else 0.0
            rouge_l_scores.append(f_lcs)

        return {
            "bleu_1": round(float(np.mean(bleu1_scores)), 4),
            "bleu_2": round(float(np.mean(bleu2_scores)), 4),
            "rouge_l": round(float(np.mean(rouge_l_scores)), 4)
        }

    @staticmethod
    def compute_semantic_similarity(candidates: List[str], references: List[str]) -> float:
        """Calculates TF-IDF Cosine Semantic Similarity between generated replies and target references."""
        sims = []
        vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        for cand, ref in zip(candidates, references):
            if not cand.strip() or not ref.strip():
                sims.append(0.0)
                continue
            try:
                matrix = vec.fit_transform([cand, ref])
                sim = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
                sims.append(float(sim))
            except Exception:
                sims.append(0.0)
        return round(float(np.mean(sims)), 4)

    @staticmethod
    def compute_compliance_metrics(replies: List[str], escalations: List[bool]) -> Dict[str, float]:
        """Measures length constraint compliance (<= 280 chars) and DM link presence on escalations."""
        length_compliant = [len(r) <= 280 for r in replies]
        dm_link_compliant = []

        for reply, is_escalated in zip(replies, escalations):
            if is_escalated:
                has_dm_link = "https://t.co/" in reply or "dm" in reply.lower() or "https://" in reply
                dm_link_compliant.append(has_dm_link)
            else:
                dm_link_compliant.append(True)

        return {
            "char_limit_compliance_pct": round(float(np.mean(length_compliant)) * 100, 2),
            "escalation_dm_routing_pct": round(float(np.mean(dm_link_compliant)) * 100, 2),
            "avg_reply_length_chars": round(float(np.mean([len(r) for r in replies])), 1)
        }
