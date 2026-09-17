"""
Simple Baseline: Naive TF-IDF Classifier, Generic Ungrounded Generation, Heuristic Keyword Escalation.
"""

from typing import Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import IntentCategory, INTENT_REGISTRY, EscalationReason
from src.data_processor import DataProcessor


class SimpleBaselineAgent:
    """
    Simple Baseline Agent:
    - Intent: Basic TF-IDF similarity over category descriptions (no exemplar tuning or keyword boosts).
    - Reply: Ungrounded static response templates per intent without retrieval or device context extraction.
    - Escalation: Naive keyword search (only escalates if 'angry', 'broken', 'money', 'lawyer' is found).
    """

    GENERIC_REPLIES = {
        IntentCategory.BATTERY_POWER.value: "@Customer Please check your battery settings or try restarting your phone.",
        IntentCategory.IOS_UPDATE_OS_BUGS.value: "@Customer Please make sure you have updated your device to the latest software version.",
        IntentCategory.CONNECTIVITY_NETWORK.value: "@Customer Please try turning your Wi-Fi or Bluetooth off and on again.",
        IntentCategory.APPLE_ID_ICLOUD_SECURITY.value: "@Customer Please visit appleid.apple.com to manage your account details.",
        IntentCategory.APP_STORE_BILLING_SUBSCRIPTIONS.value: "@Customer For billing questions, check your purchase history online.",
        IntentCategory.HARDWARE_PHYSICAL_DAMAGE.value: "@Customer For hardware repairs, visit an Apple Store near you.",
        IntentCategory.GENERAL_FAQ_HOWTO.value: "@Customer You can read our official user guides on support.apple.com.",
        IntentCategory.OUT_OF_SCOPE_COMPLAINT.value: "@Customer Thanks for your feedback regarding Apple products."
    }

    NAIVE_ESCALATE_KEYWORDS = ["angry", "furious", "broken", "money", "refund", "lawyer", "sue", "stolen"]

    def __init__(self):
        self.categories = list(INTENT_REGISTRY.keys())
        self.descriptions = [INTENT_REGISTRY[c].description for c in self.categories]
        self.vectorizer = TfidfVectorizer()
        self.doc_matrix = self.vectorizer.fit_transform(self.descriptions)

    def process_message(self, customer_message: str) -> Dict[str, Any]:
        cleaned = DataProcessor.clean_tweet_text(customer_message)
        query_vec = self.vectorizer.transform([cleaned])
        sims = cosine_similarity(query_vec, self.doc_matrix)[0]
        best_idx = sims.argmax()
        pred_intent = self.categories[best_idx].value
        confidence = float(sims[best_idx])

        # Naive keyword escalation
        text_lower = cleaned.lower()
        should_escalate = any(kw in text_lower for kw in self.NAIVE_ESCALATE_KEYWORDS)

        return {
            "customer_query": customer_message,
            "predicted_intent": pred_intent,
            "intent_confidence": round(confidence, 4),
            "is_confident": confidence > 0.20,
            "draft_reply": self.GENERIC_REPLIES.get(pred_intent, self.GENERIC_REPLIES[IntentCategory.GENERAL_FAQ_HOWTO.value]),
            "auto_handle": not should_escalate,
            "escalation_reason": EscalationReason.HIGH_USER_AGITATION_OR_LEGAL.value if should_escalate else EscalationReason.NONE.value,
            "escalation_explanation": "Simple baseline heuristic keyword filter."
        }
