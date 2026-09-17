"""
Trivial Baseline: Majority Class Classifier, Canned Static Reply, Always Escalate.
"""

from typing import Dict, Any
from src.config import IntentCategory, EscalationReason


class TrivialBaselineAgent:
    """
    Trivial Baseline Agent:
    - Intent: Always predicts majority class ('ios_update_os_bugs').
    - Reply: Static generic template reply.
    - Escalation: Always escalates to human.
    """

    def __init__(self):
        self.majority_intent = IntentCategory.IOS_UPDATE_OS_BUGS.value
        self.static_reply = "@Customer Thanks for reaching out to Apple Support. Please send us a DM with your device details so we can help: https://t.co/GDrqU22YpT"

    def process_message(self, customer_message: str) -> Dict[str, Any]:
        return {
            "customer_query": customer_message,
            "predicted_intent": self.majority_intent,
            "intent_confidence": 0.125,
            "is_confident": False,
            "draft_reply": self.static_reply,
            "auto_handle": False,
            "escalation_reason": EscalationReason.LOW_CONFIDENCE_OR_AMBIGUOUS.value,
            "escalation_explanation": "Trivial Baseline default: always escalate every customer message."
        }
