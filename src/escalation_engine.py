"""
Escalation and Safety Guardrail Engine for AppleSupport AI Agent.
Determines whether an incoming customer inquiry should be auto-handled or escalated
to a human specialist with a rigorous, stated reason and explanation.
"""

import re
from typing import Dict, Any, Tuple
from src.config import (
    IntentCategory,
    EscalationReason,
    INTENT_REGISTRY,
    CONFIDENCE_THRESHOLD_AUTO_HANDLE,
    HIGH_AGITATION_KEYWORDS
)
from src.data_processor import DataProcessor


class EscalationEngine:
    """Evaluates customer messages against safety policies, intent boundaries, and risk signals."""

    REPEATED_CONTACT_PATTERNS = [
        r"\b(?:case\s*#?\s*\d{5,10})\b",
        r"\b(?:repair\s*#?\s*[A-Za-z0-9]{6,12})\b",
        r"\b(?:contacted|called|messaged|tweeted)\s*(?:you\s*)?(?:\d+|multiple|several|many)\s*times\b",
        r"\b(?:already\s*tried\s*(?:everything|all\s*of\s*that))\b",
        r"\b(?:still\s*not\s*fixed|still\s*haven't\s*heard)\b",
        r"\b(?:waited|waiting)\s*(?:for\s*)?(?:\d+\s*(?:days|weeks|hours))\b"
    ]

    HARDWARE_HAZARD_PATTERNS = [
        r"\b(?:swelling|swollen|bulging|expanded)\s*(?:battery)?\b",
        r"\b(?:smoke|spark|sparking|fire|burned|burning\s*hot|exploded)\b",
        r"\b(?:shattered|cracked\s*screen|broken\s*glass|water\s*damage|liquid\s*damage)\b"
    ]

    def evaluate(
        self,
        text: str,
        predicted_intent: str,
        intent_confidence: float,
        matched_keywords: list = None
    ) -> Dict[str, Any]:
        """
        Evaluates the customer tweet and returns auto-handle decision, stated reason, and explanation.
        """
        cleaned = DataProcessor.clean_tweet_text(text)
        text_lower = cleaned.lower()

        # 1. Check for Severe Thermal/Hardware Safety Hazards
        for pattern in self.HARDWARE_HAZARD_PATTERNS:
            if re.search(pattern, text_lower):
                return {
                    "auto_handle": False,
                    "escalation_reason": EscalationReason.PHYSICAL_HARDWARE_REPAIR.value,
                    "explanation": "Critical hardware/safety hazard detected (physical damage, liquid contact, or thermal/battery risk). Requires human technical intake."
                }

        # 2. Check for High Agitation, Hostility, or Legal Threats
        found_agitation = [kw for kw in HIGH_AGITATION_KEYWORDS if re.search(r"\b" + re.escape(kw) + r"\b", text_lower)]
        if found_agitation:
            return {
                "auto_handle": False,
                "escalation_reason": EscalationReason.HIGH_USER_AGITATION_OR_LEGAL.value,
                "explanation": f"High customer distress/legal escalation signal detected ('{found_agitation[0]}'). Requires empathetic human advisor intervention."
            }

        # 3. Check for Repeated Unresolved Inquiries or Active Case Numbers
        for pattern in self.REPEATED_CONTACT_PATTERNS:
            if re.search(pattern, text_lower):
                return {
                    "auto_handle": False,
                    "escalation_reason": EscalationReason.REPEATED_UNRESOLVED_ISSUE.value,
                    "explanation": "Customer cites recurring unresolved contact or existing support case ID. Requires specialist case review."
                }

        # 4. Check for Intent-Specific Policy Escalations (PII / Financial / Hardware)
        try:
            intent_enum = IntentCategory(predicted_intent)
        except ValueError:
            intent_enum = IntentCategory.OUT_OF_SCOPE_COMPLAINT

        if intent_enum == IntentCategory.APPLE_ID_ICLOUD_SECURITY:
            # Check if it's a simple informational FAQ (e.g. how to turn on 2FA) vs active lockout
            is_generic_faq = any(p in text_lower for p in ["how to turn on", "how do i turn on", "how much does icloud cost", "family sharing"])
            if not is_generic_faq:
                return {
                    "auto_handle": False,
                    "escalation_reason": EscalationReason.ACCOUNT_SECURITY_OR_PII.value,
                    "explanation": "Apple ID account lockout / credential verification involves private PII. Directing to secure self-service / private DM verification."
                }

        elif intent_enum == IntentCategory.APP_STORE_BILLING_SUBSCRIPTIONS:
            # Check if it's simple cancel instructions vs actual refund / unauthorized charge claim
            is_refund_or_dispute = any(p in text_lower for p in ["charged", "refund", "stole", "unauthorized", "dispute", "double bill", "bank statement", "billed twice"])
            if is_refund_or_dispute:
                return {
                    "auto_handle": False,
                    "escalation_reason": EscalationReason.BILLING_REFUND_AUTHORIZATION.value,
                    "explanation": "Financial transaction or refund claim requires account verification and human billing authorization."
                }

        elif intent_enum == IntentCategory.HARDWARE_PHYSICAL_DAMAGE:
            return {
                "auto_handle": False,
                "escalation_reason": EscalationReason.PHYSICAL_HARDWARE_REPAIR.value,
                "explanation": "Hardware damage or repair reservation requires physical inspection and Genius Bar / mail-in logistics."
            }

        elif intent_enum == IntentCategory.OUT_OF_SCOPE_COMPLAINT:
            # If low confidence or pure venting, escalate for human triage or close politely
            if intent_confidence < CONFIDENCE_THRESHOLD_AUTO_HANDLE or len(cleaned.split()) < 3:
                return {
                    "auto_handle": False,
                    "escalation_reason": EscalationReason.LOW_CONFIDENCE_OR_AMBIGUOUS.value,
                    "explanation": "Inquiry is ambiguous, out-of-scope, or lacks actionable technical details. Escalating for manual triage."
                }

        # 5. Check Confidence Threshold
        if intent_confidence < CONFIDENCE_THRESHOLD_AUTO_HANDLE:
            return {
                "auto_handle": False,
                "escalation_reason": EscalationReason.LOW_CONFIDENCE_OR_AMBIGUOUS.value,
                "explanation": f"Intent classification confidence ({intent_confidence:.2f}) below threshold ({CONFIDENCE_THRESHOLD_AUTO_HANDLE:.2f}). Escalating to avoid misdiagnosis."
            }

        # 6. Default Safe Auto-Handling
        return {
            "auto_handle": True,
            "escalation_reason": EscalationReason.NONE.value,
            "explanation": "Standard software troubleshooting or how-to inquiry with established safe self-serve resolution playbook."
        }
