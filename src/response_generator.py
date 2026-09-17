"""
Grounded Response Drafting Engine for AppleSupport AI Agent.
Drafts empathetic, brand-compliant Twitter replies grounded in historical resolution playbooks
and calibrated for Twitter character constraints (<= 280 chars).
"""

from typing import Dict, Any, Optional
from src.config import IntentCategory, BRAND_VOICE, EscalationReason
from src.data_processor import DataProcessor


class ResponseGenerator:
    """Generates customer-support replies grounded in historical resolutions and brand guidelines."""

    def __init__(self):
        self.dm_link = BRAND_VOICE["dm_link_placeholder"]
        self.max_chars = BRAND_VOICE["max_length_chars"]

    def generate_reply(
        self,
        customer_text: str,
        predicted_intent: str,
        retrieval_context: Dict[str, Any],
        escalation_decision: Dict[str, Any],
        device_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Drafts a grounded reply tailored to the customer inquiry and triage decision.
        """
        cleaned = DataProcessor.clean_tweet_text(customer_text)
        text_lower = cleaned.lower()
        
        if device_context is None:
            device_context = DataProcessor.extract_device_context(cleaned)

        model = device_context.get("device_model")
        ios = device_context.get("ios_version")
        auto_handle = escalation_decision.get("auto_handle", True)
        reason = escalation_decision.get("escalation_reason", EscalationReason.NONE.value)
        playbook = retrieval_context.get("playbook")

        # 1. Handling Escalated Inquiries
        if not auto_handle:
            if reason == EscalationReason.ACCOUNT_SECURITY_OR_PII.value:
                if "iforgot" in text_lower or "password" in text_lower or "locked" in text_lower:
                    reply = f"@Customer We want to help secure your account. You can unlock your Apple ID or reset your password at https://iforgot.apple.com, or meet us in DM so we can assist safely: {self.dm_link}"
                else:
                    reply = f"@Customer Your account security is paramount. To protect your details, please join us in a DM with your Apple ID info: {self.dm_link}"

            elif reason == EscalationReason.BILLING_REFUND_AUTHORIZATION.value:
                reply = f"@Customer We can help with your billing inquiry. You can review purchases and request refunds at https://reportaproblem.apple.com, or send us a DM to investigate: {self.dm_link}"

            elif reason == EscalationReason.PHYSICAL_HARDWARE_REPAIR.value:
                if any(w in text_lower for w in ["swelling", "swollen", "fire", "burned", "smoke"]):
                    reply = f"@Customer Safety is our top priority. Please disconnect the device from power immediately, avoid pressing the screen, and DM us right away: {self.dm_link}"
                elif "water" in text_lower or "pool" in text_lower:
                    reply = f"@Customer If liquid contact occurred, unplug the device and let it dry. You can check coverage & set up service at https://getsupport.apple.com or DM us: {self.dm_link}"
                else:
                    reply = f"@Customer We can help arrange a repair. Check your AppleCare coverage at https://checkcoverage.apple.com and reserve a Genius Bar visit at https://getsupport.apple.com, or DM us: {self.dm_link}"

            elif reason == EscalationReason.HIGH_USER_AGITATION_OR_LEGAL.value:
                reply = f"@Customer We understand your frustration and want to make this right. Please send us a direct message so a senior advisor can assist you directly: {self.dm_link}"

            elif reason == EscalationReason.REPEATED_UNRESOLVED_ISSUE.value:
                reply = f"@Customer We're sorry for the ongoing trouble. Please send us a DM with your case number or details so we can take a closer look together: {self.dm_link}"

            else:  # LOW_CONFIDENCE_OR_AMBIGUOUS or OUT_OF_SCOPE
                reply = f"@Customer We'd love to help! Could you DM us more details about what's happening and your device model? {self.dm_link}"

        # 2. Handling Auto-Handled Technical Troubleshooting & FAQs
        else:
            try:
                intent_enum = IntentCategory(predicted_intent)
            except ValueError:
                intent_enum = IntentCategory.GENERAL_FAQ_HOWTO

            if intent_enum == IntentCategory.BATTERY_POWER:
                if not ios and not model:
                    reply = f"@Customer Battery life is important. Which iPhone model and iOS version (Settings > General > About) are you on? Also check Settings > Battery for heavy background apps."
                else:
                    reply = f"@Customer Let's check your battery! Check Settings > Battery > Battery Health for capacity. A forced restart (Vol Up, Vol Down, hold Side button) can also help clear rogue processes."

            elif intent_enum == IntentCategory.IOS_UPDATE_OS_BUGS:
                if "autocorrect" in text_lower or "letter 'i'" in text_lower or "weird symbol" in text_lower or "mail app" in text_lower:
                    reply = f"@Customer For the keyboard autocorrect glitch, go to Settings > General > Keyboard > Text Replacement and add Phrase 'I' with Shortcut 'i'!"
                elif not ios:
                    reply = f"@Customer We'd love to help with this glitch. Which version of iOS is installed in Settings > General > About? Does restarting your device resolve it temporarily?"
                else:
                    reply = f"@Customer Thanks for the details. Make sure you have at least 5GB free storage in Settings > General > iPhone Storage, and try a force restart to refresh system caches."

            elif intent_enum == IntentCategory.CONNECTIVITY_NETWORK:
                if "airpods" in text_lower:
                    reply = f"@Customer For AirPods connection issues, place both in the case, charge for 15m, then hold the setup button on the back for 15s to reset and re-pair."
                elif "no service" in text_lower:
                    reply = f"@Customer If your phone shows No Service, toggle Airplane Mode, check Settings > General > About for a carrier update, or re-insert your SIM."
                else:
                    reply = f"@Customer Let's get you connected. Try Settings > General > Reset > Reset Network Settings (clears Wi-Fi passwords) and reboot your router to test."

            elif intent_enum == IntentCategory.GENERAL_FAQ_HOWTO:
                if "screenshot" in text_lower and ("iphone x" in text_lower or "without a home button" in text_lower):
                    reply = f"@Customer To take a screenshot on iPhone X or newer, press the Side button and Volume Up button at the same time, then quickly release both!"
                elif "move to ios" in text_lower or "android" in text_lower:
                    reply = f"@Customer Moving from Android is easy! Download the 'Move to iOS' app from Google Play Store onto your Android and follow the setup assistant prompts on iPhone."
                elif "warranty" in text_lower or "applecare" in text_lower:
                    reply = f"@Customer You can check your warranty and AppleCare status anytime by entering your serial number at https://checkcoverage.apple.com."
                else:
                    reply = f"@Customer We're here to guide you! You can find step-by-step feature guides on support.apple.com, or let us know what specific steps you'd like help with."

            elif intent_enum == IntentCategory.OUT_OF_SCOPE_COMPLAINT:
                reply = f"@Customer We appreciate your feedback! You can share your thoughts directly with our product teams at https://www.apple.com/feedback. Let us know if we can help with anything else!"

            else:
                reply = f"@Customer We're here to help! Let us know your device model and iOS version from Settings > General > About so we can guide you on the best steps."

        # Ensure length constraint <= 280 characters
        if len(reply) > self.max_chars:
            reply = reply[:self.max_chars - 3] + "..."

        return reply
