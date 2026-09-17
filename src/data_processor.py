"""
Data Processor and Knowledge Base Builder for Twitter Customer Support Dataset.
"""

import re
import html
import json
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import pandas as pd

from src.config import IntentCategory, INTENT_REGISTRY

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class DataProcessor:
    """Preprocesses raw Twitter support text, reconstructs threads, and extracts intent indicators."""

    @staticmethod
    def clean_tweet_text(text: str, mask_author_ids: bool = True) -> str:
        """Cleans and standardizes raw tweet text."""
        if not isinstance(text, str):
            return ""

        # Unescape HTML entities
        text = html.unescape(text)

        # Standardize invisible unicode characters and byte-order markers (like \ufe0f)
        text = text.replace("\ufe0f", "").replace("\u200b", "").replace("\xa0", " ")

        # Optionally normalize anonymized numeric user mentions (@115854 -> @Customer)
        if mask_author_ids:
            text = re.sub(r"@\d+", "@Customer", text)

        # Normalize multiple spaces and newlines
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def extract_device_context(text: str) -> Dict[str, Optional[str]]:
        """Extracts device model and iOS version mentions from tweet text."""
        text_lower = text.lower()
        context = {
            "device_model": None,
            "ios_version": None,
            "has_error_code": False,
            "error_code": None
        }

        # Match iPhone models
        iphone_match = re.search(r"iphone\s*(x|se|\d{1,2}(?:\s*(?:plus|pro|max|mini))?)", text_lower)
        if iphone_match:
            context["device_model"] = f"iPhone {iphone_match.group(1).upper()}"
        elif "ipad" in text_lower:
            context["device_model"] = "iPad"
        elif "apple watch" in text_lower or "watch" in text_lower:
            context["device_model"] = "Apple Watch"
        elif "macbook" in text_lower or "mac" in text_lower:
            context["device_model"] = "Mac"

        # Match iOS versions
        ios_match = re.search(r"(?:ios|update|version)\s*(\d{1,2}(?:\.\d{1,2})*)", text_lower)
        if ios_match:
            context["ios_version"] = f"iOS {ios_match.group(1)}"

        # Match error codes
        err_match = re.search(r"error\s*(?:code)?\s*(-?\d{3,5})", text_lower)
        if err_match:
            context["has_error_code"] = True
            context["error_code"] = err_match.group(1)

        return context

    @classmethod
    def extract_brand_dialogues(
        cls,
        csv_path: str,
        brand_id: str = "AppleSupport",
        max_rows: Optional[int] = 500000
    ) -> List[Dict[str, any]]:
        """
        Parses twcs.csv and reconstructs paired (customer inbound -> brand outbound) dialogues.
        """
        logger.info(f"Loading dataset from {csv_path} for brand {brand_id}...")
        df = pd.read_csv(csv_path, nrows=max_rows)

        # Filter brand outbound tweets
        brand_replies = df[(df['author_id'] == brand_id) & (df['inbound'] == False)]
        logger.info(f"Found {len(brand_replies)} outbound replies for {brand_id}.")

        # Create quick index of tweets by tweet_id
        tweet_lookup = {}
        for _, row in df.iterrows():
            tweet_lookup[int(row['tweet_id'])] = row

        dialogues = []
        for _, brand_row in brand_replies.iterrows():
            if pd.isna(brand_row['in_response_to_tweet_id']):
                continue
            try:
                parent_id = int(brand_row['in_response_to_tweet_id'])
                if parent_id in tweet_lookup:
                    parent_tweet = tweet_lookup[parent_id]
                    if bool(parent_tweet['inbound']) == True:
                        cleaned_customer = cls.clean_tweet_text(str(parent_tweet['text']))
                        cleaned_agent = cls.clean_tweet_text(str(brand_row['text']))

                        # Ignore very short or empty messages
                        if len(cleaned_customer) < 10 or len(cleaned_agent) < 10:
                            continue

                        dialogues.append({
                            "customer_tweet_id": int(parent_tweet['tweet_id']),
                            "customer_text": cleaned_customer,
                            "agent_tweet_id": int(brand_row['tweet_id']),
                            "agent_text": cleaned_agent,
                            "context": cls.extract_device_context(cleaned_customer)
                        })
            except Exception as e:
                continue

        logger.info(f"Extracted {len(dialogues)} high-quality paired dialogues for {brand_id}.")
        return dialogues


def build_knowledge_base_from_data(
    dialogues: List[Dict[str, any]],
    output_path: str = "data/historical_knowledge_base.json"
):
    """
    Builds a curated historical knowledge base indexed by domain intent and verified resolution patterns.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Core verified troubleshooting resolutions & policies
    knowledge_base = {
        "brand": "@AppleSupport",
        "verified_playbooks": [
            {
                "intent": IntentCategory.BATTERY_POWER.value,
                "title": "Battery Drain & Power Diagnostics",
                "trigger_keywords": ["battery", "drain", "charge", "health", "overheating"],
                "resolution_strategy": "Ask for current iOS version and device model. Guide customer to check Settings > Battery to inspect battery usage by app and Settings > Battery > Battery Health for maximum capacity. Suggest force restart and 24-48h post-update stabilization.",
                "verified_steps": [
                    "Check Settings > General > About for current iOS version.",
                    "Review Settings > Battery > Battery Health & Charging.",
                    "Identify heavy background apps under Battery Usage by App.",
                    "Perform a forced restart: Press & quickly release Volume Up, then Volume Down, then hold Side button until Apple logo appears."
                ],
                "sample_historical_replies": [
                    "@Customer We're here to help with your battery. What iOS version is your device running under Settings > General > About? Also check Settings > Battery for any apps using high background power.",
                    "@Customer Battery life is important to us. Does your battery drain during active use or while on standby? Let us know your model and iOS version.",
                    "@Customer Overheating during fast charging can happen temporarily. Check Settings > Battery > Battery Health to confirm peak performance capability."
                ]
            },
            {
                "intent": IntentCategory.IOS_UPDATE_OS_BUGS.value,
                "title": "iOS Update Errors, Freezing & Keyboard Glitches",
                "trigger_keywords": ["update", "freeze", "slow", "lag", "autocorrect", "keyboard", "boot loop"],
                "resolution_strategy": "Acknowledge issue post-update. For keyboard text replacement bug ('I' to symbols), recommend Settings > General > Keyboard > Text Replacement. For system lag, recommend checking available storage (Settings > General > iPhone Storage) and rebooting.",
                "verified_steps": [
                    "For keyboard glitches: Settings > General > Keyboard > Text Replacement, add Phrase: 'I' and Shortcut: 'i'.",
                    "For system freezing: Force restart device to clear cache.",
                    "For storage lag: Ensure at least 10% of internal storage is free in Settings > General > iPhone Storage.",
                    "For failed update / error code: Connect to computer (Finder / iTunes) and run Update in Recovery Mode."
                ],
                "sample_historical_replies": [
                    "@Customer We understand lag after an update can be frustrating. Let's make sure you have at least 5GB free storage in Settings > General > iPhone Storage and try a force restart.",
                    "@Customer If you are seeing the autocorrect symbol glitch with 'I', go to Settings > General > Keyboard > Text Replacement and set 'I' as phrase and 'i' as shortcut.",
                    "@Customer If your device is stuck on the Apple logo, connect to your computer and update via Finder/iTunes in Recovery Mode."
                ]
            },
            {
                "intent": IntentCategory.CONNECTIVITY_NETWORK.value,
                "title": "WiFi, Bluetooth & Cellular Signal Troubleshooting",
                "trigger_keywords": ["wifi", "bluetooth", "cellular", "no service", "airpods", "airdrop"],
                "resolution_strategy": "Guide customer through network isolation: toggle Airplane Mode, restart router/device, and perform Reset Network Settings. For Bluetooth pairing, forget device and re-pair.",
                "verified_steps": [
                    "Toggle Airplane Mode ON for 15 seconds, then OFF.",
                    "Settings > General > Transfer or Reset iPhone > Reset > Reset Network Settings.",
                    "For Bluetooth: Settings > Bluetooth > tap 'i' next to device > Forget This Device, then put accessory into pairing mode.",
                    "For 'No Service': Remove and re-insert SIM card / check Carrier Settings Update."
                ],
                "sample_historical_replies": [
                    "@Customer Let's get your connection back up! Try heading to Settings > General > Reset > Reset Network Settings (note: this clears saved Wi-Fi passwords) and test again.",
                    "@Customer Having Bluetooth drops with your AirPods? Try placing them in the case, hold the back button for 15s to reset, and re-pair with your iPhone.",
                    "@Customer If your iPhone says No Service, check Settings > General > About for a Carrier Settings update prompt, or try reseating your SIM card."
                ]
            },
            {
                "intent": IntentCategory.APPLE_ID_ICLOUD_SECURITY.value,
                "title": "Apple ID Lockout, 2FA & iCloud Security",
                "trigger_keywords": ["apple id", "icloud", "locked", "passcode", "password", "2fa", "recovery"],
                "resolution_strategy": "Security-sensitive intent requiring DM or iforgot.apple.com self-service. Guide to iforgot.apple.com for automated account recovery. For active lockouts requiring verification, direct to private DM link.",
                "verified_steps": [
                    "Direct to iforgot.apple.com to reset password or unlock Apple ID.",
                    "For 2FA missing code: Use 'Didn't get a verification code?' on trusted device or SMS.",
                    "Do NOT request passwords or sensitive credentials over public Twitter.",
                    "Escalate to DM with secure Apple Support link if account recovery fails."
                ],
                "sample_historical_replies": [
                    "@Customer Account security is paramount. You can securely unlock your Apple ID or reset your password at https://iforgot.apple.com. Let us know if you need further help!",
                    "@Customer We want to help protect your Apple ID. Please meet us in DM so we can verify your situation securely without sharing private details publicly: https://t.co/GDrqU22YpT",
                    "@Customer If you are locked out of iCloud, please use https://iforgot.apple.com from any browser or DM us for guided account recovery."
                ]
            },
            {
                "intent": IntentCategory.APP_STORE_BILLING_SUBSCRIPTIONS.value,
                "title": "App Store Charges, Subscriptions & Refund Routing",
                "trigger_keywords": ["charged", "refund", "subscription", "bill", "purchase", "apple music"],
                "resolution_strategy": "Financial/billing intent. Guide user to reportaproblem.apple.com for official self-serve refund requests. Direct subscription management to Settings > [Name] > Subscriptions. Escalate to DM/Human for disputed duplicate charges.",
                "verified_steps": [
                    "Direct to https://reportaproblem.apple.com to request refund and review purchase history.",
                    "Direct to Settings > Apple ID > Subscriptions to view and cancel recurring active subscriptions.",
                    "Escalate to human support for credit card disputes or unauthorized account access."
                ],
                "sample_historical_replies": [
                    "@Customer We can help clarify this charge. You can review your purchase history and request a refund directly at https://reportaproblem.apple.com.",
                    "@Customer To view or cancel active subscriptions, open Settings > tap your Name > Subscriptions. If you see an unrecognized charge, DM us to investigate.",
                    "@Customer For billing and refund assistance, please visit https://reportaproblem.apple.com or send us a DM so we can look into your account details."
                ]
            },
            {
                "intent": IntentCategory.HARDWARE_PHYSICAL_DAMAGE.value,
                "title": "Screen Damage, Water Exposure & Genius Bar Repairs",
                "trigger_keywords": ["cracked", "screen", "broken", "water", "repair", "genius bar", "applecare"],
                "resolution_strategy": "Hardware repair triage. Advise against powering on water-damaged devices. Guide user to check AppleCare coverage (checkcoverage.apple.com) and book a Genius Bar appointment at getsupport.apple.com or the Apple Support app.",
                "verified_steps": [
                    "For water damage: Do not plug into power; allow to dry completely in a well-ventilated dry area.",
                    "Check warranty/AppleCare+ status at https://checkcoverage.apple.com.",
                    "Schedule an in-person Genius Bar appointment or mail-in repair via https://getsupport.apple.com.",
                    "Escalate to human support specialist for repair quote and authorized service provider routing."
                ],
                "sample_historical_replies": [
                    "@Customer We can help arrange a repair for your damaged screen. Check your AppleCare coverage at https://checkcoverage.apple.com and book a Genius Bar appointment at https://getsupport.apple.com.",
                    "@Customer If your device had liquid contact, please unplug it immediately and do not charge it. You can schedule an inspection at your nearest Apple Store via https://getsupport.apple.com.",
                    "@Customer We'd be glad to help set up a repair reservation for you. Send us a DM with your location so we can find the nearest authorized service provider."
                ]
            },
            {
                "intent": IntentCategory.GENERAL_FAQ_HOWTO.value,
                "title": "General How-To, Setup & Feature Guidance",
                "trigger_keywords": ["how to", "screenshot", "transfer", "trade in", "setup", "feature"],
                "resolution_strategy": "Provide direct step-by-step instructions for standard iOS features or data transfer (Quick Start).",
                "verified_steps": [
                    "Screenshot on iPhone X/Face ID: Press Side button + Volume Up simultaneously.",
                    "Data Transfer: Use Quick Start by bringing new and old devices near each other during initial setup.",
                    "Warranty check: Settings > General > About > AppleCare Coverage."
                ],
                "sample_historical_replies": [
                    "@Customer To take a screenshot on iPhone X or newer, press the Side button and Volume Up button at the same time, then quickly release both!",
                    "@Customer Setting up a new iPhone is easy with Quick Start! Turn on your new device and place it near your old device with Bluetooth enabled to start automatic transfer.",
                    "@Customer You can check your trade-in estimate directly on apple.com/shop/trade-in or in the Apple Store app!"
                ]
            },
            {
                "intent": IntentCategory.OUT_OF_SCOPE_COMPLAINT.value,
                "title": "Polite De-escalation & Out-of-Scope Feedback",
                "trigger_keywords": ["worst", "hate", "feedback", "ceo", "stock"],
                "resolution_strategy": "De-escalate politely. Acknowledge customer sentiment without being defensive. Offer general assistance if there is an underlying issue, or provide feedback link at apple.com/feedback.",
                "verified_steps": [
                    "Acknowledge sentiment politely.",
                    "Offer assistance if there is an active technical question.",
                    "Provide product feedback link: https://www.apple.com/feedback."
                ],
                "sample_historical_replies": [
                    "@Customer We're sorry to hear you're feeling frustrated. If you are experiencing a specific issue with your device, please let us know and we'd love to help.",
                    "@Customer We always appreciate customer feedback. You can share your suggestions directly with our engineering teams at https://www.apple.com/feedback."
                ]
            }
        ],
        "indexed_historical_dialogues": dialogues[:2000] if dialogues else []
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(knowledge_base, f, indent=2, ensure_ascii=False)
    logger.info(f"Successfully saved knowledge base with {len(knowledge_base['verified_playbooks'])} playbooks and {len(knowledge_base['indexed_historical_dialogues'])} indexed dialogues to {output_path}")
