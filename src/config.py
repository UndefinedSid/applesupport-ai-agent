"""
Configuration and Taxonomy Definitions for AppleSupport AI Agent.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set


class IntentCategory(str, Enum):
    BATTERY_POWER = "battery_power"
    IOS_UPDATE_OS_BUGS = "ios_update_os_bugs"
    CONNECTIVITY_NETWORK = "connectivity_network"
    APPLE_ID_ICLOUD_SECURITY = "apple_id_icloud_security"
    APP_STORE_BILLING_SUBSCRIPTIONS = "app_store_billing_subscriptions"
    HARDWARE_PHYSICAL_DAMAGE = "hardware_physical_damage"
    GENERAL_FAQ_HOWTO = "general_faq_howto"
    OUT_OF_SCOPE_COMPLAINT = "out_of_scope_complaint"


class EscalationReason(str, Enum):
    ACCOUNT_SECURITY_OR_PII = "account_security_or_pii"
    BILLING_REFUND_AUTHORIZATION = "billing_refund_authorization"
    PHYSICAL_HARDWARE_REPAIR = "physical_hardware_repair"
    HIGH_USER_AGITATION_OR_LEGAL = "high_user_agitation_or_legal"
    LOW_CONFIDENCE_OR_AMBIGUOUS = "low_confidence_or_ambiguous"
    REPEATED_UNRESOLVED_ISSUE = "repeated_unresolved_issue"
    NONE = "none"


@dataclass
class IntentMetadata:
    name: str
    description: str
    sample_phrases: List[str]
    keywords: List[str]
    requires_pii: bool
    requires_hardware_inspection: bool
    default_auto_handle: bool


INTENT_REGISTRY: Dict[IntentCategory, IntentMetadata] = {
    IntentCategory.BATTERY_POWER: IntentMetadata(
        name="Battery & Power Management",
        description="Battery drain, rapid discharge, charging problems, unexpected shutdowns, battery health percentage drops, overheating while charging.",
        sample_phrases=[
            "My iPhone battery drains from 100% to 20% in two hours.",
            "Phone won't charge past 80% after latest update.",
            "My phone gets burning hot whenever I plug it into the charger.",
            "Battery health dropped 5% in one week, what is happening?",
            "Phone randomly shuts down when battery reaches 30%."
        ],
        keywords=["battery", "drain", "charge", "charging", "drains", "shutdown", "health", "overheating", "overheat", "percentage", "charger", "lightning", "magsafe"],
        requires_pii=False,
        requires_hardware_inspection=False,
        default_auto_handle=True
    ),
    IntentCategory.IOS_UPDATE_OS_BUGS: IntentMetadata(
        name="iOS Updates & OS Software Glitches",
        description="iOS installation failures, boot loops, system lag after update, keyboard autocorrect bugs, UI freezing, app crashes due to OS incompatibility.",
        sample_phrases=[
            "Since updating to iOS 11 my phone is extremely laggy and slow.",
            "Typing the letter 'I' turns into a weird symbol and question mark box.",
            "My phone is stuck on the Apple logo after downloading the update.",
            "iOS update failed with error code 4013.",
            "Keyboard keeps disappearing when I open Messages."
        ],
        keywords=["ios", "update", "updated", "updating", "freeze", "freezing", "lag", "laggy", "slow", "bug", "glitch", "apple logo", "stuck", "keyboard", "autocorrect", "crash", "os"],
        requires_pii=False,
        requires_hardware_inspection=False,
        default_auto_handle=True
    ),
    IntentCategory.CONNECTIVITY_NETWORK: IntentMetadata(
        name="Connectivity, WiFi, Bluetooth & Cellular",
        description="WiFi disconnecting, Bluetooth pairing drops (AirPods, car), Cellular 'No Service' or 'Searching', AirDrop failures, hotspot issues.",
        sample_phrases=[
            "My iPhone keeps dropping WiFi connection every 5 minutes.",
            "AirPods won't connect to my iPhone via Bluetooth.",
            "My phone shows No Service even though my cellular bill is paid.",
            "Personal Hotspot is not turning on.",
            "AirDrop cannot discover nearby devices."
        ],
        keywords=["wifi", "wi-fi", "bluetooth", "cellular", "airpods", "airdrop", "hotspot", "no service", "searching", "signal", "connect", "connection", "disconnecting", "paired"],
        requires_pii=False,
        requires_hardware_inspection=False,
        default_auto_handle=True
    ),
    IntentCategory.APPLE_ID_ICLOUD_SECURITY: IntentMetadata(
        name="Apple ID, iCloud & Account Security",
        description="Apple ID locked for security reasons, 2FA verification codes not arriving, iCloud storage full/syncing errors, forgotten passcode/Apple ID password, activation lock.",
        sample_phrases=[
            "My Apple ID has been locked for security reasons, how do I unlock it?",
            "I'm not receiving my two-factor authentication code on my phone.",
            "Photos are not uploading to iCloud even though I have 50GB plan.",
            "Forgot my Apple ID password and account recovery is taking too long.",
            "Device is stuck on Activation Lock screen after factory reset."
        ],
        keywords=["apple id", "icloud", "locked", "passcode", "password", "2fa", "two-factor", "verification code", "activation lock", "account", "security", "recovery", "sync"],
        requires_pii=True,
        requires_hardware_inspection=False,
        default_auto_handle=False  # Security & account verification requires DM / Human agent
    ),
    IntentCategory.APP_STORE_BILLING_SUBSCRIPTIONS: IntentMetadata(
        name="App Store, Subscriptions & Billing",
        description="Unauthorized credit card charges, accidental subscription renewals, requesting refunds for App Store purchases, payment method declined.",
        sample_phrases=[
            "I was charged $9.99 for an app subscription I already cancelled.",
            "How do I request a refund for an accidental in-app purchase by my child?",
            "My credit card was charged twice for Apple Music this month.",
            "Payment method declined when trying to download a free app.",
            "Where can I view and manage active Apple subscriptions?"
        ],
        keywords=["charged", "charge", "refund", "subscription", "bill", "billing", "app store", "apple music", "itunes", "payment", "receipt", "purchase", "renewed", "money"],
        requires_pii=True,
        requires_hardware_inspection=False,
        default_auto_handle=False  # Financial transaction and refunds require human / DM auth
    ),
    IntentCategory.HARDWARE_PHYSICAL_DAMAGE: IntentMetadata(
        name="Hardware Defect, Screen Damage & Repairs",
        description="Cracked screen, liquid/water damage, swollen battery, broken speaker/microphone, physical button failure, Genius Bar appointment request.",
        sample_phrases=[
            "Dropped my iPhone and screen is shattered, how much for repair?",
            "My phone fell in water and the speaker sounds distorted and crackling.",
            "Power button is physically jammed and won't click.",
            "How do I book a Genius Bar appointment at the nearest Apple Store?",
            "Back glass broke, is it covered under AppleCare+?"
        ],
        keywords=["screen", "cracked", "shattered", "water", "dropped", "broken", "speaker", "microphone", "repair", "genius bar", "applecare", "physical", "hardware", "swollen", "button"],
        requires_pii=False,
        requires_hardware_inspection=True,
        default_auto_handle=False  # Hardware triage & repair booking requires human / retail routing
    ),
    IntentCategory.GENERAL_FAQ_HOWTO: IntentMetadata(
        name="General FAQ & Feature Guidance",
        description="How to take screenshot, setting up new device, transferring data, checking warranty status, finding Apple store hours, feature instructions.",
        sample_phrases=[
            "How do I take a screenshot on iPhone X without a home button?",
            "How can I transfer data from my old Android to new iPhone?",
            "Where can I check if my device is still under limited warranty?",
            "How do I enable Dark Mode on iOS?",
            "What is the trade-in value for iPhone 7?"
        ],
        keywords=["how to", "how do i", "screenshot", "transfer", "trade in", "warranty", "setup", "feature", "settings", "guide", "tutorial", "instructions"],
        requires_pii=False,
        requires_hardware_inspection=False,
        default_auto_handle=True
    ),
    IntentCategory.OUT_OF_SCOPE_COMPLAINT: IntentMetadata(
        name="Out of Scope, Rants & General Feedback",
        description="Unintelligible tweets, general rants without technical issue, stock price comments, jokes, competitor banter, abusive venting without specific request.",
        sample_phrases=[
            "Apple is the worst company ever, switching to Android immediately.",
            "Tim Cook please make phones cheaper lol.",
            "Why is Apple stock dropping today?",
            "Just saw the new keynote, amazing presentation!",
            "I hate technology so much everything is broken."
        ],
        keywords=["worst", "hate", "stock", "tim cook", "keynote", "android", "samsung", "lol", "garbage", "trash", "ceo", "shares"],
        requires_pii=False,
        requires_hardware_inspection=False,
        default_auto_handle=False  # Requires human triage or polite non-technical closure
    ),
}

# Brand guidelines
BRAND_NAME = "@AppleSupport"
BRAND_VOICE = {
    "tone": "Empathetic, clear, calm, professional, and solutions-oriented.",
    "max_length_chars": 280,  # Standard Twitter character limit
    "dm_link_placeholder": "https://t.co/GDrqU22YpT",
    "signature": "",
    "key_habits": [
        "Acknowledge the customer's frustration with empathy without accepting legal liability.",
        "Always ask for the exact iOS version (Settings > General > About) and device model if not specified.",
        "Provide safe, concrete, non-destructive troubleshooting steps (e.g. Restart, Reset Network Settings, Check for updates).",
        "Direct the user to DM when account details, serial numbers, Apple IDs, or sensitive verification are involved.",
        "Never invent non-existent iOS settings menus or guarantee free out-of-warranty hardware replacements."
    ]
}

# Escalation thresholds (for 8-class classification, >0.50 represents absolute majority over all 7 alternatives)
CONFIDENCE_THRESHOLD_AUTO_HANDLE = 0.50
HIGH_AGITATION_KEYWORDS = {
    "lawsuit", "lawyer", "attorney", "sue", "scam", "fraud", "police", "stolen",
    "furious", "unacceptable", "fucking", "bullshit", "disaster", "incompetent", "robbed"
}
