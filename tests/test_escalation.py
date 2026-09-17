"""
Unit Tests for Escalation Engine.
"""

import pytest
from src.config import IntentCategory, EscalationReason
from src.escalation_engine import EscalationEngine


@pytest.fixture
def escalator():
    return EscalationEngine()


def test_escalate_swollen_battery(escalator):
    text = "My battery is swollen and pushing the screen out of the phone!"
    res = escalator.evaluate(text, IntentCategory.BATTERY_POWER.value, 0.90)
    assert res["auto_handle"] is False
    assert res["escalation_reason"] == EscalationReason.PHYSICAL_HARDWARE_REPAIR.value


def test_escalate_legal_threat(escalator):
    text = "You broke my phone, I will sue you and contact my lawyer immediately."
    res = escalator.evaluate(text, IntentCategory.IOS_UPDATE_OS_BUGS.value, 0.85)
    assert res["auto_handle"] is False
    assert res["escalation_reason"] == EscalationReason.HIGH_USER_AGITATION_OR_LEGAL.value


def test_escalate_apple_id_lockout(escalator):
    text = "My Apple ID is locked and I cannot receive verification codes."
    res = escalator.evaluate(text, IntentCategory.APPLE_ID_ICLOUD_SECURITY.value, 0.90)
    assert res["auto_handle"] is False
    assert res["escalation_reason"] == EscalationReason.ACCOUNT_SECURITY_OR_PII.value


def test_escalate_disputed_charge(escalator):
    text = "I was charged $49.99 unauthorized on my credit card."
    res = escalator.evaluate(text, IntentCategory.APP_STORE_BILLING_SUBSCRIPTIONS.value, 0.90)
    assert res["auto_handle"] is False
    assert res["escalation_reason"] == EscalationReason.BILLING_REFUND_AUTHORIZATION.value


def test_auto_handle_standard_software_faq(escalator):
    text = "How do I take a screenshot on iPhone X?"
    res = escalator.evaluate(text, IntentCategory.GENERAL_FAQ_HOWTO.value, 0.85)
    assert res["auto_handle"] is True
    assert res["escalation_reason"] == EscalationReason.NONE.value
