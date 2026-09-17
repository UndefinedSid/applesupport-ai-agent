"""
Unit Tests for Intent Classifier.
"""

import pytest
from src.config import IntentCategory
from src.intent_classifier import IntentClassifier


@pytest.fixture
def classifier():
    return IntentClassifier()


def test_classify_battery_issue(classifier):
    text = "My iPhone battery is draining from 100% to 20% in 2 hours."
    res = classifier.predict(text)
    assert res["intent"] == IntentCategory.BATTERY_POWER.value
    assert res["confidence"] > 0.40


def test_classify_ios_update_glitch(classifier):
    text = "When I type the letter 'I', iOS replaces it with a weird symbol A[?]."
    res = classifier.predict(text)
    assert res["intent"] == IntentCategory.IOS_UPDATE_OS_BUGS.value


def test_classify_wifi_bluetooth_issue(classifier):
    text = "AirPods will not connect to my phone via Bluetooth."
    res = classifier.predict(text)
    assert res["intent"] == IntentCategory.CONNECTIVITY_NETWORK.value


def test_classify_apple_id_lockout(classifier):
    text = "My Apple ID is locked for security reasons and 2FA code is not coming."
    res = classifier.predict(text)
    assert res["intent"] == IntentCategory.APPLE_ID_ICLOUD_SECURITY.value


def test_classify_billing_refund(classifier):
    text = "I was charged twice on my credit card for Apple Music subscription."
    res = classifier.predict(text)
    assert res["intent"] == IntentCategory.APP_STORE_BILLING_SUBSCRIPTIONS.value


def test_classify_screen_damage(classifier):
    text = "Dropped my phone and the screen is completely cracked and shattered."
    res = classifier.predict(text)
    assert res["intent"] == IntentCategory.HARDWARE_PHYSICAL_DAMAGE.value


def test_classify_how_to_faq(classifier):
    text = "How do I take a screenshot on iPhone X?"
    res = classifier.predict(text)
    assert res["intent"] == IntentCategory.GENERAL_FAQ_HOWTO.value


def test_classify_out_of_scope(classifier):
    text = "asdfghjkl random gibberish !!!"
    res = classifier.predict(text)
    assert res["intent"] == IntentCategory.OUT_OF_SCOPE_COMPLAINT.value or not res["is_confident"]
