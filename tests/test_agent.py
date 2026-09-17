"""
Integration Tests for Unified AppleSupport AI Agent.
"""

import pytest
from src.agent import AppleSupportAgent, AgentOutput


@pytest.fixture
def agent():
    return AppleSupportAgent(knowledge_base_path="data/historical_knowledge_base.json")


def test_agent_end_to_end_battery_query(agent):
    msg = "My iPhone 8 battery drains from 100% to 15% in two hours after updating to iOS 11."
    out = agent.process_message(msg)

    assert isinstance(out, AgentOutput)
    assert out.predicted_intent == "battery_power"
    assert len(out.draft_reply) <= 280
    assert out.draft_reply.startswith("@")
    assert out.device_context["device_model"] == "iPhone 8"
    assert out.device_context["ios_version"] == "iOS 11"


def test_agent_end_to_end_hardware_escalation(agent):
    msg = "Dropped my iPhone and the screen is completely shattered and touch doesn't work."
    out = agent.process_message(msg)

    assert out.predicted_intent == "hardware_physical_damage"
    assert out.auto_handle is False
    assert out.escalation_reason == "physical_hardware_repair"
    assert "https://" in out.draft_reply or "DM" in out.draft_reply


def test_agent_character_limit_guarantee(agent):
    # Test adversarial long message
    long_msg = "My phone " * 50
    out = agent.process_message(long_msg)
    assert len(out.draft_reply) <= 280
