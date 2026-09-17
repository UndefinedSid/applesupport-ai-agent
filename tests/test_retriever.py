"""
Unit Tests for Historical Retriever.
"""

import pytest
from src.config import IntentCategory
from src.retriever import HistoricalRetriever


@pytest.fixture
def retriever():
    return HistoricalRetriever(knowledge_base_path="data/historical_knowledge_base.json")


def test_retrieve_playbook_by_intent(retriever):
    res = retriever.retrieve("My battery is dying fast", intent=IntentCategory.BATTERY_POWER.value)
    assert res["playbook"] is not None
    assert "Battery" in res["playbook"]["title"]
    assert len(res["playbook"]["verified_steps"]) > 0


def test_retrieve_dialogues(retriever):
    res = retriever.retrieve("iOS 11 update is slow and laggy", intent=IntentCategory.IOS_UPDATE_OS_BUGS.value, top_k=3)
    assert "nearest_historical_dialogues" in res
    assert len(res["nearest_historical_dialogues"]) <= 3


def test_retrieve_empty_query(retriever):
    res = retriever.retrieve("", intent=None)
    assert "nearest_historical_dialogues" in res
