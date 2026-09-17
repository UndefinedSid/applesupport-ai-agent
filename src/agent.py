"""
Unified AppleSupport AI Customer Support Agent Orchestrator.
Integrates intent classification, historical resolution retrieval, grounded reply generation,
and confidence-calibrated escalation decision making.
"""

import os
import sys
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import BRAND_NAME
from src.data_processor import DataProcessor
from src.intent_classifier import IntentClassifier
from src.retriever import HistoricalRetriever
from src.escalation_engine import EscalationEngine
from src.response_generator import ResponseGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class AgentOutput:
    """Unified response object returned by the AppleSupport AI Agent."""
    customer_query: str
    cleaned_query: str
    predicted_intent: str
    intent_confidence: float
    is_confident: bool
    draft_reply: str
    auto_handle: bool
    escalation_reason: str
    escalation_explanation: str
    device_context: Dict[str, Any]
    retrieved_playbook_title: Optional[str]
    retrieval_similarity: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AppleSupportAgent:
    """End-to-end production AI Support Agent for @AppleSupport."""

    def __init__(self, knowledge_base_path: str = "data/historical_knowledge_base.json"):
        logger.info("Initializing AppleSupport AI Agent components...")
        self.processor = DataProcessor()
        self.classifier = IntentClassifier()
        self.retriever = HistoricalRetriever(knowledge_base_path=knowledge_base_path)
        self.escalator = EscalationEngine()
        self.generator = ResponseGenerator()
        logger.info("AppleSupport AI Agent successfully initialized.")

    def process_message(self, customer_message: str) -> AgentOutput:
        """
        Executes the full agent pipeline for an incoming customer message.
        """
        cleaned_text = self.processor.clean_tweet_text(customer_message)
        device_ctx = self.processor.extract_device_context(cleaned_text)

        # 1. Intent Classification
        clf_result = self.classifier.predict(cleaned_text)
        pred_intent = clf_result["intent"]
        confidence = clf_result["confidence"]

        # 2. Historical Resolution Retrieval
        retrieval_res = self.retriever.retrieve(cleaned_text, intent=pred_intent, top_k=3)
        playbook = retrieval_res.get("playbook")
        top_dialogues = retrieval_res.get("nearest_historical_dialogues", [])
        top_sim = top_dialogues[0]["similarity_score"] if top_dialogues else 0.0
        pb_title = playbook["title"] if playbook else None

        # 3. Escalation Decision Triage
        esc_decision = self.escalator.evaluate(
            text=cleaned_text,
            predicted_intent=pred_intent,
            intent_confidence=confidence,
            matched_keywords=clf_result.get("matched_keywords", [])
        )

        # 4. Grounded Response Drafting
        draft_reply = self.generator.generate_reply(
            customer_text=cleaned_text,
            predicted_intent=pred_intent,
            retrieval_context=retrieval_res,
            escalation_decision=esc_decision,
            device_context=device_ctx
        )

        return AgentOutput(
            customer_query=customer_message,
            cleaned_query=cleaned_text,
            predicted_intent=pred_intent,
            intent_confidence=confidence,
            is_confident=clf_result["is_confident"],
            draft_reply=draft_reply,
            auto_handle=esc_decision["auto_handle"],
            escalation_reason=esc_decision["escalation_reason"],
            escalation_explanation=esc_decision["explanation"],
            device_context=device_ctx,
            retrieved_playbook_title=pb_title,
            retrieval_similarity=top_sim
        )


def interactive_cli():
    """Runs an interactive terminal session for testing the agent."""
    agent = AppleSupportAgent()
    print("\n=======================================================")
    print("  AppleSupport AI Agent Interactive Terminal")
    print("  Type 'exit' or 'quit' to stop.")
    print("=======================================================\n")

    while True:
        try:
            user_input = input("\n[Customer Tweet] > ").strip()
            if not user_input or user_input.lower() in ["exit", "quit"]:
                break

            output = agent.process_message(user_input)
            print("\n---------------- AGENT DIAGNOSTICS ----------------")
            print(f"• Intent:          {output.predicted_intent} (Confidence: {output.intent_confidence:.2f})")
            print(f"• Triage Action:   {'AUTO-HANDLE' if output.auto_handle else 'ESCALATE TO HUMAN'}")
            print(f"• Stated Reason:   {output.escalation_reason}")
            print(f"• Explanation:     {output.escalation_explanation}")
            print(f"• Device Context:  {output.device_context}")
            print(f"• Grounding Source:{output.retrieved_playbook_title} (Sim: {output.retrieval_similarity:.2f})")
            print("---------------- DRAFTED TWITTER REPLY -------------")
            print(f"{output.draft_reply}")
            print(f"Length: {len(output.draft_reply)}/280 chars")
            print("----------------------------------------------------\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error processing message: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_cli()
    else:
        # Quick demonstration test
        agent = AppleSupportAgent()
        sample = "My iPhone 8 battery drains from 100% to 10% in two hours after updating to iOS 11. Help!"
        res = agent.process_message(sample)
        print(json.dumps(res.to_dict(), indent=2))
