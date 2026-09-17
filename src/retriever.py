"""
Historical Resolution Retrieval Engine for AppleSupport AI Agent.
Performs hybrid lexical and semantic retrieval over historical customer-support dialogues
and verified troubleshooting playbooks to ground agent replies.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data_processor import DataProcessor
from src.config import IntentCategory

logger = logging.getLogger(__name__)


class HistoricalRetriever:
    """Retrieves verified historical resolutions and matching customer support threads."""

    def __init__(self, knowledge_base_path: str = "data/historical_knowledge_base.json"):
        self.kb_path = Path(knowledge_base_path)
        self.playbooks = {}
        self.historical_dialogues = []
        self.dialogue_vectorizer = None
        self.dialogue_tfidf_matrix = None
        self._load_knowledge_base()

    def _load_knowledge_base(self):
        """Loads and indexes the knowledge base from JSON."""
        if not self.kb_path.exists():
            logger.warning(f"Knowledge base not found at {self.kb_path}. Running with empty KB.")
            return

        with open(self.kb_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Index playbooks by intent
        for pb in data.get("verified_playbooks", []):
            self.playbooks[pb["intent"]] = pb

        # Index historical dialogues
        self.historical_dialogues = data.get("indexed_historical_dialogues", [])
        
        if self.historical_dialogues:
            customer_queries = [
                d.get("customer_text", "") for d in self.historical_dialogues
            ]
            self.dialogue_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
            self.dialogue_tfidf_matrix = self.dialogue_vectorizer.fit_transform(customer_queries)
            logger.info(f"Indexed {len(self.historical_dialogues)} historical dialogues for retrieval.")

    def retrieve(self, query: str, intent: Optional[str] = None, top_k: int = 3) -> Dict[str, Any]:
        """
        Retrieves matching playbook and top-k historical resolution examples for an incoming query.
        """
        cleaned_query = DataProcessor.clean_tweet_text(query)
        
        # 1. Retrieve authoritative playbook for the predicted intent
        matched_playbook = None
        if intent and intent in self.playbooks:
            matched_playbook = self.playbooks[intent]
        else:
            # Search playbooks by keyword overlap
            best_pb = None
            best_overlap = 0
            query_lower = cleaned_query.lower()
            for pb in self.playbooks.values():
                overlap = sum(1 for kw in pb.get("trigger_keywords", []) if kw in query_lower)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_pb = pb
            matched_playbook = best_pb

        # 2. Retrieve top-k nearest historical dialogues
        top_dialogues = []
        if self.dialogue_vectorizer and self.dialogue_tfidf_matrix is not None:
            query_vec = self.dialogue_vectorizer.transform([cleaned_query])
            sims = cosine_similarity(query_vec, self.dialogue_tfidf_matrix)[0]
            top_indices = sims.argsort()[::-1][:top_k]

            for idx in top_indices:
                score = float(sims[idx])
                if score > 0.05:  # Relevance cutoff
                    d = self.historical_dialogues[idx]
                    top_dialogues.append({
                        "similarity_score": round(score, 4),
                        "customer_text": d.get("customer_text", ""),
                        "historical_agent_reply": d.get("agent_text", ""),
                        "device_context": d.get("context", {})
                    })

        return {
            "query": cleaned_query,
            "intent": intent,
            "playbook": matched_playbook,
            "nearest_historical_dialogues": top_dialogues,
            "has_high_relevance_match": len(top_dialogues) > 0 and top_dialogues[0]["similarity_score"] > 0.30
        }
