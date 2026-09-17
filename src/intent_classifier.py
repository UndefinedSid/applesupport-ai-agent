"""
Intent Classification Engine for AppleSupport AI Agent.
Combines TF-IDF semantic matching, lexical keyword scoring, and exemplar similarity
to produce calibrated intent predictions with confidence metrics.
"""

import re
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import IntentCategory, INTENT_REGISTRY, CONFIDENCE_THRESHOLD_AUTO_HANDLE
from src.data_processor import DataProcessor


class IntentClassifier:
    """Classifies customer tweets into defined AppleSupport intent categories."""

    def __init__(self):
        self.categories = list(INTENT_REGISTRY.keys())
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, stop_words="english")
        
        # Build training corpus from sample phrases and descriptions
        self.corpus = []
        self.corpus_labels = []
        
        for cat, meta in INTENT_REGISTRY.items():
            # Add description
            self.corpus.append(meta.description)
            self.corpus_labels.append(cat)
            
            # Add sample phrases
            for phrase in meta.sample_phrases:
                self.corpus.append(phrase)
                self.corpus_labels.append(cat)
                
            # Add expanded keyword strings
            self.corpus.append(" ".join(meta.keywords * 4))
            self.corpus_labels.append(cat)

        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)

    def _compute_keyword_boost(self, text: str) -> Dict[IntentCategory, float]:
        """Calculates exact keyword and indicator match boosts."""
        text_lower = text.lower()
        boosts = {cat: 0.0 for cat in self.categories}
        
        for cat, meta in INTENT_REGISTRY.items():
            matches = 0.0
            for kw in meta.keywords:
                # Multi-word phrase matching gets higher weight
                if " " in kw:
                    if kw in text_lower:
                        matches += 2.5
                else:
                    if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
                        matches += 1.2
            boosts[cat] = matches
        return boosts

    def predict(self, text: str) -> Dict[str, any]:
        """
        Classifies incoming text into an IntentCategory with confidence score and probabilities.
        """
        cleaned = DataProcessor.clean_tweet_text(text)
        if not cleaned or len(cleaned.strip()) == 0:
            return {
                "intent": IntentCategory.OUT_OF_SCOPE_COMPLAINT.value,
                "confidence": 0.0,
                "is_confident": False,
                "intent_probabilities": {cat.value: 0.0 for cat in self.categories},
                "matched_keywords": []
            }

        query_vec = self.vectorizer.transform([cleaned])
        sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        # Aggregate similarity scores by category
        cat_scores = {cat: 0.0 for cat in self.categories}
        
        for score, label in zip(sims, self.corpus_labels):
            cat_scores[label] = max(cat_scores[label], score)

        # Apply keyword boosts
        kw_boosts = self._compute_keyword_boost(cleaned)
        
        combined_scores = {}
        for cat in self.categories:
            tfidf_score = cat_scores[cat]
            kw_score = min(kw_boosts[cat] * 0.25, 0.75)
            combined_scores[cat] = tfidf_score * 0.5 + kw_score

        # Specific high-priority pattern heuristics
        text_lower = cleaned.lower()
        
        # Apple ID / 2FA / Password priority
        if any(w in text_lower for w in ["apple id", "icloud lock", "iforgot", "passcode", "password", "2fa", "verification code", "two-factor"]):
            combined_scores[IntentCategory.APPLE_ID_ICLOUD_SECURITY] += 0.60
            
        # Billing & Refund priority
        if any(w in text_lower for w in ["charged", "refund", "subscription", "bill", "invoice", "unauthorized purchase", "cancel subscription", "billed twice"]):
            combined_scores[IntentCategory.APP_STORE_BILLING_SUBSCRIPTIONS] += 0.60
            
        # Hardware / Physical Damage priority
        if any(w in text_lower for w in ["cracked", "shattered", "water", "genius bar", "swollen", "broken screen", "dropped my phone", "screen replacement"]):
            combined_scores[IntentCategory.HARDWARE_PHYSICAL_DAMAGE] += 0.60

        # Battery / Power priority
        if any(w in text_lower for w in ["battery", "drain", "draining", "charge past", "won't charge", "overheating", "overheat", "shut down"]):
            combined_scores[IntentCategory.BATTERY_POWER] += 0.50

        # iOS update priority
        if any(w in text_lower for w in ["update", "updated", "updating", "lag", "glitch", "autocorrect", "ios 11", "ios 10", "spinning wheel"]):
            combined_scores[IntentCategory.IOS_UPDATE_OS_BUGS] += 0.50

        # Connectivity priority
        if any(w in text_lower for w in ["wifi", "wi-fi", "bluetooth", "airpods", "airdrop", "no service", "hotspot"]):
            combined_scores[IntentCategory.CONNECTIVITY_NETWORK] += 0.50

        # FAQ / How-To priority
        if any(w in text_lower for w in ["how to", "how do i", "screenshot", "transfer data", "move to ios", "where can i"]):
            combined_scores[IntentCategory.GENERAL_FAQ_HOWTO] += 0.50

        # Sharp softmax with temperature scaling
        scores_array = np.array([combined_scores[cat] for cat in self.categories])
        
        # Max-min normalization + Temperature
        max_score = np.max(scores_array)
        if max_score > 0:
            exp_scores = np.exp(scores_array * 5.0)
            probabilities = exp_scores / np.sum(exp_scores)
        else:
            probabilities = np.ones(len(self.categories)) / len(self.categories)

        sorted_indices = np.argsort(probabilities)[::-1]
        best_idx = sorted_indices[0]
        runner_up_idx = sorted_indices[1]

        best_category = self.categories[best_idx]
        best_confidence = float(probabilities[best_idx])
        runner_up_confidence = float(probabilities[runner_up_idx])
        confidence_margin = best_confidence - runner_up_confidence

        matched_kws = [
            kw for kw in INTENT_REGISTRY[best_category].keywords 
            if kw in text_lower
        ]

        # Check if score is negligible and no keywords matched -> OUT_OF_SCOPE
        if max_score < 0.15 and len(matched_kws) == 0:
            best_category = IntentCategory.OUT_OF_SCOPE_COMPLAINT
            best_confidence = 0.40
            confidence_margin = 0.10

        return {
            "intent": best_category.value,
            "confidence": round(best_confidence, 4),
            "confidence_margin": round(confidence_margin, 4),
            "is_confident": best_confidence >= CONFIDENCE_THRESHOLD_AUTO_HANDLE,
            "intent_probabilities": {
                cat.value: round(float(prob), 4) 
                for cat, prob in zip(self.categories, probabilities)
            },
            "matched_keywords": matched_kws
        }
