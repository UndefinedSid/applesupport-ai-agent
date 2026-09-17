# Technical Guide: AppleSupport AI Support Agent System

Welcome to the comprehensive technical guide for the **AppleSupport AI Customer Support Agent**. This guide explains every architectural component, mathematical formulation, taxonomy definition, evaluation rubric, and developer workflow in the codebase.

---

## Table of Contents
1. [System Architecture Overview](#1-system-architecture-overview)
2. [Data Pipeline & Knowledge Base Construction](#2-data-pipeline--knowledge-base-construction)
3. [Intent Taxonomy & Classification Engine](#3-intent-taxonomy--classification-engine)
4. [Historical Resolution Retrieval Engine (RAG)](#4-historical-resolution-retrieval-engine-rag)
5. [Escalation & Safety Guardrail Engine](#5-escalation--safety-guardrail-engine)
6. [Grounded Response Drafting Engine](#6-grounded-response-drafting-engine)
7. [Evaluation Harness & Metric Mathematics](#7-evaluation-harness--metric-mathematics)
8. [LLM-as-a-Judge Rubric & Human Calibration Protocol](#8-llm-as-a-judge-rubric--human-calibration-protocol)
9. [Baseline Models](#9-baseline-models)
10. [Developer & Operations Quickstart](#10-developer--operations-quickstart)

---

## 1. System Architecture Overview

The system is organized into decoupled, production-ready modules adhering to clean architecture principles:

```
Hiver_assign/
├── README.md                          # Quickstart, headline results, reproducibility in <15 min
├── REPORT.md                          # Final comprehensive 6-page equivalent report
├── DECISION_LOG.md                    # 15 non-obvious engineering decisions and trade-offs
├── GUIDE.md                           # End-to-end technical guide (this document)
├── pyproject.toml                     # Modern Python build metadata
├── requirements.txt                   # Frozen dependency specifications
├── src/
│   ├── config.py                      # Taxonomies, brand voice, escalation thresholds
│   ├── data_processor.py              # Twitter text normalization, thread reconstruction, entity extraction
│   ├── intent_classifier.py           # Calibrated TF-IDF + keyword boost intent prediction engine
│   ├── retriever.py                   # Hybrid BM25/TF-IDF historical resolution retrieval engine
│   ├── escalation_engine.py           # Multi-factor risk, PII, hardware, and legal escalation engine
│   ├── response_generator.py          # Grounded reply drafter compliant with 280-char Twitter limit
│   ├── agent.py                       # Unified AppleSupportAgent orchestrator + Interactive CLI
│   ├── evaluation/
│   │   ├── metrics.py                 # Automated metrics (Accuracy, F1, BLEU, ROUGE-L, Cosine Sim)
│   │   ├── llm_judge.py               # 4-dimensional rubric scoring engine
│   │   ├── human_calibration.py       # Cohen's Kappa, Pearson r, and Spearman rho analysis
│   │   └── harness.py                 # Full evaluation harness over 200 golden examples
│   └── baselines/
│       ├── trivial_baseline.py        # Baseline 1: Majority Class + Canned Reply + Always Escalate
│       └── simple_baseline.py         # Baseline 2: Naive TF-IDF + Generic Template + Heuristic Escalation
├── data/
│   ├── golden_set.json                # 200 curated, hand-labeled golden benchmark examples
│   ├── golden_set_human_labels.json   # 50 dual-annotated examples for human-judge calibration
│   ├── historical_knowledge_base.json # 8 verified playbooks + 1000 indexed historical dialogues
│   └── sampling_and_labeling_note.md  # Detailed sampling methodology and annotation rules
├── experiments/
│   ├── run_benchmark.py               # Runs Agent vs Baselines, prints tables, saves artifacts
│   ├── benchmark_results.json         # Raw benchmark output data
│   └── benchmark_summary.md           # Markdown formatted benchmark tables
└── tests/
    ├── test_classifier.py             # 8 unit tests for intent classification
    ├── test_retriever.py              # 3 unit tests for historical resolution retrieval
    ├── test_escalation.py             # 5 unit tests for escalation guardrails
    └── test_agent.py                  # 3 integration tests for end-to-end pipeline
```

---

## 2. Data Pipeline & Knowledge Base Construction

### 2.1 Dataset Ingestion (`src/data_processor.py`)
- The raw dataset (`dataset/twcs/twcs.csv`) contains ~3M multi-turn customer support tweets.
- Outbound `@AppleSupport` tweets are paired with their parent customer inquiries by matching `in_response_to_tweet_id` to `tweet_id`.
- **Text Normalization**:
  - HTML unescaping (`&amp;` -> `&`, `&gt;` -> `>`).
  - Removal of invisible unicode variation selectors (`\ufe0f`, `\u200b`).
  - Anonymization of numeric customer IDs (`@115854` -> `@Customer`).
- **Entity Extraction**:
  - Automatically identifies Apple device models (`iPhone X`, `iPhone 8 Plus`, `iPad`, `Apple Watch`, `Mac`).
  - Automatically detects iOS versions (`iOS 11`, `iOS 10.3.3`).
  - Flags error codes (e.g. `Error 4013`, `Error 9`).

### 2.2 Knowledge Base (`data/historical_knowledge_base.json`)
- Combines **8 authoritative troubleshooting playbooks** derived from official Apple Support workflows.
- Indexes **1,000 high-quality historical dialogue pairs** extracted directly from real customer resolutions.

---

## 3. Intent Taxonomy & Classification Engine

### 3.1 Domain Intent Taxonomy (`src/config.py`)
1. **`battery_power`**: Rapid battery drain, charging failure, overheating, battery health degradation.
2. **`ios_update_os_bugs`**: Update stall/error codes, post-update system lag, keyboard autocorrect symbol bugs, UI freezing.
3. **`connectivity_network`**: Wi-Fi dropping, Bluetooth audio stutter/pairing, cellular 'No Service', AirDrop failure.
4. **`apple_id_icloud_security`**: Locked Apple ID, 2FA verification code loops, iCloud storage sync, forgotten password.
5. **`app_store_billing_subscriptions`**: Unauthorized charges, accidental minor in-app purchases, refund claims, subscription cancel.
6. **`hardware_physical_damage`**: Shattered screen, liquid ingress, jammed buttons, swollen battery, Genius Bar reservation.
7. **`general_faq_howto`**: Screenshots, data migration (Move to iOS), warranty lookup, feature settings.
8. **`out_of_scope_complaint`**: General rants, competitor praise, stock price questions, jokes, gibberish.

### 3.2 Classification Algorithm (`src/intent_classifier.py`)
1. **Semantic Representation**: TF-IDF n-gram vectorization ($n \in \{1, 2\}$) over intent descriptions, verified playbooks, and expanded exemplar keywords.
2. **Lexical Matching Boost**: Exact keyword matches receive weighted boosts ($\beta = 1.2$ for single terms, $\beta = 2.5$ for multi-word phrases).
3. **Calibrated Temperature-Scaled Softmax**:
   $$\text{Score}_k = 0.5 \cdot \text{Sim}_{\text{TF-IDF}}(q, C_k) + \min(0.25 \cdot \text{Matches}_k, 0.75)$$
   $$P(C_k \mid q) = \frac{\exp(\text{Score}_k \cdot \tau)}{\sum_{j=1}^K \exp(\text{Score}_j \cdot \tau)}, \quad \tau = 5.0$$
4. **Confidence Margin & OOD Fallback**:
   $$\text{Margin} = P(C_{\text{top1}} \mid q) - P(C_{\text{top2}} \mid q)$$
   If $P(C_{\text{top1}} \mid q) < 0.50$ or $\max(\text{Score}) < 0.15$, the message is flagged as ambiguous / low-confidence.

---

## 4. Historical Resolution Retrieval Engine (RAG)

The `HistoricalRetriever` (`src/retriever.py`) executes a two-stage hybrid retrieval strategy:
1. **Stage 1 (Authoritative Playbook Lookup)**: Directly fetches the expert-verified troubleshooting steps and resolution strategies associated with the predicted intent.
2. **Stage 2 (Nearest Historical Dialogue Retrieval)**: Computes cosine similarity between the incoming customer inquiry vector and the indexed 1,000 historical customer tweet vectors:
   $$\text{CosineSim}(q, d_i) = \frac{\mathbf{v}_q \cdot \mathbf{v}_{d_i}}{\|\mathbf{v}_q\| \|\mathbf{v}_{d_i}\|}$$
3. Outputs top-3 historical dialogues with similarity scores to provide grounding context for reply formulation.

---

## 5. Escalation & Safety Guardrail Engine

The `EscalationEngine` (`src/escalation_engine.py`) determines whether the agent can autonomously answer or must escalate to a human specialist with an explicit `EscalationReason`:

```
                       CUSTOMER MESSAGE + INTENT + CONFIDENCE
                                        │
             ┌──────────────────────────┴──────────────────────────┐
             ▼                                                     ▼
    [Deterministic Safety]                               [Policy & Statistical]
   • Swelling / Fire / Smoke                             • Account Security / PII (Apple ID)
   • Legal Threat / High Hostility                       • Financial / Refund Claims
   • Active Case # / Repeated Contact                    • Physical Hardware Inspection
             │                                           • Confidence Score < 0.50
             └──────────────────────────┬──────────────────────────┘
                                        ▼
                             [ESCALATION DECISION]
                        auto_handle: bool
                        escalation_reason: EscalationReason enum
                        explanation: str
```

### Stated Reasons Supported:
- `ACCOUNT_SECURITY_OR_PII`: Passwords, locked Apple IDs, 2FA codes.
- `BILLING_REFUND_AUTHORIZATION`: Refund requests, disputed charges.
- `PHYSICAL_HARDWARE_REPAIR`: Cracked screens, liquid damage, battery swelling.
- `HIGH_USER_AGITATION_OR_LEGAL`: Lawsuits, lawyers, extreme profanity.
- `REPEATED_UNRESOLVED_ISSUE`: Prior case IDs, multiple failed contacts.
- `LOW_CONFIDENCE_OR_AMBIGUOUS`: Model confidence $< 0.50$ or incoherent query.
- `NONE`: Safe for autonomous self-serve guidance.

---

## 6. Grounded Response Drafting Engine

The `ResponseGenerator` (`src/response_generator.py`) generates customer-facing replies satisfying all constraints:
- **Brand Voice**: Starts with `@Customer`, expresses standard Apple empathy, provides crisp instructions.
- **Diagnostic Clarification**: If device model or iOS version is missing for complex software issues, asks the user to check `Settings > General > About`.
- **Authoritative Link Insertion**: Directs to official Apple portals (`https://iforgot.apple.com`, `https://reportaproblem.apple.com`, `https://checkcoverage.apple.com`, `https://getsupport.apple.com`).
- **Private Channel Routing**: Automatically attaches the official Apple Support DM link placeholder (`https://t.co/GDrqU22YpT`) on all escalated tickets.
- **Twitter Length Budget**: Guarantees $\le 280$ characters with automated truncation protection.

---

## 7. Evaluation Harness & Metric Mathematics

### 7.1 Classification & Triage Metrics (`src/evaluation/metrics.py`)
- **Intent Accuracy**: Fraction of exactly matched intents across 8 classes.
- **Macro F1-Score**: Unweighted mean of F1 scores across all 8 classes:
  $$\text{Macro F1} = \frac{1}{K} \sum_{k=1}^K \frac{2 \cdot P_k \cdot R_k}{P_k + R_k}$$
- **Escalation Accuracy & F1**: Precision, recall, and harmonic mean for binary auto-handle vs. escalation decisions.

### 7.2 Generation Quality Metrics
- **ROUGE-L**: Longest Common Subsequence (LCS) F1 score between candidate reply and ground-truth reference resolution:
  $$R_{\text{LCS}} = \frac{\text{LCS}(\text{cand}, \text{ref})}{|\text{ref}|}, \quad P_{\text{LCS}} = \frac{\text{LCS}(\text{cand}, \text{ref})}{|\text{cand}|}, \quad \text{ROUGE-L} = \frac{2 P_{\text{LCS}} R_{\text{LCS}}}{P_{\text{LCS}} + R_{\text{LCS}}}$$
- **BLEU-1 & BLEU-2**: Modified n-gram precision with brevity penalty.
- **Semantic Cosine Similarity**: TF-IDF embedding cosine distance between candidate and target reference resolution.

---

## 8. LLM-as-a-Judge Rubric & Human Calibration Protocol

### 8.1 4-Dimensional Rubric (`src/evaluation/llm_judge.py`)
Each reply is scored on a **1 to 5 continuous scale**:
1. **Factual Grounding (1-5)**: Factual fidelity to Apple Support playbooks; penalties for wrong intent (-2) or hallucinated settings (-1).
2. **Brand Voice & Tone (1-5)**: Length compliance (<= 280 chars), empathetic greeting, customer handle inclusion.
3. **Actionability & Clarity (1-5)**: Presence of exact navigation steps (`>`), force restart instructions, or official support links.
4. **Safety & Escalation Soundness (1-5)**: Score = 1 (Critical Failure) if an account lockout or dangerous hardware hazard is auto-handled. Score = 5 for sound triage.

### 8.2 Inter-Rater Calibration Math (`src/evaluation/human_calibration.py`)
- **Cohen's Kappa ($\kappa$)**:
  $$\kappa = \frac{P_o - P_e}{1 - P_e}$$
  Where $P_o$ is observed agreement and $P_e$ is expected chance agreement.
- **Pearson Product-Moment Correlation ($r$)**:
  $$r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}$$
- **Spearman Rank Correlation ($\rho$)**: Monotonic ranking alignment between human consensus and judge score.

---

## 9. Baseline Models

1. **Baseline 1 (Trivial Baseline - `src/baselines/trivial_baseline.py`)**:
   - Always predicts majority intent (`ios_update_os_bugs`).
   - Always sends static canned reply: *"@Customer Thanks for reaching out! Please DM us your device details: https://t.co/GDrqU22YpT"*.
   - Always escalates to human (`auto_handle = False`).
2. **Baseline 2 (Simple Baseline - `src/baselines/simple_baseline.py`)**:
   - Uncalibrated raw TF-IDF over intent descriptions.
   - Ungrounded static template reply per intent.
   - Naive keyword escalation (*"angry", "broken", "money", "lawyer"*).

---

## 10. Developer & Operations Quickstart

### Environment Setup
```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

### Running Unit & Integration Tests (< 2 seconds)
```bash
PYTHONPATH=. pytest tests/ -v
```

### Running Full Benchmark & Generating Tables (< 3 seconds)
```bash
PYTHONPATH=. python experiments/run_benchmark.py
```

### Running Interactive Terminal CLI
```bash
PYTHONPATH=. python src/agent.py --interactive
```

---

## Summary
This architecture provides a verifiable, deterministic, and highly accurate AI Support Agent for `@AppleSupport`, combining the best of semantic search, deterministic safety guardrails, and rigorous evaluation methodologies.
