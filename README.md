# `@AppleSupport` AI Customer Support Agent

> **Brand**: `@AppleSupport` (from Kaggle `thoughtvector/customer-support-on-twitter`)   

---

## ⚡ Quickstart: Reproduce All Headline Results in < 2 Minutes

The entire pipeline runs 100% offline with zero external API key requirements.

```bash
# 1. Clone or navigate to the repository
cd Hiver_assign

# 2. Set up Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install frozen dependencies
pip install -r requirements.txt

# 4. Run full test suite (19 unit & integration tests)
PYTHONPATH=. pytest tests/ -v

# 5. Run headline benchmark across 200 Golden Examples (< 3 seconds)
PYTHONPATH=. python experiments/run_benchmark.py
```

To interactively chat with the agent in your terminal:
```bash
PYTHONPATH=. python src/agent.py --interactive
```

---

## 📊 Headline Benchmark Comparison Results

Evaluated across the **200-sample hand-labeled Golden Evaluation Set** (`data/golden_set.json`) spanning 8 balanced domain intents, 3 difficulty tiers, and realistic edge-case escalation scenarios.

| Model | Intent Acc | Intent Macro F1 | Triage Acc | Escalation F1 | ROUGE-L | Semantic Sim | LLM Judge Score | Mean Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1 (Trivial)** | 12.5% | 0.028 | 40.0% | 0.571 | 0.044 | 0.013 | 4.26 / 5.0 | 0.0 ms |
| **Baseline 2 (Simple)** | 47.5% | 0.450 | 64.0% | 0.217 | 0.042 | 0.024 | 3.97 / 5.0 | 0.4 ms |
| **Proposed AI Agent** | **71.0%** | **0.714** | **69.5%** | **0.684** | **0.079** | **0.048** | **4.59 / 5.0** | **1.1 ms** |

### 🎯 LLM-as-a-Judge Rubric Breakdown (1–5 Scale)

| Model | Grounding (1-5) | Tone (1-5) | Actionability (1-5) | Safety (1-5) | Overall Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1 (Trivial)** | 3.25 | 5.00 | 5.00 | 3.80 | **4.26 / 5.0** |
| **Baseline 2 (Simple)** | 3.95 | 4.00 | 4.35 | 3.58 | **3.97 / 5.0** |
| **Proposed AI Agent** | **4.42** | **4.68** | **4.99** | **4.25** | **4.59 / 5.0** |

### 🤝 Evidence of Human-Judge Agreement & Calibration

Evaluated across a **50-example multi-annotator subset** (`data/golden_set_human_labels.json`):
- **Inter-Human Annotator Agreement**: $\kappa = 1.000$ (Near-perfect consensus).
- **Judge vs. Human Consensus Agreement**: $\kappa = 0.4849$ (Moderate agreement on boundary cases).
- **Binary Triage Raw Agreement**: **74.0%** exact agreement on auto-handle vs. escalation decisions.
- **Rubric Pearson Correlation ($r$)**: **$r = 0.2988$ ($p = 0.0350$)**, confirming statistically significant correlation with human expert quality ratings.
- **Rating Mean Absolute Error (MAE)**: **$0.2925$ points** on a 5-point scale.

---

## 📁 Key Deliverables & Documentation Index

| Deliverable | File Link | Description |
| :--- | :--- | :--- |
| **1. Complete System Report** | [REPORT.md](REPORT.md) | Comprehensive 6-page report covering Problem Framing, Baseline Results, Top 5 Failure Modes with real examples, *"What is misleading about my headline number?"*, and Future Roadmap. |
| **2. Decision Log** | [DECISION_LOG.md](DECISION_LOG.md) | 15 non-obvious engineering and product decisions with explicit trade-offs. |
| **3. Technical Guide** | [GUIDE.md](GUIDE.md) | In-depth technical documentation covering system math, algorithms, and prompt architecture. |
| **4. Golden Evaluation Set** | [golden_set.json](data/golden_set.json) | 200 hand-labeled benchmark examples with intents, difficulty tiers, escalation reasons, and reference resolutions. |
| **5. Sampling & Labeling Note** | [sampling_and_labeling_note.md](data/sampling_and_labeling_note.md) | Detailed methodology note on dataset stratification, edge cases, and labeling guidelines. |
| **6. Human Calibration Data** | [golden_set_human_labels.json](data/golden_set_human_labels.json) | 50 dual-annotated human ratings across 4 quality dimensions for Judge validation. |
| **7. Benchmark Runner** | [run_benchmark.py](experiments/run_benchmark.py) | Runnable script that reproduces all tables and metric artifacts in under 5 seconds. |

---

## 🏛️ System Architecture

```
                                  INCOMING TWEET
                                        │
                         ┌──────────────┴──────────────┐
                         ▼                             ▼
              [Intent Classification]         [Entity Extraction]
              • Calibrated Softmax            • iPhone / iPad / Mac
              • Domain Keyword Boosts         • iOS Version / Errors
                         │                             │
                         └──────────────┬──────────────┘
                                        ▼
                      [Escalation & Safety Guardrails]
                      • Account Security / PII (Apple ID)
                      • Hardware Hazards (Swelling / Fire)
                      • Billing / In-App Refund Claims
                      • Legal Threats / High Agitation
                      • Confidence Margin Threshold (< 0.50)
                                        │
                         ┌──────────────┴──────────────┐
                         ▼                             ▼
               [Auto-Handle Path]             [Escalate to Human]
            • Historical Playbook RAG      • Stated Escalation Reason
            • Diagnostic Settings Step     • Secure DM Link Triage
                         │                             │
                         └──────────────┬──────────────┘
                                        ▼
                           [DRAFTED TWITTER REPLY]
                         • Under 280-char limit
                         • Apple Support Brand Voice
```

---

## 🧪 Testing & Validation

All 19 unit and integration tests pass in `< 2.0s`:

```bash
$ PYTHONPATH=. pytest tests/ -v
============================== 19 passed in 1.76s ==============================
```

To run individual test modules:
```bash
PYTHONPATH=. pytest tests/test_classifier.py
PYTHONPATH=. pytest tests/test_retriever.py
PYTHONPATH=. pytest tests/test_escalation.py
PYTHONPATH=. pytest tests/test_agent.py
```

---

## 📬 Submission Checklist

- [x] Runnable pipeline with reproduction in under 15 minutes (`README.md`).
- [x] Golden evaluation set with 200 hand-labeled examples (`data/golden_set.json`).
- [x] Sampling and labeling methodology note (`data/sampling_and_labeling_note.md`).
- [x] Automated metrics + LLM-as-judge rubric + Human-judge calibration evidence (`src/evaluation/`).
- [x] Final Report covering Problem Framing, 2 Baselines, Top 5 Failure Modes, *"What is misleading about my headline number?"*, and Next Steps (`REPORT.md`).
- [x] Decision Log with 15 non-obvious technical decisions (`DECISION_LOG.md`).
- [x] Ready to submit to `anurag@hiverhq.com`.
