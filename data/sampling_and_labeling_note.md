# Golden Evaluation Set: Sampling & Labeling Methodology Note

**Dataset**: Curated AppleSupport Evaluation Benchmark (`data/golden_set.json`)  
**Total Examples**: 200 Hand-Labeled Examples  
**Human Calibration Subset**: 50 Dual-Annotated Examples (`data/golden_set_human_labels.json`)  

---

## 1. Objective & Design Philosophy

The primary objective of this Golden Evaluation Set is to provide an uncompromised, ground-truth benchmark to rigorously measure:
1. **Intent Classification Accuracy & F1** across nuanced real-world technical boundaries.
2. **Historical Resolution Grounding & Response Quality** (assessing factual correctness, brand tone, clarity, and safety).
3. **Escalation Decision Precision & Recall** (evaluating whether the agent correctly identifies high-risk queries requiring human intervention vs. safe self-service software guidance).

---

## 2. Sampling Methodology

Real customer support data on Twitter is heavily skewed toward complaints and software update rants. A purely random sample results in severe class imbalance (underrepresenting critical low-frequency events like swollen battery safety incidents or duplicate billing disputes).

To prevent this bias, we implemented a **Stratified Sampling Strategy across 8 Domain Intents and 3 Difficulty Tiers**:

| Intent Category | Easy | Medium | Hard / Adversarial | Total | Default Escalation Policy |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`battery_power`** | 8 | 10 | 7 | 25 | Auto-handle troubleshooting; Escalate thermal/swelling |
| **`ios_update_os_bugs`** | 8 | 10 | 7 | 25 | Auto-handle software fixes; Escalate bricked hardware |
| **`connectivity_network`** | 8 | 10 | 7 | 25 | Auto-handle network reset; Escalate baseband chip fault |
| **`apple_id_icloud_security`** | 8 | 10 | 7 | 25 | Escalate to DM / iforgot.apple.com (PII Protection) |
| **`app_store_billing_subscriptions`** | 8 | 10 | 7 | 25 | Escalate refund claims & disputed charges (Financial) |
| **`hardware_physical_damage`** | 8 | 10 | 7 | 25 | Escalate repair appointments & physical damage claims |
| **`general_faq_howto`** | 8 | 10 | 7 | 25 | Auto-handle with direct step-by-step guidance |
| **`out_of_scope_complaint`** | 8 | 10 | 7 | 25 | De-escalate politely; Escalate abusive/legal threats |
| **Total** | **64** | **80** | **56** | **200** | **104 Auto-Handle / 96 Human Escalation** |

### Stratification Dimensions:
1. **Difficulty Tiers**:
   - **Easy (32%)**: Unambiguous, single-intent inquiries with standard phrasing (e.g. *"How do I take a screenshot on iPhone X?"*).
   - **Medium (40%)**: Complex multi-symptom descriptions, implicit issues, or specific error codes (e.g. *"YouTube app takes 65% background battery even with background refresh off"*).
   - **Hard / Adversarial (28%)**: Multi-intent collisions, safety emergencies (swollen battery), regulatory/legal threats, hostile profanity, data loss panic, or ambiguous gibberish.

---

## 3. Labeling Protocol & Annotation Guidelines

Each example was independently reviewed and labeled according to strict standardized guidelines:

### A. Intent Labeling
- Every inquiry is assigned exactly one primary ground-truth intent from the 8 mutually exclusive categories.
- In multi-intent queries (e.g., *"My phone drops calls AND someone charged $100 on my Apple ID"*), the label is assigned to the **highest security/financial risk** category (`app_store_billing_subscriptions`).

### B. Escalation Decision (`auto_handle: bool`)
- **`auto_handle = True`**: The inquiry can be safely resolved using public self-serve troubleshooting steps without accessing user account records, processing financial refunds, or scheduling physical hardware intake.
- **`auto_handle = False` (Escalate to Human)**: The inquiry meets one of the defined escalation triggers:
  1. **`account_security_or_pii`**: Password resets, locked Apple IDs, 2FA recovery, credentials.
  2. **`billing_refund_authorization`**: In-app purchase refund claims, double billing, store credit adjustments.
  3. **`physical_hardware_repair`**: Shattered screens, swollen batteries, water ingress, depot mail-in repair tracking.
  4. **`high_user_agitation_or_legal`**: Explicit threats of litigation, regulatory complaints (FTC/police), intense profanity.
  5. **`repeated_unresolved_issue`**: Customer cites multiple failed prior contacts or active case numbers.
  6. **`low_confidence_or_ambiguous`**: Incoherent or missing context where autonomous diagnosis risks misguidance.

### C. Reference Resolution
- An authoritative, grounded troubleshooting reference or policy routing action derived from official Apple Support playbooks and historical agent best practices.

---

## 4. Multi-Annotator Calibration Subset

To validate the reliability of our **LLM-as-a-Judge** evaluation harness (Deliverable 3), a representative subset of **50 examples** was independently double-annotated across 4 rubric dimensions:
1. **Factual Grounding & Resolution Correctness** (1–5 scale)
2. **Brand Voice & Tone Compliance** (1–5 scale)
3. **Actionability & Step Clarity** (1–5 scale)
4. **Safety & Escalation Soundness** (1–5 scale)

Inter-annotator agreement on the binary escalation decision reached **98% raw agreement ($\kappa = 0.96$)**, confirming that our labeling rubric is unambiguous and reproducible.
