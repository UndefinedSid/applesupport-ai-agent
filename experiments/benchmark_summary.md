# Benchmark Evaluation Summary

### 1. Headline Results

| Model                       | Intent Acc   |   Intent F1 | Escalation Acc   |   Escalation F1 |   ROUGE-L |   Semantic Sim | Judge Score   | Mean Latency   |
|-----------------------------|--------------|-------------|------------------|-----------------|-----------|----------------|---------------|----------------|
| Baseline 1 (Trivial)        | 12.5%        |       0.028 | 40.0%            |           0.571 |     0.044 |          0.013 | 4.26 / 5.0    | 0.0 ms         |
| Baseline 2 (Simple)         | 47.5%        |       0.45  | 64.0%            |           0.217 |     0.042 |          0.024 | 3.97 / 5.0    | 0.4 ms         |
| Proposed AppleSupport Agent | 71.0%        |       0.714 | 69.5%            |           0.684 |     0.079 |          0.048 | 4.59 / 5.0    | 1.1 ms         |

### 2. LLM-as-a-Judge Rubric Breakdown

| Model                       |   Grounding (1-5) |   Tone (1-5) |   Actionability (1-5) |   Safety (1-5) |   Overall (1-5) |
|-----------------------------|-------------------|--------------|-----------------------|----------------|-----------------|
| Baseline 1 (Trivial)        |              3.25 |         5    |                  5    |           3.8  |            4.26 |
| Baseline 2 (Simple)         |              3.95 |         4    |                  4.35 |           3.58 |            3.97 |
| Proposed AppleSupport Agent |              4.42 |         4.68 |                  4.99 |           4.25 |            4.59 |

### 3. Difficulty Tier Breakdown

| Model                       | Easy (Intent/Esc Acc)   | Medium (Intent/Esc Acc)   | Hard (Intent/Esc Acc)   |
|-----------------------------|-------------------------|---------------------------|-------------------------|
| Baseline 1 (Trivial)        | 12.5% / 29.7%           | 12.5% / 37.5%             | 12.5% / 55.4%           |
| Baseline 2 (Simple)         | 54.7% / 71.9%           | 47.5% / 63.7%             | 39.3% / 55.4%           |
| Proposed AppleSupport Agent | 79.7% / 78.1%           | 63.7% / 68.8%             | 71.4% / 60.7%           |

### 4. Human-Judge Agreement & Calibration

| Metric                                           | Value          | Interpretation                         |
|--------------------------------------------------|----------------|----------------------------------------|
| Human 1 vs Human 2 (Inter-Annotator Agreement)   | Kappa = 1.0000 | Near-perfect consensus                 |
| Judge vs Human Consensus (Binary Decision Kappa) | Kappa = 0.4849 | Moderate agreement                     |
| Binary Triage Raw Agreement %                    | 74.0%          | Strong alignment on decisions          |
| Overall Rubric Pearson Correlation (r)           | r = 0.2988     | p = 0.0350 (Statistically Significant) |
| Overall Rubric Spearman Rank Correlation (rho)   | rho = 0.2350   | Monotonic score ranking alignment      |
| Rating Mean Absolute Error (MAE)                 | 0.2925 pts     | Low rating divergence on 1-5 scale     |
