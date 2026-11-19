# Stage 4 results: our router on LLMRouterBench

Our router on LLMRouterBench (Li et al., arXiv 2601.07206), the benchmark of the Avengers-Pro authors, performance-cost setting: 13 flagship models, 10 datasets, 12,446 questions, the paper's own 70/30 split and five seeds (42, 999, 2024, 2025, 3407). The 70% is the warm start; the 30% test part (3737 questions) is the stream with online learning (10 blocks). AvgAcc = mean over the datasets of the mean score (0-100), Cost = total USD on the test part, as in the paper; +- is the spread over seeds. Published numbers are the paper's Tables 12 and 13 (same test parts, mean of the same five seeds). The data has no latency, so latency and reputation (our extra dimensions) are off here.

## Findings

1. Highest accuracy ties the best published router: 68.73 +- 0.87 at $134.28 (+4.2% over GPT-5), against Avengers-Pro's best 68.58 at $132.98 (+4.0%).
2. From $28 to $134 we are level with or above Avengers-Pro at the same cost (+0.03 to +1.11 pts); the largest edge is w_C 0.02: 65.69 at $70.88, 99.6% of GPT-5's accuracy for 44% less cost.
3. Paper's CostSave (cheapest setting at least as accurate as GPT-5): ours w_C 0.01, 67.07 at $95.58 = 23.9% saved; Avengers-Pro 66.21 at $87.34 = 30.5% (31.7% in the paper's text). Our w_C 0.02 misses GPT-5 by 0.27 pts; no setting between 0.01 and 0.02 was run.
4. Below $12 Avengers-Pro is better by 0.54 to 1.27 pts.
5. Against the other published routers: RouteLLM 67.67 at $111.26 vs ours 68.48 at $113.82; GraphRouter 67.79 at $150.32 vs ours 68.73 at $134.28 (more accurate and cheaper); FrugalGPT, HybridLLM, OpenRouter and the random router are below our curve at every cost.
6. Online learning gives no gain here: -0.27 pts against the frozen router on average (better in 1 of 8 settings); the warm start is already large (8709 questions).
7. Room to improve: perfect picks reach 85.58 at $23.23. As cost falls, the hard sets lose most (HLE, tau2-bench), where only the expensive models succeed.

## 1. Our router (online learning)

| w_C | AvgAcc | Cost | vs GPT-5 | Cost vs GPT-5 | vs Avengers-Pro at the same cost |
|---|---|---|---|---|---|
| 0 | 68.73 +- 0.87 | $134.28 | +4.2% | +7% | +0.15 |
| 0.005 | 68.48 +- 1.36 | $113.82 | +3.8% | -9% | +0.03 |
| 0.01 | 67.07 +- 1.28 | $95.58 | +1.7% | -24% | +0.25 |
| 0.02 | 65.69 +- 1.20 | $70.88 | -0.4% | -44% | +1.11 |
| 0.05 | 61.34 +- 0.94 | $27.54 | -7.0% | -78% | +0.17 |
| 0.1 | 58.46 +- 1.67 | $10.92 | -11.4% | -91% | -1.17 |
| 0.2 | 56.64 +- 1.82 | $5.70 | -14.1% | -95% | -1.27 |
| 0.5 | 54.83 +- 1.39 | $3.74 | -16.9% | -97% | -0.54 |

Avengers-Pro at the same cost: straight line between its two published settings around that cost.

## 2. Our router without online learning

| w_C | AvgAcc | Cost | Learning minus frozen |
|---|---|---|---|
| 0 | 68.89 +- 0.79 | $135.09 | -0.16 |
| 0.005 | 68.35 +- 0.48 | $113.96 | +0.13 |
| 0.01 | 67.63 +- 0.65 | $95.39 | -0.56 |
| 0.02 | 65.78 +- 0.44 | $71.28 | -0.09 |
| 0.05 | 61.61 +- 1.08 | $27.39 | -0.27 |
| 0.1 | 58.81 +- 0.93 | $10.87 | -0.35 |
| 0.2 | 56.99 +- 1.69 | $5.55 | -0.34 |
| 0.5 | 55.34 +- 1.90 | $3.78 | -0.51 |

## 3. Published routers (same test parts)

| Router | AvgAcc | Cost |
|---|---|---|
| GPT-5 (best single model) | 65.96 | $125.60 |
| Avengers-Pro (alpha 1.00) | 68.58 | $132.98 |
| Avengers-Pro (alpha 0.80) | 67.88 | $104.41 |
| Avengers-Pro (alpha 0.70) | 66.79 | $95.21 |
| Avengers-Pro (alpha 0.65) | 66.21 | $87.34 |
| Avengers-Pro (alpha 0.60) | 65.27 | $78.17 |
| Avengers-Pro (alpha 0.50) | 63.71 | $62.75 |
| Avengers-Pro (alpha 0.40) | 62.22 | $45.05 |
| Avengers-Pro (alpha 0.35) | 60.38 | $14.44 |
| Avengers-Pro (alpha 0.30) | 59.70 | $11.13 |
| Avengers-Pro (alpha 0.25) | 58.46 | $7.21 |
| Avengers-Pro (alpha 0.10) | 57.29 | $4.40 |
| GraphRouter (performance first) | 67.79 | $150.32 |
| RouteLLM | 67.67 | $111.26 |
| FrugalGPT | 59.71 | $59.15 |
| HybridLLM | 57.56 | $18.65 |
| Random router | 53.45 | $52.17 |
| OpenRouter (commercial) | 49.67 | $82.31 |
| Oracle | 85.66 | $23.04 |

## 4. Single models and baselines on the same test parts (our run)

| | AvgAcc | Cost |
|---|---|---|
| claude-sonnet-4 | 47.27 | $70.57 |
| deepseek-r1-0528 | 57.86 | $59.60 |
| deepseek-v3-0324 | 44.41 | $5.00 |
| deepseek-v3.1-terminus | 53.11 | $9.05 |
| gemini-2.5-flash | 47.84 | $32.88 |
| gemini-2.5-pro | 60.80 | $229.89 |
| glm-4.6 | 55.61 | $62.90 |
| gpt-5 | 65.96 | $125.60 |
| gpt-5-chat | 47.22 | $18.04 |
| intern-s1 | 44.57 | $14.02 |
| kimi-k2-0905 | 54.59 | $14.37 |
| qwen3-235b-a22b-2507 | 55.74 | $6.93 |
| qwen3-235b-a22b-thinking-2507 | 57.65 | $36.43 |
| random | 52.97 | $50.97 |
| oracle | 85.58 | $23.23 |

Check: GPT-5 65.96 at $125.60, Qwen3-235B 55.74 at $6.93, Gemini-2.5-Pro 60.80 at $229.89, GLM-4.6 55.61 at $62.90 are identical to the paper; our oracle (best score, cheapest first) gives 85.58 at $23.23 against the paper's 85.66 at $23.04 (ties broken differently).

## 5. Our router per dataset (score 0-100)

| w_C | AIME | LiveMathBench | LiveCodeBench | SWE-bench | GPQA | HLE | MMLU-Pro | SimpleQA | ArenaHard | tau2-bench |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 87.8 | 82.2 | 83.7 | 34.0 | 85.0 | 24.5 | 86.7 | 57.7 | 76.3 | 69.5 |
| 0.005 | 88.9 | 82.2 | 84.7 | 32.5 | 82.3 | 23.8 | 86.1 | 58.0 | 76.2 | 70.0 |
| 0.01 | 86.7 | 79.5 | 84.3 | 30.1 | 82.7 | 23.5 | 85.8 | 58.8 | 74.8 | 64.5 |
| 0.02 | 86.7 | 80.0 | 81.8 | 27.1 | 77.0 | 21.8 | 85.0 | 57.5 | 75.5 | 64.5 |
| 0.05 | 78.9 | 78.4 | 75.1 | 24.0 | 74.3 | 12.9 | 84.4 | 57.6 | 73.5 | 54.3 |
| 0.1 | 75.6 | 78.4 | 70.5 | 22.5 | 72.3 | 8.8 | 84.0 | 55.7 | 73.2 | 43.6 |
| 0.2 | 75.6 | 77.3 | 65.7 | 23.9 | 70.3 | 9.1 | 83.5 | 55.0 | 71.5 | 34.5 |
| 0.5 | 75.6 | 79.5 | 64.8 | 21.5 | 69.0 | 8.4 | 84.1 | 54.6 | 71.6 | 19.3 |

## 6. Which models are picked (top four per setting)

| w_C | Picks |
|---|---|
| 0 | gpt-5 35%, gemini-2.5-pro 31%, qwen3-235b-a22b-2507 17%, deepseek-r1-0528 5% |
| 0.005 | gpt-5 35%, gemini-2.5-pro 24%, qwen3-235b-a22b-2507 19%, deepseek-r1-0528 6% |
| 0.01 | gpt-5 33%, qwen3-235b-a22b-2507 21%, gemini-2.5-pro 18%, deepseek-r1-0528 8% |
| 0.02 | gpt-5 27%, qwen3-235b-a22b-2507 25%, gemini-2.5-pro 13%, qwen3-235b-a22b-thinking-2507 8% |
| 0.05 | qwen3-235b-a22b-2507 37%, deepseek-v3.1-terminus 14%, qwen3-235b-a22b-thinking-2507 10%, gpt-5 8% |
| 0.1 | qwen3-235b-a22b-2507 47%, deepseek-v3.1-terminus 20%, qwen3-235b-a22b-thinking-2507 7%, kimi-k2-0905 7% |
| 0.2 | qwen3-235b-a22b-2507 55%, deepseek-v3.1-terminus 23%, kimi-k2-0905 6%, deepseek-v3-0324 5% |
| 0.5 | qwen3-235b-a22b-2507 59%, deepseek-v3.1-terminus 24%, deepseek-v3-0324 8%, kimi-k2-0905 5% |
