# Stage 3 results: five live models together

Five live models chosen together (Claude Haiku/Sonnet/Opus 5.5, GPT-6.1 Terra/Sol) on 250 SuperGPQA questions: 75 warm start + 175 stream, online learning (test-then-train, blocks of 25), 5 seeds. Cost is USD per 1,000 questions, latency in seconds; +- is the spread over seeds. Reputation off (w_R = 0) unless stated.

## Findings

1. Savings against always using Opus: 95% of its quality for 42% less cost and 13% less latency; 90% for 88% less cost and 35% less latency; 85% for 93% / 41%; 80% for 97% / 37%. These are the best setting per quality level, chosen after the runs.
2. Highest router accuracy 75.0% at $7.09 and 5.16 s (w_C 0, w_L 0.02), against Opus alone at 77.7%, $11.74 and 5.99 s.
3. The weights steer as intended: w_C moves picks from Opus to Haiku and Terra, w_L to the fast models, w_R toward Claude (w_R 0.1: Claude share 46% -> 70% for -0.8 pts accuracy and +28% cost).
4. Against simply mixing single models at the same cost the router is -1.2 pts on average (better in 11 of 81 settings, best +1.0, worst -3.9).
5. Room to improve: perfect picks reach 83.8% at $1.01.
6. Online learning is a tie at this size: +0.07 pts against the frozen router (better in 35, worse in 40 of 81 settings). It stays in the design for the ledger.

## 1. Baselines

| | Accuracy | Cost | Latency |
|---|---|---|---|
| Claude Haiku 5.5 | 62.3% +- 2.8% | $0.37 | 3.76 s |
| Claude Sonnet 5.5 | 69.6% +- 1.5% | $4.78 | 3.98 s |
| Claude Opus 5.5 | 77.7% +- 2.3% | $11.74 | 5.99 s |
| GPT-6.1 Terra | 69.0% +- 1.9% | $1.00 | 3.36 s |
| GPT-6.1 Sol | 73.3% +- 1.5% | $2.39 | 5.29 s |
| Random pick | 69.6% +- 4.1% | $3.87 | 4.46 s |
| Perfect picks (oracle) | 83.8% +- 1.7% | $1.01 | 3.87 s |

## 2. Router, every setting (rows w_C, columns w_L)

Accuracy (%)

| w_C \ w_L | 0 | 0.01 | 0.02 | 0.05 | 0.1 | 0.2 | 0.5 | 1 | 2 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 73.9 | 73.8 | 75.0 | 73.0 | 71.3 | 70.7 | 69.1 | 69.4 | 69.8 |
| 0.01 | 73.4 | 72.5 | 71.5 | 72.9 | 70.7 | 69.8 | 69.0 | 68.8 | 69.4 |
| 0.02 | 72.1 | 71.3 | 72.3 | 72.7 | 71.1 | 69.4 | 69.4 | 68.6 | 69.4 |
| 0.05 | 69.8 | 70.4 | 69.8 | 70.4 | 70.2 | 68.8 | 68.0 | 68.3 | 69.1 |
| 0.1 | 70.6 | 71.0 | 70.3 | 69.6 | 69.9 | 69.5 | 68.2 | 68.0 | 69.3 |
| 0.2 | 68.7 | 69.0 | 69.0 | 68.3 | 67.1 | 68.0 | 67.4 | 68.5 | 69.1 |
| 0.5 | 65.7 | 65.3 | 65.3 | 64.9 | 64.9 | 65.3 | 64.7 | 66.1 | 66.9 |
| 1 | 63.2 | 63.1 | 63.8 | 63.5 | 63.8 | 64.2 | 64.5 | 65.1 | 66.3 |
| 2 | 62.4 | 62.4 | 62.5 | 62.9 | 62.9 | 62.7 | 63.0 | 63.1 | 65.6 |

Cost ($ per 1,000 questions)

| w_C \ w_L | 0 | 0.01 | 0.02 | 0.05 | 0.1 | 0.2 | 0.5 | 1 | 2 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 7.08 | 6.78 | 7.09 | 6.47 | 5.31 | 3.83 | 2.10 | 1.67 | 1.51 |
| 0.01 | 6.26 | 5.62 | 5.84 | 5.22 | 4.08 | 2.95 | 1.93 | 1.65 | 1.46 |
| 0.02 | 4.89 | 4.72 | 4.92 | 3.80 | 3.28 | 2.43 | 1.70 | 1.55 | 1.45 |
| 0.05 | 2.49 | 2.43 | 2.37 | 2.13 | 1.96 | 1.90 | 1.52 | 1.44 | 1.37 |
| 0.1 | 1.64 | 1.68 | 1.62 | 1.46 | 1.45 | 1.41 | 1.34 | 1.23 | 1.24 |
| 0.2 | 1.12 | 1.08 | 1.12 | 1.09 | 1.01 | 0.96 | 0.91 | 0.95 | 1.11 |
| 0.5 | 0.72 | 0.72 | 0.68 | 0.70 | 0.67 | 0.71 | 0.73 | 0.80 | 0.84 |
| 1 | 0.47 | 0.46 | 0.46 | 0.47 | 0.49 | 0.50 | 0.56 | 0.65 | 0.77 |
| 2 | 0.38 | 0.38 | 0.38 | 0.39 | 0.39 | 0.39 | 0.42 | 0.46 | 0.58 |

Latency (s)

| w_C \ w_L | 0 | 0.01 | 0.02 | 0.05 | 0.1 | 0.2 | 0.5 | 1 | 2 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 5.26 | 5.23 | 5.16 | 5.05 | 4.62 | 4.22 | 3.55 | 3.47 | 3.47 |
| 0.01 | 5.15 | 5.03 | 4.99 | 4.78 | 4.39 | 4.01 | 3.51 | 3.45 | 3.44 |
| 0.02 | 4.93 | 4.80 | 4.85 | 4.54 | 4.28 | 3.94 | 3.53 | 3.46 | 3.46 |
| 0.05 | 4.36 | 4.32 | 4.38 | 4.18 | 4.08 | 3.85 | 3.45 | 3.43 | 3.41 |
| 0.1 | 4.16 | 4.23 | 4.14 | 3.93 | 3.91 | 3.74 | 3.47 | 3.41 | 3.41 |
| 0.2 | 3.83 | 3.77 | 3.87 | 3.82 | 3.65 | 3.52 | 3.42 | 3.42 | 3.42 |
| 0.5 | 3.67 | 3.64 | 3.60 | 3.56 | 3.64 | 3.59 | 3.47 | 3.45 | 3.41 |
| 1 | 3.71 | 3.73 | 3.72 | 3.71 | 3.70 | 3.69 | 3.66 | 3.58 | 3.51 |
| 2 | 3.75 | 3.75 | 3.75 | 3.73 | 3.74 | 3.74 | 3.73 | 3.69 | 3.55 |

## 3. Best trade-offs (no other setting is both more accurate and cheaper)

| w_C | w_L | Accuracy | Cost | Latency |
|---|---|---|---|---|
| 2 | 0 | 62.4% +- 2.3% | $0.38 | 3.75 s |
| 0.5 | 0 | 65.7% +- 1.1% | $0.72 | 3.67 s |
| 0.5 | 2 | 66.9% +- 2.6% | $0.84 | 3.41 s |
| 0.2 | 1 | 68.5% +- 2.1% | $0.95 | 3.42 s |
| 0.1 | 0.1 | 69.9% +- 2.7% | $1.45 | 3.91 s |
| 0.1 | 0.01 | 71.0% +- 1.1% | $1.68 | 4.23 s |
| 0.02 | 0.05 | 72.7% +- 2.4% | $3.80 | 4.54 s |
| 0 | 0.02 | 75.0% +- 1.8% | $7.09 | 5.16 s |

## 4. Savings against always using Opus (best setting per level, chosen after the runs)

| Quality kept | Setting (w_C, w_L) | Accuracy | Cost | Latency | Cost saved | Latency saved |
|---|---|---|---|---|---|---|
| >= 95% (95.0%) | 0, 0.01 | 73.8% | $6.78 | 5.23 s | 42% | 13% |
| >= 90% (90.1%) | 0.1, 0.1 | 69.9% | $1.45 | 3.91 s | 88% | 35% |
| >= 85% (85.3%) | 1, 2 | 66.3% | $0.77 | 3.51 s | 93% | 41% |
| >= 80% (80.4%) | 2, 0 | 62.4% | $0.38 | 3.75 s | 97% | 37% |

## 5. Router against mixing single models at the same cost

Average -1.2 pts; router better in 11 of 81 settings; best +1.0, worst -3.9.

## 6. Online learning

Learning minus frozen router: +0.07 pts on average (better in 35, worse in 40 of 81 settings). Share of Opus's quality, first half of the stream 87.2%, second half 88.4% (different questions, a rough hint only).

## 7. Reputation (averaged over the 81 w_C x w_L settings)

| w_R | Claude share | Accuracy | Cost | Latency |
|---|---|---|---|---|
| 0 | 46% | 68.2% | $1.98 | 3.92 s |
| 0.02 | 51% | 67.8% | $2.10 | 3.93 s |
| 0.05 | 58% | 67.6% | $2.25 | 3.95 s |
| 0.1 | 70% | 67.3% | $2.55 | 3.98 s |
| 0.2 | 87% | 66.4% | $2.84 | 4.04 s |
| 0.5 | 98% | 65.7% | $2.93 | 4.09 s |

## 8. Which models are picked

| Setting (w_C, w_L) | Haiku | Sonnet | Opus | Terra | Sol |
|---|---|---|---|---|---|
| quality first (0, 0.02) | 4% | 15% | 46% | 9% | 27% |
| balanced (0.1, 0.01) | 13% | 4% | 0% | 36% | 46% |
| cheap and fast (0.5, 2) | 29% | 0% | 0% | 71% | 0% |
| cheapest (2, 0) | 97% | 0% | 0% | 3% | 0% |
