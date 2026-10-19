# ChainRouteLLM

Experimental LLM routing with interchangeable evidence storage backends.

## Project layout

| Directory | Purpose |
| --- | --- |
| `configs/` | Experiment configurations, model pools, and seeds |
| `data/` | Dataset inputs, processed data, and split manifests |
| `src/data/` | Dataset loading, validation, and preprocessing |
| `src/routing/` | Embeddings, predictors, routing policies, and online learning |
| `src/feedback/` | Feedback simulation and validation |
| `src/evidence/` | Evidence schema, reputation, and storage adapters |
| `src/attacks/` | Controlled attack scenarios |
| `src/evaluation/` | Replay experiments, baselines, and metrics |
| `contracts/` | Solidity contracts and contract tooling |
| `infrastructure/` | Docker, Besu, and benchmark infrastructure |
| `tests/` | Automated tests |
| `results/` | Experiment reports, tables, and plots |

This is the initial directory scaffold. Dependencies, executable code, and experiment configurations have not been added yet.
