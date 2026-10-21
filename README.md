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

## Experimental specification

Step 1 is recorded in [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) and
[configs/experiment.json](configs/experiment.json). The configuration describes
the planned experiments; an executable runner has not been implemented yet.
Unresolved prerequisites are listed explicitly and must be resolved before full runs.

Implementation proceeds in three phases: **router first, reputation score second,
blockchain ledger last**. The current phase builds and evaluates the base router
using local feedback storage. Reputation and blockchain integration are deferred.

Local RouterBench pickle downloads live in `data/` and are excluded from Git.
Dependencies and executable implementation code have not been added yet.
