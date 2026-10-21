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

## Data preparation

`data/DataAnalysis.ipynb` looks at the data and exports `df0_filtered.csv` and
`df5_filtered.csv` for reading. The experiments use the scripts' own table in
`data/processed/`.

Python 3.9 or newer. Run from the repository root:

```bash
pip install -r requirements.txt
python -m src.data.download_routerbench   # RouterBench pickles -> data/
python -m src.data.build_dataset --pickle data/routerbench_0shot.pkl
python -m src.data.splits --data data/processed/routerbench_0shot.csv.gz \
    --config configs/experiment_historical.json
python -m pytest
```

`build_dataset` keeps each model's score, cost and answer length, cleans the
rows in code and writes `data/processed/<name>.csv.gz` with a `.report.json`
that lists every removed row and the file checksums. `splits` writes one
manifest per seed to `data/splits/`; rows with the same prompt stay in the same
split. The processed tables are rebuilt by the script and stay out of Git; the
reports and split manifests are small and are committed.

Do not open and re-save these CSV files in Excel: it changes the delimiter and
the text encoding, and non-English prompts get garbled.
