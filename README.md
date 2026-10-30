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

## Stage 1: router

Run after the data preparation above:

```bash
python -m src.routing.embeddings --data data/processed/routerbench_0shot.csv.gz \
    --config configs/experiment_historical.json      # BERT, about 17 min on an M3
python -m src.routing.run_stage1 --data data/processed/routerbench_0shot.csv.gz \
    --config configs/experiment_historical.json      # about 1 min
```

`embeddings` turns every question into BERT's numbers and saves them to
`data/embeddings/` (large, stays out of Git). `run_stage1` learns to guess each
model's accuracy and cost from those numbers on the offline questions, picks a
model for every test question with score = accuracy - w_C * cost / c, and prints
how often each model is picked for each w_C. Add `--seed 23` for another split.

## Stage 2: comparison with MixLLM

For a fair comparison only, this test uses MixLLM's setup: 11 models and an
80/20 train/test split. The project design stays 30/60/10 with 5 models.

```bash
python -m src.data.build_dataset --pickle data/routerbench_0shot.pkl \
    --out data/processed/11_models/routerbench_0shot.csv.gz --models gpt-4-1106-preview \
    gpt-3.5-turbo-1106 claude-v1 claude-v2 claude-instant-v1 meta/llama-2-70b-chat \
    mistralai/mixtral-8x7b-chat zero-one-ai/Yi-34B-Chat WizardLM/WizardLM-13B-V1.2 \
    meta/code-llama-instruct-34b-chat mistralai/mistral-7b-chat
python -m src.data.splits --data data/processed/11_models/routerbench_0shot.csv.gz \
    --config configs/experiment_mixllm_8020.json --out-dir data/splits/80_20
python -m src.routing.run_stage2 --data data/processed/11_models/routerbench_0shot.csv.gz \
    --config configs/experiment_mixllm_8020.json --splits-dir data/splits/80_20   # about 30 min
```

The 11-model table has the same questions as the 5-model one, so it reuses the
same BERT file. `run_stage2` runs the router for every seed with no online
learning and prints quality and cost as a share of always using GPT-4, the
way MixLLM reports them. Every seed's numbers go to
`results/stage2/offline_80_20.csv`.

### Online learning

`src/routing/online.py` lets the router keep learning. It sees only the picked
model's real accuracy and cost, and every `update_interval` (1,000) questions
it retrains the forests of the models that got new answers. `--online` learns
from the split's online questions before the test; `--test-updates` also
learns from each test question after it is scored. MixLLM's paper can be read
both ways, so there are three runs:

```bash
python -m src.data.splits --data data/processed/11_models/routerbench_0shot.csv.gz \
    --config configs/experiment_mixllm_64_16_20.json --out-dir data/splits/64_16_20
# 1. 64/16/20: learn online, then test
python -m src.routing.run_stage2 --data data/processed/11_models/routerbench_0shot.csv.gz \
    --config configs/experiment_mixllm_64_16_20.json --splits-dir data/splits/64_16_20 --online
# 2. 64/16/20: learn online, then keep learning during the test
python -m src.routing.run_stage2 --data data/processed/11_models/routerbench_0shot.csv.gz \
    --config configs/experiment_mixllm_64_16_20.json --splits-dir data/splits/64_16_20 --online --test-updates
# 3. 80/20: learn during the test
python -m src.routing.run_stage2 --data data/processed/11_models/routerbench_0shot.csv.gz \
    --config configs/experiment_mixllm_8020.json --splits-dir data/splits/80_20 --test-updates \
    --out results/stage2/online_test_updates_80_20.csv
```

The 64/16/20 split has the same test questions as the 80/20 one. Add `--w-C`
or `--seeds` to run only some w_C values or seeds; results go to
`results/stage2/<mode>_<split>.csv`.
