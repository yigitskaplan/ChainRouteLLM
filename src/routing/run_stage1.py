"""Run the Stage 1 router: guess accuracy and cost, then pick a model.

    python -m src.routing.run_stage1 --data data/processed/routerbench_0shot.csv.gz \
        --config configs/experiment_historical.json

Needs the split files (src.data.splits) and BERT's numbers (src.routing.embeddings).
The guessers learn from the offline questions of one seed; the router then picks
a model for every test question, once for each w_C in the config, and prints
how often each model is picked.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.data.splits import load_split
from src.data.tables import PathLike, table_stem
from src.routing.embeddings import cache_path, load_embeddings
from src.routing.predictors import Predictors
from src.routing.router import Router


def real_outcomes(frame: pd.DataFrame, models: Sequence[str]) -> Dict[str, np.ndarray]:
    """Each model's real accuracy and cost (USD) on every row, each [rows, models]."""
    return {key: frame[[f"{model}|{key}" for model in models]].to_numpy(dtype=float) for key in ("quality", "cost")}


def route_test(data_path: PathLike, split_path: PathLike, config: Mapping[str, Any], seed: int) -> pd.DataFrame:
    """The model picked for every test question: one column per w_C."""
    models = config["data"]["model_keys_primary"]
    settings = config["router"]
    bert_file = cache_path(data_path, settings["encoder_checkpoint"])
    parts = load_split(data_path, split_path)
    offline = pd.concat([parts["offline_fit"], parts["offline_val"]])
    test = parts["test"]

    # Step 2: learn to guess each model's accuracy and cost from BERT's numbers.
    predictors = Predictors(models, settings["predictors"]["random_forest"], seed)
    predictors.fit(load_embeddings(bert_file, offline["sample_id"]), real_outcomes(offline, models))
    guess = predictors.predict(load_embeddings(bert_file, test["sample_id"]))

    # Step 3: score every model and pick the best, for each w_C.
    picks = pd.DataFrame({"sample_id": test["sample_id"]})
    for w_C in settings["w_C_grid"]:
        picks[f"w_C={w_C:g}"] = np.asarray(models)[Router(w_C, predictors.cost_scale_).route(guess)]
    return picks


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Stage 1 router on the test questions of one seed.")
    parser.add_argument("--data", required=True, type=Path, help="table from src.data.build_dataset")
    parser.add_argument("--config", required=True, type=Path, help="experiment config")
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--seed", type=int, default=None, help="default: the config's first seed")
    args = parser.parse_args(argv)

    config = json.loads(args.config.read_text(encoding="utf-8"))
    seed = args.seed if args.seed is not None else int(config["seeds"][0])
    split_path = args.splits_dir / f"{table_stem(args.data)}_seed{seed}.csv.gz"
    picks = route_test(args.data, split_path, config, seed)

    models = config["data"]["model_keys_primary"]
    shares = picks.drop(columns="sample_id").apply(lambda column: column.value_counts(normalize=True))
    print(f"seed {seed}: share of the {len(picks)} test questions sent to each model")
    print(shares.reindex(models).fillna(0).to_string(float_format=lambda share: f"{share:.1%}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
