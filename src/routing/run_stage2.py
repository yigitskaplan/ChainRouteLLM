"""Stage 2: our router on MixLLM's 80/20 split with 11 models.

    python -m src.routing.run_stage2 --data data/processed/11_models/routerbench_0shot.csv.gz \
        --config configs/experiment_mixllm_8020.json --splits-dir data/splits/80_20

For every seed the guessers learn from the 80% training questions and the router
picks a model for every test question (run_stage1.route_test). With ``--online``
the router first keeps learning from the split's online questions. Like MixLLM,
quality and cost are given as a share of always using GPT-4 on the same test
questions. Prints the mean ± sd over the seeds and saves every seed's numbers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.data.tables import read_table, table_stem
from src.routing.run_stage1 import real_outcomes, route_test


def measure(table: pd.DataFrame, picks: pd.DataFrame, models: Sequence[str], reference: str) -> pd.DataFrame:
    """Quality and cost of every w_C column of ``picks`` as a share of always picking ``reference``."""
    real = real_outcomes(table.set_index("sample_id").loc[picks["sample_id"]], models)
    rows = np.arange(len(picks))
    ref = list(models).index(reference)
    shares = []
    for column in picks.columns.drop("sample_id"):
        chosen = pd.Index(models).get_indexer(picks[column])
        shares.append(
            {
                "w_C": float(column.split("=")[1]),
                "quality_vs_gpt4": real["quality"][rows, chosen].sum() / real["quality"][:, ref].sum(),
                "cost_vs_gpt4": real["cost"][rows, chosen].sum() / real["cost"][:, ref].sum(),
            }
        )
    return pd.DataFrame(shares)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Compare the router with MixLLM on the 80/20 split.")
    parser.add_argument("--data", required=True, type=Path, help="table from src.data.build_dataset")
    parser.add_argument("--config", required=True, type=Path, help="experiment config")
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits/80_20"))
    parser.add_argument("--online", action="store_true", help="learn from the online questions first")
    parser.add_argument("--out", type=Path, default=None, help="default: results/stage2/<offline|online>_80_20.csv")
    args = parser.parse_args(argv)
    out = args.out or Path(f"results/stage2/{'online' if args.online else 'offline'}_80_20.csv")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    models = config["data"]["model_keys_primary"]
    reference = config["data"]["reference_model"]
    table = read_table(args.data, columns=["sample_id"] + [f"{m}|{k}" for m in models for k in ("quality", "cost")])

    results = []
    for seed in config["seeds"]:
        split_path = args.splits_dir / f"{table_stem(args.data)}_seed{seed}.csv.gz"
        picks = route_test(args.data, split_path, config, int(seed), online=args.online)
        results.append(measure(table, picks, models, reference).assign(seed=int(seed)))
        print(f"seed {seed} done")
    results = pd.concat(results, ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False)

    summary = results.groupby("w_C")[["quality_vs_gpt4", "cost_vs_gpt4"]].agg(["mean", "std"])
    print(f"share of always using {reference} on the test questions, mean ± sd over {len(config['seeds'])} seeds")
    for w_C, row in summary.iterrows():
        quality, cost = row["quality_vs_gpt4"], row["cost_vs_gpt4"]
        print(
            f"w_C={w_C:<5g} quality {quality['mean']:7.2%} ± {quality['std']:.2%}"
            f"   cost {cost['mean']:7.2%} ± {cost['std']:.2%}"
        )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
