"""Stage 3: the router with latency and online learning on the live answers.

    python -m src.data.live_table
    python -m src.routing.embeddings --data data/processed/supergpqa_live.csv.gz \
        --config configs/experiment_stage3.json
    python -m src.data.splits --data data/processed/supergpqa_live.csv.gz \
        --config configs/experiment_stage3.json --out-dir data/splits/live_30_70
    python -m src.routing.run_stage3 --config configs/experiment_stage3.json

Online evaluation the way online routers are evaluated (warm start, then a
stream graded as it goes; "test-then-train"). For every seed the manifest in
data/splits/live_30_70/ gives a warm start (30%), where every model's answer is
known, and a stream (70%) in arrival order. The guessers learn accuracy, cost
and latency from the warm start. The stream comes in blocks: every question of
a block is routed and scored before any of its answers is seen, then the
router learns from the block (online.py: the picked models' forests are
retrained and c and l take in the answers). The router is graded on the whole
stream, for every (w_C, w_L) pair, next to the same router without learning,
each single model, a random pick and the oracle (a right model, cheapest
first), all on the same stream questions. Quality (share right), cost (USD) and
latency (seconds) are reported as means and as a share of always using the
reference model, for the whole stream and for its first and second half.
Prints the mean ± sd over the seeds and saves every seed's numbers.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.data.tables import read_table, table_stem
from src.routing.embeddings import cache_path, load_embeddings
from src.routing.online import learn_online
from src.routing.predictors import Predictors
from src.routing.router import Router, Scales

KEYS = ("quality", "cost", "latency")


def real_outcomes(frame: pd.DataFrame, models: Sequence[str]) -> Dict[str, np.ndarray]:
    """Each model's real accuracy, cost (USD) and latency (seconds) on every row, each [rows, models]."""
    return {key: frame[[f"{model}|{key}" for model in models]].to_numpy(dtype=float) for key in KEYS}


def oracle(real: Mapping[str, np.ndarray]) -> np.ndarray:
    """For every row a model that is right, the cheapest one; the cheapest model if none is."""
    return np.lexsort((real["cost"].T, -real["quality"].T), axis=0)[0]


def measure(real: Mapping[str, np.ndarray], picks: np.ndarray, reference: int, models: Sequence[str]) -> Dict[str, float]:
    """Means and shares of always using ``reference`` over all rows, and the quality share in each half."""
    rows = np.arange(len(picks))
    out = {}
    for key in KEYS:
        chosen = real[key][rows, picks]
        out[key] = chosen.mean()
        out[f"{key}_vs_ref"] = chosen.sum() / real[key][:, reference].sum()
    half = len(picks) // 2
    for name, part in (("first_half", slice(0, half)), ("second_half", slice(half, None))):
        chosen = real["quality"][rows[part], picks[part]]
        out[f"quality_vs_ref_{name}"] = chosen.sum() / real["quality"][part, reference].sum()
    counts = np.bincount(picks, minlength=len(models)) / len(picks)
    out.update({f"picks_{model}": share for model, share in zip(models, counts)})
    return out


def run_seed(table: pd.DataFrame, X: np.ndarray, config: Mapping[str, Any], split_path: Path, seed: int) -> pd.DataFrame:
    models = config["data"]["model_keys_primary"]
    reference = models.index(config["data"]["reference_model"])
    settings = config["router"]
    split = read_table(split_path).set_index("sample_id")
    position = pd.Series(np.arange(len(table)), index=table["sample_id"])

    def rows(name: str) -> np.ndarray:
        part = split[split["split"] == name].sort_values("order")  # stream rows in arrival order
        return position.loc[part.index].to_numpy()

    warm = np.concatenate([rows("offline_fit"), rows("offline_val")])
    stream = np.concatenate([rows("online"), rows("test")])
    real = real_outcomes(table, models)
    part = lambda index: {key: value[index] for key, value in real.items()}

    guessers = Predictors(models, settings["predictors"]["random_forest"], seed).fit(X[warm], part(warm))
    frozen_guess = guessers.predict(X[stream])
    block = int(settings["online_block"])
    results = []
    for w_C, w_L in itertools.product(settings["w_C_grid"], settings["w_L_grid"]):
        # Learning: every block is routed and scored first, then learned from.
        scales = Scales(part(warm))
        _, picks = learn_online(guessers, Router(w_C, scales, w_L), X[warm], part(warm), X[stream], part(stream), block)
        results.append(dict(method="router", w_C=w_C, w_L=w_L, c=scales.c, l=scales.l,
                            **measure(part(stream), picks, reference, models)))
        # The same router without learning: warm-start guessers, c and l throughout.
        frozen = Router(w_C, Scales(part(warm)), w_L).route(frozen_guess)
        results.append(dict(method="router, no learning", w_C=w_C, w_L=w_L,
                            **measure(part(stream), frozen, reference, models)))

    rng = np.random.default_rng(seed)
    baselines = {f"only {model}": np.full(len(stream), j) for j, model in enumerate(models)}
    baselines["random"] = rng.integers(0, len(models), len(stream))
    baselines["oracle"] = oracle(part(stream))
    for name, picks in baselines.items():
        results.append(dict(method=name, **measure(part(stream), picks, reference, models)))
    return pd.DataFrame(results).assign(seed=seed, warm_start=len(warm), stream_questions=len(stream))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the Stage 3 router online on the live answers.")
    parser.add_argument("--data", type=Path, default=Path("data/processed/supergpqa_live.csv.gz"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits/live_30_70"))
    parser.add_argument("--out", type=Path, default=None, help="default: results/stage3/<splits dir>.csv")
    args = parser.parse_args(argv)
    out = args.out or Path(f"results/stage3/{args.splits_dir.name}.csv")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    models = config["data"]["model_keys_primary"]
    table = read_table(args.data)
    X = load_embeddings(cache_path(args.data, config["router"]["encoder_checkpoint"]), table["sample_id"])

    splits = {int(seed): args.splits_dir / f"{table_stem(args.data)}_seed{seed}.csv.gz" for seed in config["seeds"]}
    results = pd.concat([run_seed(table, X, config, path, seed) for seed, path in splits.items()], ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False)

    reference = config["data"]["reference_model"]
    columns = ["quality_vs_ref", "cost_vs_ref", "latency_vs_ref"]
    summary = results.groupby(["method", "w_C", "w_L"], dropna=False, sort=False)
    stats = summary[columns].agg(["mean", "std"])
    halves = summary[["quality_vs_ref_first_half", "quality_vs_ref_second_half"]].mean()
    picks = summary[[f"picks_{m}" for m in models]].mean()
    print(f"warm start {results.warm_start.iloc[0]}, stream {results.stream_questions.iloc[0]} questions, "
          f"{len(config['seeds'])} seeds; quality / cost / latency as a share of always using {reference}, mean ± sd; "
          "quality in the stream's first -> second half; picks")
    # The three summaries share the group order; baselines have no w_C / w_L, so match rows by position.
    for ((method, w_C, w_L), row), (_, h), (_, p) in zip(stats.iterrows(), halves.iterrows(), picks.iterrows()):
        name = f"{method} w_C={w_C:g} w_L={w_L:g}" if method.startswith("router") else method
        shares = "   ".join(f"{row[(k, 'mean')]:6.1%} ± {row[(k, 'std')]:4.1%}" for k in columns)
        print(f"{name:40s} {shares}   {h.iloc[0]:6.1%} -> {h.iloc[1]:6.1%}   " + "/".join(f"{v:.0%}" for v in p))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
