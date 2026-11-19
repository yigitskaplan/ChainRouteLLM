"""Stage 4: our router on LLMRouterBench's own benchmark, protocol and metrics.

    python -m src.data.llmrouterbench
    python -m src.routing.embeddings --data data/processed/llmrouterbench_perf_cost.csv.gz \
        --config configs/experiment_stage4_llmrouterbench.json
    python -m src.routing.run_stage4 --config configs/experiment_stage4_llmrouterbench.json

LLMRouterBench (Li et al., arXiv 2601.07206) scores routers between 13 flagship
models on 10 datasets with its own 70/30 split and five seeds (src/data/
llmrouterbench.py redoes the split exactly). Here the 70% is the warm start,
where every model's answer is known, and the 30% test part is the stream: block
by block it is routed and scored first, then learned from (online.py), with c
following the evidence (router.Scales). The data has no latency, so the score
is quality and cost only. Scores are the paper's:

* AvgAcc: the mean over the datasets of the mean score (0-100) of the picked answers;
* Cost: the total USD of the picked answers on the test part.

Next to the router (with and without learning, for every w_C) it reports every
single model, a random pick and the oracle (the best score, cheapest first) on
the same questions, and prints the paper's own numbers for comparison. Writes
results/stage4/llmrouterbench.csv (one row per method, w_C and seed).
"""
from __future__ import annotations

import argparse
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

KEYS = ("quality", "cost")


def real_outcomes(frame: pd.DataFrame, models: Sequence[str]) -> Dict[str, np.ndarray]:
    """Each model's real score and cost (USD) on every row, each [rows, models]."""
    return {key: frame[[f"{model}|{key}" for model in models]].to_numpy(dtype=float) for key in KEYS}


def oracle(real: Mapping[str, np.ndarray]) -> np.ndarray:
    """For every row the model with the best score, the cheapest one among equals."""
    return np.lexsort((real["cost"].T, -real["quality"].T), axis=0)[0]


def measure(real: Mapping[str, np.ndarray], picks: np.ndarray, datasets: np.ndarray, models: Sequence[str]) -> Dict[str, float]:
    """The paper's AvgAcc and Cost of the picked answers, each dataset's score, and the share of each model."""
    chosen = real["quality"][np.arange(len(picks)), picks]
    out = {f"acc_{name}": 100 * chosen[datasets == name].mean() for name in pd.unique(datasets)}
    out = {"avg_acc": float(np.mean(list(out.values()))), "cost": real["cost"][np.arange(len(picks)), picks].sum(), **out}
    out.update({f"picks_{model}": share for model, share in zip(models, np.bincount(picks, minlength=len(models)) / len(picks))})
    return out


def run_seed(table: pd.DataFrame, X: np.ndarray, config: Mapping[str, Any], split_path: Path, seed: int) -> pd.DataFrame:
    models = config["data"]["model_keys_primary"]
    settings = config["router"]
    split = read_table(split_path).set_index("sample_id")
    position = pd.Series(np.arange(len(table)), index=table["sample_id"])
    rows = lambda name: position.loc[split[split["split"] == name].sort_values("order").index].to_numpy()
    warm, stream = rows("offline_fit"), rows("online")  # the paper's 70% and 30%; the stream in arrival order
    real = real_outcomes(table, models)
    part = lambda index: {key: value[index] for key, value in real.items()}
    datasets = table["dataset"].to_numpy()[stream]

    guessers = Predictors(models, settings["predictors"]["random_forest"], seed).fit(X[warm], part(warm))
    frozen_guess = guessers.predict(X[stream])
    results = []
    for w_C in settings["w_C_grid"]:
        scales = Scales(part(warm))
        _, picks = learn_online(guessers, Router(w_C, scales), X[warm], part(warm), X[stream], part(stream), int(settings["online_block"]))
        results.append(dict(method="router", w_C=w_C, c=scales.c, **measure(part(stream), picks, datasets, models)))
        frozen = Router(w_C, Scales(part(warm))).route(frozen_guess)
        results.append(dict(method="router, no learning", w_C=w_C, **measure(part(stream), frozen, datasets, models)))
        print(f"  seed {seed} w_C {w_C:g}: AvgAcc {results[-2]['avg_acc']:.2f}, cost ${results[-2]['cost']:.2f}", flush=True)

    baselines = {f"only {model}": np.full(len(stream), j) for j, model in enumerate(models)}
    baselines["random"] = np.random.default_rng(seed).integers(0, len(models), len(stream))
    baselines["oracle"] = oracle(part(stream))
    for name, picks in baselines.items():
        results.append(dict(method=name, **measure(part(stream), picks, datasets, models)))
    return pd.DataFrame(results).assign(seed=seed, warm_start=len(warm), stream_questions=len(stream))


def summary(results: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    """Mean over seeds, with the paper's PerfGain and CostSave against the best single model."""
    mean = results.groupby(["method", "w_C"], dropna=False, sort=False)[["avg_acc", "cost"]].mean().reset_index()
    best = mean[mean["method"] == f"only {config['data']['reference_model']}"].iloc[0]
    mean["perf_gain"] = mean["avg_acc"] / best["avg_acc"] - 1
    mean["cost_save"] = np.where(mean["avg_acc"] >= best["avg_acc"], 1 - mean["cost"] / best["cost"], np.nan)
    return mean


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Our router on LLMRouterBench (performance-cost setting).")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/stage4/llmrouterbench.csv"))
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    data = Path(config["data"]["path"])
    models = config["data"]["model_keys_primary"]
    table = read_table(data, columns=["sample_id", "dataset"] + [f"{m}|{k}" for m in models for k in KEYS])
    X = load_embeddings(cache_path(data, config["router"]["encoder_checkpoint"]), table["sample_id"])

    splits_dir = Path(config["data"]["splits_dir"])
    results = pd.concat(
        [run_seed(table, X, config, splits_dir / f"{table_stem(data)}_seed{seed}.csv.gz", int(seed)) for seed in config["seeds"]],
        ignore_index=True,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out, index=False)

    mean = summary(results, config)
    print(f"\nLLMRouterBench performance-cost, {results.stream_questions.iloc[0]} test questions, {len(config['seeds'])} seeds (mean)")
    for _, row in mean.iterrows():
        name = f"{row.method} w_C={row.w_C:g}" if row.method.startswith("router") else row.method
        save = f"  cost saved {row.cost_save:.1%}" if row.cost_save == row.cost_save else ""
        print(f"  {name:40s} AvgAcc {row.avg_acc:6.2f}  cost ${row.cost:7.2f}  vs best single {row.perf_gain:+.1%}{save}")
    print("published (same test parts):")
    for item in [config["published"]["best_single"]] + config["published"]["routers"]:
        print(f"  {item['name']:40s} AvgAcc {item['avg_acc']:6.2f}  cost ${item['cost']:7.2f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
