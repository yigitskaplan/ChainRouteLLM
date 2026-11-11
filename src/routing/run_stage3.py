"""Stage 3: the router with latency on the live answers.

    python -m src.data.live_table
    python -m src.routing.embeddings --data data/processed/supergpqa_live.csv.gz \
        --config configs/experiment_stage3.json
    python -m src.routing.run_stage3 --config configs/experiment_stage3.json

For every seed the questions are split 30/60/10 (offline / online / test). The
guessers learn accuracy, cost and latency from the offline answers. The online
questions are routed in blocks; the picked model's real cost and latency go
into c and l (router.Scales), the guessers stay as they are. Then every test
question is routed with the final c and l, for every (w_C, w_L) pair. On the
test questions quality (share right), cost (USD) and latency (seconds) are
reported as means and as a share of always using the reference model, next to
each single model, a random pick and the oracle (a right model, cheapest
first). Prints the mean ± sd over the seeds and saves every seed's numbers.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.data.splits import make_split
from src.data.tables import read_table
from src.routing.embeddings import cache_path, load_embeddings
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
    rows = np.arange(len(picks))
    out = {}
    for key in KEYS:
        chosen = real[key][rows, picks]
        out[key] = chosen.mean()
        out[f"{key}_vs_ref"] = chosen.sum() / real[key][:, reference].sum()
    counts = np.bincount(picks, minlength=len(models)) / len(picks)
    out.update({f"picks_{model}": share for model, share in zip(models, counts)})
    return out


def run_seed(table: pd.DataFrame, X: np.ndarray, config: Mapping[str, Any], seed: int) -> pd.DataFrame:
    models = config["data"]["model_keys_primary"]
    reference = models.index(config["data"]["reference_model"])
    settings = config["router"]
    fractions = {key: float(config["data"][key]) for key in
                 ("offline_fraction", "online_fraction", "test_fraction", "validation_fraction_of_offline")}
    split = make_split(table[["sample_id", "group_id"]], seed, **fractions).set_index("sample_id")
    position = pd.Series(np.arange(len(table)), index=table["sample_id"])

    def rows(name: str) -> np.ndarray:
        part = split[split["split"] == name].sort_values("order")  # online rows in arrival order
        return position.loc[part.index].to_numpy()

    offline, online, test = np.concatenate([rows("offline_fit"), rows("offline_val")]), rows("online"), rows("test")
    real = real_outcomes(table, models)
    part = lambda index: {key: value[index] for key, value in real.items()}

    guessers = Predictors(models, settings["predictors"]["random_forest"], seed).fit(X[offline], part(offline))
    online_guess, test_guess = guessers.predict(X[online]), guessers.predict(X[test])
    block = int(settings["online_block"])
    results = []
    for w_C, w_L in itertools.product(settings["w_C_grid"], settings["w_L_grid"]):
        scales = Scales(part(offline))
        router = Router(w_C, scales, w_L)
        # Route the online questions block by block; each picked answer goes into c and l.
        for start in range(0, len(online), block):
            index = np.arange(start, min(start + block, len(online)))
            picks = router.route({key: value[index] for key, value in online_guess.items()})
            for i, m in zip(online[index], picks):
                scales.add(m, real["cost"][i, m], real["latency"][i, m])
        result = measure(part(test), router.route(test_guess), reference, models)
        results.append(dict(method="router", w_C=w_C, w_L=w_L, c=scales.c, l=scales.l, **result))

    rng = np.random.default_rng(seed)
    baselines = {f"only {model}": np.full(len(test), j) for j, model in enumerate(models)}
    baselines["random"] = rng.integers(0, len(models), len(test))
    baselines["oracle"] = oracle(part(test))
    for name, picks in baselines.items():
        results.append(dict(method=name, **measure(part(test), picks, reference, models)))
    return pd.DataFrame(results).assign(seed=seed, test_questions=len(test))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the Stage 3 router with latency on the live answers.")
    parser.add_argument("--data", type=Path, default=Path("data/processed/supergpqa_live.csv.gz"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/stage3/live_claude.csv"))
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    models = config["data"]["model_keys_primary"]
    table = read_table(args.data)
    X = load_embeddings(cache_path(args.data, config["router"]["encoder_checkpoint"]), table["sample_id"])

    results = pd.concat([run_seed(table, X, config, int(seed)) for seed in config["seeds"]], ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out, index=False)

    reference = config["data"]["reference_model"]
    columns = ["quality_vs_ref", "cost_vs_ref", "latency_vs_ref"] + [f"picks_{m}" for m in models]
    summary = results.groupby(["method", "w_C", "w_L"], dropna=False, sort=False)[columns].agg(["mean", "std"])
    print(f"{len(table)} questions, {results.test_questions.iloc[0]} test questions per seed, "
          f"{len(config['seeds'])} seeds; quality / cost / latency as a share of always using {reference}, mean ± sd")
    for (method, w_C, w_L), row in summary.iterrows():
        name = method if method != "router" else f"router w_C={w_C:g} w_L={w_L:g}"
        shares = "   ".join(f"{row[(k, 'mean')]:6.1%} ± {row[(k, 'std')]:4.1%}" for k in columns[:3])
        picks = "/".join(f"{row[(f'picks_{m}', 'mean')]:.0%}" for m in models)
        print(f"{name:28s} {shares}   picks {picks}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
