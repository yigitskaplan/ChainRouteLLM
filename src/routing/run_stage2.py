"""Stage 2: our router on MixLLM's 80/20 split with 11 models.

    python -m src.routing.run_stage2 --data data/processed/11_models/routerbench_0shot.csv.gz \
        --config configs/experiment_mixllm_8020.json --splits-dir data/splits/80_20

For every seed the guessers learn from the training questions and the router
picks a model for every test question. With ``--online`` the router first keeps
learning from the split's online questions (online.py); with ``--test-updates``
it also learns from each test question after it is scored. Like MixLLM,
quality and cost are given as a share of always using GPT-4 on the same test
questions; ``quality`` is the mean score, as in MixLLM's Table 1. Prints the
mean ± sd over the seeds and saves every seed's numbers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.data.splits import load_split
from src.data.tables import PathLike, read_table, table_stem
from src.routing.embeddings import cache_path, load_embeddings
from src.routing.online import learn_online
from src.routing.predictors import Predictors
from src.routing.router import Router


def real_outcomes(frame: pd.DataFrame, models: Sequence[str]) -> Dict[str, np.ndarray]:
    """Each model's real accuracy and cost (USD) on every row, each [rows, models]."""
    return {key: frame[[f"{model}|{key}" for model in models]].to_numpy(dtype=float) for key in ("quality", "cost")}


def route_test(
    data_path: PathLike,
    split_path: PathLike,
    config: Mapping[str, Any],
    seed: int,
    online: bool = False,
    test_updates: bool = False,
) -> pd.DataFrame:
    """The model picked for every test question: one column per w_C.

    With ``online`` the router first learns from the online questions. With
    ``test_updates`` it also keeps learning during the test: each test question
    is scored first, then learned from. Learning runs separately for each w_C
    because the picks, and so the answers it sees, differ.
    """
    models = config["data"]["model_keys_primary"]
    settings = config["router"]
    bert_file = cache_path(data_path, settings["encoder_checkpoint"])
    parts = load_split(data_path, split_path)
    offline = pd.concat([parts["offline_fit"], parts["offline_val"]])
    test = parts["test"]

    # Learn to guess each model's accuracy and cost from BERT's numbers.
    X_offline, offline_outcomes = load_embeddings(bert_file, offline["sample_id"]), real_outcomes(offline, models)
    predictors = Predictors(models, settings["predictors"]["random_forest"], seed).fit(X_offline, offline_outcomes)
    X_test = load_embeddings(bert_file, test["sample_id"])
    offline_guess = predictors.predict(X_test)
    # The questions the router learns from, in order: the online ones, then the test ones.
    learn = [parts["online"]] * online + [test] * test_updates
    if learn:
        stream = pd.concat(learn, ignore_index=True)
        X_stream, stream_outcomes = load_embeddings(bert_file, stream["sample_id"]), real_outcomes(stream, models)

    # Score every model and pick the best, for each w_C.
    picks = pd.DataFrame({"sample_id": test["sample_id"]})
    for w_C in settings["w_C_grid"]:
        router = Router(w_C, predictors.cost_scale_)
        chosen = router.route(offline_guess)
        if learn:
            # Keep learning from the stream; each pick is made before its answer is seen.
            learned, stream_picks = learn_online(
                predictors, router, X_offline, offline_outcomes, X_stream, stream_outcomes,
                int(settings["update_interval"]),
            )
            # With test updates the test questions are the end of the stream; otherwise they are only routed.
            chosen = stream_picks[len(stream) - len(test) :] if test_updates else router.route(learned.predict(X_test))
        picks[f"w_C={w_C:g}"] = np.asarray(models)[chosen]
    return picks


def measure(table: pd.DataFrame, picks: pd.DataFrame, models: Sequence[str], reference: str) -> pd.DataFrame:
    """Mean quality, and quality and cost as a share of always picking ``reference``, for every w_C column."""
    real = real_outcomes(table.set_index("sample_id").loc[picks["sample_id"]], models)
    rows = np.arange(len(picks))
    ref = list(models).index(reference)
    shares = []
    for column in picks.columns.drop("sample_id"):
        chosen = pd.Index(models).get_indexer(picks[column])
        shares.append(
            {
                "w_C": float(column.split("=")[1]),
                "quality": real["quality"][rows, chosen].mean(),
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
    parser.add_argument("--test-updates", action="store_true", help="keep learning during the test, after scoring")
    parser.add_argument("--w-C", type=float, nargs="+", default=None, help="default: the config's w_C_grid")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="default: the config's seeds")
    parser.add_argument("--out", type=Path, default=None, help="default: results/stage2/<mode>_<splits dir>.csv")
    args = parser.parse_args(argv)
    mode = "_".join(["online"] * args.online + ["test_updates"] * args.test_updates) or "offline"
    out = args.out or Path(f"results/stage2/{mode}_{args.splits_dir.name}.csv")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.w_C:
        config["router"]["w_C_grid"] = args.w_C
    seeds = args.seeds or [int(seed) for seed in config["seeds"]]
    models = config["data"]["model_keys_primary"]
    reference = config["data"]["reference_model"]
    table = read_table(args.data, columns=["sample_id"] + [f"{m}|{k}" for m in models for k in ("quality", "cost")])

    results = []
    for seed in seeds:
        split_path = args.splits_dir / f"{table_stem(args.data)}_seed{seed}.csv.gz"
        picks = route_test(args.data, split_path, config, seed, online=args.online, test_updates=args.test_updates)
        results.append(measure(table, picks, models, reference).assign(seed=seed))
        shares = results[-1].itertuples()
        print(f"seed {seed}: " + ", ".join(f"w_C={r.w_C:g} {r.quality_vs_gpt4:.2%}/{r.cost_vs_gpt4:.2%}" for r in shares))
    results = pd.concat(results, ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False)

    summary = results.groupby("w_C")[["quality", "quality_vs_gpt4", "cost_vs_gpt4"]].agg(["mean", "std"])
    print(f"{mode}: share of always using {reference} on the test questions, mean ± sd over {len(seeds)} seeds")
    for w_C, row in summary.iterrows():
        mean_quality, quality, cost = row["quality"], row["quality_vs_gpt4"], row["cost_vs_gpt4"]
        print(
            f"w_C={w_C:<5g} quality {quality['mean']:7.2%} ± {quality['std']:.2%}"
            f"   cost {cost['mean']:7.2%} ± {cost['std']:.2%}"
            f"   mean quality {mean_quality['mean']:.2%}"
        )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
