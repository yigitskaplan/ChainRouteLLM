"""Phase 5: data, frozen offline predictors and their cached guesses.

The offline random forests are exactly the Phase 4 / Stage 1 ones (src.routing.predictors, same
settings and seeds). They are fitted once per (pool, seed); their guesses for the online and test
queries are cached in results/phase5/cache/ so every Phase 5 experiment reuses the same router.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.data.splits import load_split
from src.data.tables import table_stem
from src.routing.embeddings import cache_path, load_embeddings
from src.routing.predictors import Predictors

POOLS = {
    # primary design: 5 proprietary models, 30/60/10 (offline / online stream / test)
    "p5": {"data": "data/processed/routerbench_0shot.csv.gz", "config": "configs/experiment_historical.json",
           "splits": "data/splits"},
    # MixLLM comparison condition: 11 models, 80/20
    "p11": {"data": "data/processed/11_models/routerbench_0shot.csv.gz",
            "config": "configs/experiment_mixllm_8020.json", "splits": "data/splits/80_20"},
}
CACHE = Path("results/phase5/cache")


@dataclass
class Part:
    ids: np.ndarray
    X: np.ndarray            # BERT embeddings
    Q: np.ndarray            # real quality [rows, models]
    C: np.ndarray            # real billed cost USD [rows, models]
    task: np.ndarray         # RouterBench eval_name (analysis only, never used for routing)
    A_hat: np.ndarray = None  # offline-forest guessed quality
    C_hat: np.ndarray = None  # offline-forest guessed cost


@dataclass
class Fold:
    pool: str
    seed: int
    models: List[str]
    cost_scale: float        # mean offline cost over all models (the router's c-bar)
    offline: Part
    online: Part             # empty for the 80/20 pool
    test: Part
    fit_seconds: float = 0.0


def _part(frame: pd.DataFrame, models, bert_file) -> Part:
    ids = frame["sample_id"].to_numpy()
    X = load_embeddings(bert_file, ids) if len(frame) else np.zeros((0, 768), np.float32)
    Q = frame[[f"{m}|quality" for m in models]].to_numpy(float)
    C = frame[[f"{m}|cost" for m in models]].to_numpy(float)
    task = frame["benchmark"].to_numpy() if "benchmark" in frame else np.array([""] * len(frame))
    return Part(ids, X, Q, C, task)


def load_fold(pool: str, seed: int, refit: bool = False) -> Fold:
    spec = POOLS[pool]
    cfg = json.loads(Path(spec["config"]).read_text(encoding="utf-8"))
    models = cfg["data"]["model_keys_primary"]
    settings = cfg["router"]
    bert_file = cache_path(spec["data"], settings["encoder_checkpoint"])
    split_path = Path(spec["splits"]) / f"{table_stem(spec['data'])}_seed{seed}.csv.gz"
    parts = load_split(Path(spec["data"]), split_path)
    offline = _part(pd.concat([parts["offline_fit"], parts["offline_val"]], ignore_index=True), models, bert_file)
    online = _part(parts["online"], models, bert_file)
    test = _part(parts["test"], models, bert_file)
    cache = CACHE / f"{pool}_seed{seed}.npz"
    t = 0.0
    if cache.exists() and not refit:
        z = np.load(cache, allow_pickle=False)
        assert list(z["models"]) == models and (z["test_ids"] == np.asarray(test.ids, dtype=str)).all()
        online.A_hat, online.C_hat = z["online_A"], z["online_C"]
        test.A_hat, test.C_hat = z["test_A"], z["test_C"]
        scale, t = float(z["cost_scale"]), float(z["fit_seconds"])
    else:
        t0 = time.time()
        pred = Predictors(models, settings["predictors"]["random_forest"], seed).fit(
            offline.X, {"quality": offline.Q, "cost": offline.C})
        scale = pred.cost_scale_
        for p in (online, test):
            if len(p.ids):
                g = pred.predict(p.X)
                p.A_hat, p.C_hat = g["quality"], g["cost"]
            else:
                p.A_hat = p.C_hat = np.zeros((0, len(models)))
        t = time.time() - t0
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez(cache, models=np.asarray(models, dtype=str), test_ids=np.asarray(test.ids, dtype=str), online_A=online.A_hat, online_C=online.C_hat,
                 test_A=test.A_hat, test_C=test.C_hat, cost_scale=scale, fit_seconds=t)
    return Fold(pool, seed, models, scale, offline, online, test, t)


def seeds_of(pool: str) -> List[int]:
    cfg = json.loads(Path(POOLS[pool]["config"]).read_text(encoding="utf-8"))
    return [int(s) for s in cfg["seeds"]]


def w_grid(pool: str) -> List[float]:
    cfg = json.loads(Path(POOLS[pool]["config"]).read_text(encoding="utf-8"))
    return [float(w) for w in cfg["router"]["w_C_grid"]]


def config_of(pool: str) -> Dict:
    return json.loads(Path(POOLS[pool]["config"]).read_text(encoding="utf-8"))
