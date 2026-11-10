"""Phase 4's online learning (forest retraining, src.routing.online) on the 5-model 30/60/10 design,
for comparison with the Phase 5 evidence-corrected router: honest stream and silent substitution.

    python -m src.evaluation.run_phase5_rf --w 0.1

The stream is the online split followed by the test split; every pick is made before its outcome is
used (predict-before-update). Writes results/phase5/e2_rf_online.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.evaluation.phase5_data import config_of, load_fold, seeds_of
from src.routing.online import learn_online
from src.routing.predictors import Predictors
from src.routing.router import Router

OUT = Path("results/phase5")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=float, nargs="+", default=[0.1])
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    args = ap.parse_args(argv)
    cfg = config_of("p5")
    forest = cfg["router"]["predictors"]["random_forest"]
    rows_out = []
    out_path = OUT / "e2_rf_online.json"
    if out_path.exists():
        rows_out = json.loads(out_path.read_text())
    done = {(r["seed"], r["w_C"], r["scenario"]) for r in rows_out}
    for seed in args.seeds or seeds_of("p5"):
        f = load_fold("p5", seed)
        t0 = time.time()
        pred = Predictors(f.models, forest, seed).fit(f.offline.X, {"quality": f.offline.Q, "cost": f.offline.C})
        fit_s = time.time() - t0
        X_s = np.vstack([f.online.X, f.test.X])
        n_on = len(f.online.ids)
        for w in args.w:
            for scen in ("honest", "substitution"):
                if (seed, w, scen) in done:
                    continue
                Q_s = np.vstack([f.online.Q, f.test.Q]).copy()
                if scen == "substitution":
                    Q_s[int(0.25 * n_on):, 0] = Q_s[int(0.25 * n_on):, 1]
                C_s = np.vstack([f.online.C, f.test.C])
                t = time.time()
                _, picks = learn_online(pred, Router(w, pred.cost_scale_), f.offline.X,
                                        {"quality": f.offline.Q, "cost": f.offline.C}, X_s,
                                        {"quality": Q_s, "cost": C_s}, int(1000))
                dt = time.time() - t
                te = picks[n_on:]
                r = np.arange(len(te))
                Qt, Ct = Q_s[n_on:], C_s[n_on:]
                rows_out.append({"seed": seed, "w_C": w, "scenario": scen, "quality": float(Qt[r, te].mean()),
                                 "cost": float(Ct[r, te].mean()), "gpt4_share": float((te == 0).mean()),
                                 "seconds": dt, "offline_fit_seconds": fit_s,
                                 "gpt4_share_by_1000": [float((picks[i:i + 1000] == 0).mean())
                                                        for i in range(0, n_on, 1000)]})
                OUT.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(rows_out, indent=1))
                print(f"seed {seed} w {w} {scen}: quality {rows_out[-1]['quality']:.4f} cost "
                      f"{rows_out[-1]['cost'] * 1e3:.3f}m gpt4 {rows_out[-1]['gpt4_share']:.2f} ({dt / 60:.1f} min)",
                      flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
