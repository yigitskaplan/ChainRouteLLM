"""Stage 7: the router on the chain.

    python -m src.routing.run_stage7 --experiment 1            # live data: one router learning from the chain
    python -m src.routing.run_stage7 --experiment 1 --chain    # one run of it repeated on the real chain
    python -m src.routing.run_stage7 --experiment 2            # LLMRouterBench: 5 routers, alone vs sharing
    python -m src.routing.run_stage7 --experiment 2 --chain    # one sharing run repeated on the real chain
    python -m src.routing.run_stage7 --experiment 3            # live data: a lying router, a secret model swap
    python -m src.routing.run_stage7 --experiment 3 --chain    # one run of each attack on the real chain

Every router keeps its own forests (they retrain on its own answers, block by block, as in Stage 3)
and takes from the receipts it can see (src/evidence/shared.py): c and l, and an accuracy correction
per (group, model). Alone, a router sees only its own receipts; sharing, it sees every router's.
All routers start from the same warm start; each stream question goes to one router (seeded).

Experiment 1 (configs/experiment_stage3.json data, 5 seeds): one router for each of 5 settings,
learning from its receipts, against the same router without the correction and the Stage 3 router.
Experiment 2 (LLMRouterBench, 3 seeds, no latency): 5 routers, alone against sharing.
Experiment 3 (live data, 5 seeds, the 5 routers sharing): attack 6, one router lies in all its
receipts (Haiku always right, Opus always wrong), against no attack and against spot-check audits
that remove the liar (Fevzi's E4); the liar also makes up receipts (Haiku right, Opus wrong) for every question it
handles, without calling those models; attack 7, from a quarter of the stream on a model quietly
returns a weaker model's answers at its own price (Sol -> Terra, and Opus -> Haiku), with the
routers sharing or alone, each against the same setting without the swap (Fevzi's E3).
Learning is simulated in memory; --chain repeats one run with the receipts on the real chain and
checks that every pick is the same. Results go to results/stage7/.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence

import numpy as np
import pandas as pd

from src.data.collect_live import MODELS as LIVE_MODELS
from src.data.tables import read_table, table_stem
from src.evidence import groups as groups_
from src.evidence.shared import ChainEvidence, Evidence, StreamReceipt, correction, mean_guess, scales, units
from src.routing.embeddings import cache_path, load_embeddings
from src.routing.predictors import Predictors
from src.routing.router import Router
from src.routing.run_live import model_reputation

CONFIG = Path("configs/experiment_stage7.json")
OUT = Path("results/stage7")


class Data(NamedTuple):
    X: np.ndarray
    real: Dict[str, np.ndarray]      # quality, cost (and latency on live data), each [rows, models]
    models: List[str]
    group: np.ndarray                # [rows]
    reputation: np.ndarray           # [models]
    warm: np.ndarray                 # row indices
    stream: np.ndarray               # row indices in arrival order
    dataset: Optional[np.ndarray]    # [rows] benchmark of each question (LLMRouterBench)
    text: pd.Series                  # question texts (for the question hash on the chain)


def _split(split_path: Path, sample_ids: pd.Series, warm_names, stream_names):
    split = read_table(split_path).set_index("sample_id")
    position = pd.Series(np.arange(len(sample_ids)), index=sample_ids)
    rows = lambda name: position.loc[split[split["split"] == name].sort_values("order").index].to_numpy()
    return np.concatenate([rows(n) for n in warm_names]), np.concatenate([rows(n) for n in stream_names])


def live_data(cfg: Mapping[str, Any], seed: int) -> Data:
    live = cfg["live"]
    c3 = json.loads(Path(live["config"]).read_text())
    models = c3["data"]["model_keys_primary"]
    table = read_table(live["data"])
    X = load_embeddings(cache_path(live["data"], c3["router"]["encoder_checkpoint"]), table["sample_id"])
    real = {k: table[[f"{m}|{k}" for m in models]].to_numpy(float) for k in ("quality", "cost", "latency")}
    split = Path(live["splits_dir"]) / f"{table_stem(live['data'])}_seed{seed}.csv.gz"
    warm, stream = _split(split, table["sample_id"], ("offline_fit", "offline_val"), ("online", "test"))
    return Data(X, real, models, groups_.assign(X, groups_.load()), model_reputation(c3, models), warm, stream, None, table["prompt"])


def llmrouterbench_data(cfg: Mapping[str, Any], seed: int) -> Data:
    c4 = json.loads(Path(cfg["llmrouterbench"]["config"]).read_text())
    models, path = c4["data"]["model_keys_primary"], c4["data"]["path"]
    table = read_table(path, columns=["sample_id", "dataset", "prompt"] + [f"{m}|{k}" for m in models for k in ("quality", "cost")])
    X = load_embeddings(cache_path(path, c4["router"]["encoder_checkpoint"]), table["sample_id"])
    real = {k: table[[f"{m}|{k}" for m in models]].to_numpy(float) for k in ("quality", "cost")}
    split = Path(c4["data"]["splits_dir"]) / f"{table_stem(path)}_seed{seed}.csv.gz"
    warm, stream = _split(split, table["sample_id"], ("offline_fit",), ("online",))
    centroids = groups_.fit(X[warm], k=int(cfg["groups"]), seed=seed)  # groups from this seed's warm start
    return Data(X, real, models, groups_.assign(X, centroids), np.zeros(len(models)), warm, stream,
                table["dataset"].to_numpy(), table["prompt"])


class SimRouter:
    """One router: its own copy of the forests, its own answers, its own mean guesses per group."""

    def __init__(self, index: int, setting: Mapping[str, float], base: Predictors, warm_guess: np.ndarray):
        self.index, self.setting = index, setting
        self.forests = copy.copy(base)
        self.forests.quality_, self.forests.cost_ = list(base.quality_), list(base.cost_)
        self.forests.latency_ = list(base.latency_)
        self.guess = warm_guess.copy()
        self.answered: Dict[int, List[int]] = {}  # model -> stream rows it answered


class Attack(NamedTuple):
    """A lying router (experiment 3): its receipts report ``lie(model, true score)`` instead of the truth,
    and for every question it handles it also makes up receipts for models it never called
    (``fabricate``), with a typical cost and latency, which the contract cannot tell apart.

    With ``audit_rate`` > 0 a trusted checker re-checks that share of the accepted receipts after each
    block; a router failing ``max_failures`` checks is removed by the owner and all its receipts are
    ignored (Fevzi Ege Yurtsevenler's "failures" rule, E4 on the phase5 branch).
    """
    liar: int
    lie: Any
    audit_rate: float = 0.0
    max_failures: int = 3
    fabricate: tuple = ()  # (model, score) receipts the liar also makes up for every question it handles
    typical: Any = None    # model -> (cost, latency) in contract units, used for made-up receipts


def run(data: Data, settings: Sequence[Mapping[str, float]], assign: np.ndarray, share: bool, block: int,
        tau: Optional[float], base: Predictors, groups: int, evidence: Evidence, forest: Mapping[str, Any], seed: int,
        attack: Optional[Attack] = None) -> tuple:
    """Route the stream; return the model picked for each stream question (stream order) and a dict of
    what happened (the block in which a liar was removed, if any)."""
    keys = [k for k in ("quality", "cost", "latency") if k in data.real]
    warm_real = {k: data.real[k][data.warm] for k in keys}
    warm_guess = mean_guess(base.predict(data.X[data.warm])["quality"], data.group[data.warm], groups)
    routers = [SimRouter(i, s, base, warm_guess) for i, s in enumerate(settings)]
    picks = np.full(len(data.stream), -1)
    audits = np.random.default_rng(seed + 7)
    failures: Dict[int, int] = {}
    info: Dict[str, Any] = {"removed_in_block": None}
    for number, start in enumerate(range(0, len(data.stream), block)):
        positions = np.arange(start, min(start + block, len(data.stream)))
        receipts, used = [], {}
        for r in routers:  # every pick of the block is made before any of its answers is seen
            mine = positions[assign[positions] == r.index]
            if not len(mine):
                continue
            rows = data.stream[mine]
            t = evidence.totals(None if share else r.index)
            guess = r.forests.predict(data.X[rows])
            if tau is not None:
                guess["quality"] = guess["quality"] + correction(t, r.guess, tau)[data.group[rows]]
            s = r.setting
            router = Router(s["w_C"], scales(warm_real, t), s.get("w_L", 0.0), s.get("w_R", 0.0), data.reputation)
            chosen = router.route(guess)
            picks[mine] = chosen
            used[r.index] = set(int(m) for m in chosen)
            for row, m in zip(rows, chosen):
                score, cost, latency = units(data.real["quality"][row, m], data.real["cost"][row, m],
                                             data.real["latency"][row, m] if "latency" in data.real else 0.0)
                if attack is not None and r.index == attack.liar:
                    score = int(attack.lie(int(m), score))
                if r.index not in evidence.excluded:  # a removed router's receipts are refused
                    receipts.append(StreamReceipt(r.index, int(row), int(data.group[row]), int(m), score, cost, latency))
                    if attack is not None and r.index == attack.liar:
                        for fake, fake_score in attack.fabricate:
                            if fake != int(m):
                                receipts.append(StreamReceipt(r.index, int(row), int(data.group[row]), fake, int(fake_score), *attack.typical[fake]))
                r.answered.setdefault(int(m), []).append(int(row))
        evidence.add(receipts)  # the answers come back: receipts go to the evidence
        if attack is not None and attack.audit_rate > 0:  # spot checks against the real answers
            for rec in receipts:
                if audits.random() < attack.audit_rate and rec.score != units(data.real["quality"][rec.question, rec.model], 0, 0)[0]:
                    failures[rec.router] = failures.get(rec.router, 0) + 1
                    if failures[rec.router] >= attack.max_failures and rec.router not in evidence.excluded:
                        evidence.exclude(rec.router)
                        info["removed_in_block"] = number + 1
        for r in routers:  # each router retrains, on its own (true) answers, the models it used in this block
            for m in used.get(r.index, ()):
                rows = np.asarray(r.answered[m])
                fitted = Predictors([data.models[m]], forest, seed).fit(
                    np.vstack([data.X[data.warm], data.X[rows]]),
                    {k: np.concatenate([data.real[k][data.warm, m], data.real[k][rows, m]])[:, None] for k in keys})
                r.forests.quality_[m], r.forests.cost_[m] = fitted.quality_[0], fitted.cost_[0]
                if fitted.latency_:
                    r.forests.latency_[m] = fitted.latency_[0]
                r.guess[:, m] = mean_guess(fitted.quality_[0].predict(data.X[data.warm])[:, None], data.group[data.warm], groups)[:, 0]
    info["failed_audits"] = failures
    return picks, info


def measure(data: Data, picks: np.ndarray, assign: np.ndarray, routers: int) -> Dict[str, float]:
    rows = data.stream
    q, c = data.real["quality"][rows, picks], data.real["cost"][rows, picks]
    out = {"quality": q.mean(), "cost_per_1k": 1000 * c.mean()}
    if "latency" in data.real:
        out["latency"] = data.real["latency"][rows, picks].mean()
    if data.dataset is not None:  # LLMRouterBench's own metrics: mean over datasets, total cost
        d = data.dataset[rows]
        out["avg_acc"] = float(np.mean([100 * q[d == name].mean() for name in pd.unique(d)]))
        out["total_cost"] = c.sum()
    for i in range(routers):
        mine = assign == i
        if mine.any():
            out[f"quality_router{i}"], out[f"cost_per_1k_router{i}"] = q[mine].mean(), 1000 * c[mine].mean()
    return out


def experiment_1(cfg: Mapping[str, Any], chain: bool) -> None:
    live = cfg["live"]
    c3 = json.loads(Path(live["config"]).read_text())
    forest, tau, groups, block = c3["router"]["predictors"]["random_forest"], float(cfg["tau"]), int(cfg["groups"]), int(live["block"])
    stage3 = pd.read_csv("results/stage3/live_30_70.csv")
    seeds = [int(live["chain_seed"])] if chain else [int(s) for s in c3["seeds"]]
    rows, checks = [], []
    for seed in seeds:
        data = live_data(cfg, seed)
        base = Predictors(data.models, forest, seed).fit(data.X[data.warm], {k: data.real[k][data.warm] for k in data.real})
        one = np.zeros(len(data.stream), dtype=int)
        for setting in live["routers"]:
            if chain and setting["name"] != live["chain_router"]:
                continue
            k = len(data.models)
            memory, _ = run(data, [setting], one, True, block, tau, base, groups, Evidence(groups, k), forest, seed)
            if chain:
                hashes = {int(r): data.text.iloc[r] for r in data.stream}
                from src.evidence.ledger import keccak
                hashes = {r: keccak(t) for r, t in hashes.items()}
                ev = ChainEvidence(groups, k, data.models, {m: LIVE_MODELS[m][1:] for m in data.models}, 1, hashes)
                on_chain, _ = run(data, [setting], one, True, block, tau, base, groups, ev, forest, seed)
                checks.append({"seed": seed, "router": setting["name"], "contract": ev.ledger.address, "receipts_on_chain": len(ev.chain_receipts()),
                               "picks_identical": bool((memory == on_chain).all()), "memory": measure(data, memory, one, 1), "chain": measure(data, on_chain, one, 1)})
                print(f"seed {seed} {setting['name']}: {len(ev.chain_receipts())} receipts on the chain, picks identical: {checks[-1]['picks_identical']}")
                continue
            plain, _ = run(data, [setting], one, True, block, None, base, groups, Evidence(groups, k), forest, seed)
            ref = stage3[(stage3.method == "router") & (stage3.seed == seed) & np.isclose(stage3.w_C, setting["w_C"])
                         & np.isclose(stage3.w_L, setting["w_L"]) & np.isclose(stage3.w_R, setting["w_R"])].iloc[0]
            for method, picks in (("router learning from its receipts (C)", memory), ("same router, no correction", plain)):
                rows.append(dict(seed=seed, router=setting["name"], method=method, **measure(data, picks, one, 1)))
            rows.append(dict(seed=seed, router=setting["name"], method="Stage 3 router (results/stage3)", quality=ref.quality,
                             cost_per_1k=1000 * ref.cost, latency=ref.latency))
            print(f"seed {seed} {setting['name']}: C {rows[-3]['quality']:.3f}  no correction {rows[-2]['quality']:.3f}  Stage 3 {ref.quality:.3f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if chain:
        (OUT / "exp1_chain_check.json").write_text(json.dumps(checks, indent=2) + "\n")
        print(f"wrote {OUT / 'exp1_chain_check.json'}")
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "exp1_live.csv", index=False)
    print(frame.groupby(["router", "method"], sort=False)[["quality", "cost_per_1k", "latency"]].mean().round(4).to_string())
    print(f"wrote {OUT / 'exp1_live.csv'}")


def experiment_2(cfg: Mapping[str, Any], chain: bool) -> None:
    lrb = cfg["llmrouterbench"]
    c4 = json.loads(Path(lrb["config"]).read_text())
    forest, tau, groups, block = c4["router"]["predictors"]["random_forest"], float(cfg["tau"]), int(cfg["groups"]), int(lrb["block"])
    settings = lrb["routers"]
    seeds = [int(lrb["chain_seed"])] if chain else [int(s) for s in lrb["seeds"]]
    rows, checks = [], []
    for seed in seeds:
        data = llmrouterbench_data(cfg, seed)
        k = len(data.models)
        base = Predictors(data.models, forest, seed).fit(data.X[data.warm], {key: data.real[key][data.warm] for key in data.real})
        assign = np.random.default_rng(seed).integers(0, len(settings), len(data.stream))
        for share in ((True,) if chain else (False, True)):
            picks, _ = run(data, settings, assign, share, block, tau, base, groups, Evidence(groups, k), forest, seed)
            if chain:
                from src.evidence.ledger import keccak
                hashes = {int(r): keccak(data.text.iloc[r]) for r in data.stream}
                prices = {m: tuple(lrb["prices"][m]) for m in data.models}
                ev = ChainEvidence(groups, k, data.models, prices, len(settings), hashes)
                on_chain, _ = run(data, settings, assign, share, block, tau, base, groups, ev, forest, seed)
                checks.append({"seed": seed, "sharing": share, "contract": ev.ledger.address, "receipts_on_chain": len(ev.chain_receipts()),
                               "picks_identical": bool((picks == on_chain).all()), "memory": measure(data, picks, assign, len(settings)),
                               "chain": measure(data, on_chain, assign, len(settings))})
                print(f"seed {seed} sharing: {len(ev.chain_receipts())} receipts on the chain, picks identical: {checks[-1]['picks_identical']}")
                continue
            rows.append(dict(seed=seed, mode="sharing" if share else "alone", **measure(data, picks, assign, len(settings))))
            print(f"seed {seed} {rows[-1]['mode']}: quality {rows[-1]['quality']:.4f}, AvgAcc {rows[-1]['avg_acc']:.2f}, cost ${rows[-1]['total_cost']:.2f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if chain:
        (OUT / "exp2_chain_check.json").write_text(json.dumps(checks, indent=2) + "\n")
        print(f"wrote {OUT / 'exp2_chain_check.json'}")
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "exp2_llmrouterbench.csv", index=False)
    print(frame.groupby("mode", sort=False).mean(numeric_only=True).drop(columns="seed").round(4).T.to_string())
    print(f"wrote {OUT / 'exp2_llmrouterbench.csv'}")


def swapped(data: Data, strong: str, weak: str, from_fraction: float) -> Data:
    """Attack 7: from ``from_fraction`` of the stream on, the strong model quietly returns the weak
    model's answers (so its quality is the weak model's), still billed and timed as itself."""
    quality = data.real["quality"].copy()
    later = data.stream[int(from_fraction * len(data.stream)):]
    quality[later, data.models.index(strong)] = quality[later, data.models.index(weak)]
    return data._replace(real={**data.real, "quality": quality})


def measure_routers(data: Data, picks: np.ndarray, mask: np.ndarray, block: int, watch: str = "claude-opus-5-5") -> Dict[str, Any]:
    """Quality, cost, latency and model shares of the picks of the masked stream questions, and the
    share of the ``watch`` model in every block."""
    rows, chosen = data.stream[mask], picks[mask]
    out = {"quality": data.real["quality"][rows, chosen].mean(), "cost_per_1k": 1000 * data.real["cost"][rows, chosen].mean(),
           "latency": data.real["latency"][rows, chosen].mean()}
    out.update({f"share_{m}": float((chosen == j).mean()) for j, m in enumerate(data.models)})
    watched = data.models.index(watch)
    blocks = np.arange(len(data.stream)) // block
    out["watched_model"] = watch
    out["watched_share_by_block"] = [round(float((picks[mask & (blocks == b)] == watched).mean()), 3) for b in range(blocks.max() + 1)]
    return out


def experiment_3(cfg: Mapping[str, Any], chain: bool) -> None:
    """Attacks 6 and 7 on the live data with the 5 routers sharing through the chain."""
    live = cfg["live"]
    c3 = json.loads(Path(live["config"]).read_text())
    forest, tau, groups, block = c3["router"]["predictors"]["random_forest"], float(cfg["tau"]), int(cfg["groups"]), int(live["block"])
    settings, a = live["routers"], cfg["attacks"]
    seeds = [int(live["chain_seed"])] if chain else [int(s) for s in c3["seeds"]]
    rows, checks = [], []
    for seed in seeds:
        data = live_data(cfg, seed)
        k = len(data.models)
        base = Predictors(data.models, forest, seed).fit(data.X[data.warm], {key: data.real[key][data.warm] for key in data.real})
        assign = np.random.default_rng(seed).integers(0, len(settings), len(data.stream))
        liar = [s["name"] for s in settings].index(a["liar"])
        honest = assign != liar
        weak, strong = data.models.index(a["lie_up"]), data.models.index(a["lie_down"])
        lie = lambda m, score: 10_000 if m == weak else 0 if m == strong else score
        typical = {j: units(0, data.real["cost"][data.warm, j].mean(), data.real["latency"][data.warm, j].mean())[1:] for j in range(k)}
        fabricate = tuple((data.models.index(name), int(round(score * 10_000))) for name, score in a["fabricate"])
        liar_attack = lambda audit: Attack(liar, lie, audit, int(a["max_failures"]), fabricate, typical)
        everyone = np.ones(len(data.stream), dtype=bool)
        scenarios = [
            ("6. lying router", "no attack", data, True, None, honest, "claude-opus-5-5"),
            ("6. lying router", "attack, no defence", data, True, liar_attack(0.0), honest, "claude-opus-5-5"),
            ("6. lying router", "attack, audits remove the liar", data, True, liar_attack(float(a["audit_rate"])), honest, "claude-opus-5-5"),
            ("7. secret model swap", "no swap, sharing", data, True, None, everyone, "gpt-6.1-sol"),
            ("7. secret model swap", "no swap, each router alone", data, False, None, everyone, "gpt-6.1-sol"),
        ]
        for swap_cfg in a["swaps"]:
            d = swapped(data, swap_cfg["strong"], swap_cfg["weak"], float(a["swap_from"]))
            label = f"{swap_cfg['strong']} -> {swap_cfg['weak']}"
            scenarios += [("7. secret model swap", f"swap {label}, sharing", d, True, None, everyone, swap_cfg["strong"]),
                          ("7. secret model swap", f"swap {label}, each router alone", d, False, None, everyone, swap_cfg["strong"])]
        for attack_name, scenario, d, share, attack, mask, watch in scenarios:
            if chain and scenario not in ("attack, audits remove the liar", f"swap {a['swaps'][0]['strong']} -> {a['swaps'][0]['weak']}, sharing"):
                continue
            picks, info = run(d, settings, assign, share, block, tau, base, groups, Evidence(groups, k), forest, seed, attack)
            if chain:
                from src.evidence.ledger import keccak
                hashes = {int(r): keccak(d.text.iloc[r]) for r in d.stream}
                ev = ChainEvidence(groups, k, d.models, {m: LIVE_MODELS[m][1:] for m in d.models}, len(settings), hashes)
                on_chain, chain_info = run(d, settings, assign, share, block, tau, base, groups, ev, forest, seed, attack)
                checks.append({"seed": seed, "attack": attack_name, "scenario": scenario, "contract": ev.ledger.address,
                               "receipts_on_chain": len(ev.chain_receipts()), "picks_identical": bool((picks == on_chain).all()),
                               "removed_in_block": [info["removed_in_block"], chain_info["removed_in_block"]]})
                print(f"seed {seed} {scenario}: {len(ev.chain_receipts())} receipts on the chain, picks identical: {checks[-1]['picks_identical']}")
                continue
            m = measure_routers(d, picks, mask, block, watch)
            rows.append(dict(seed=seed, attack=attack_name, scenario=scenario, removed_in_block=info["removed_in_block"],
                             failed_audits=json.dumps(info["failed_audits"]), **m))
            print(f"seed {seed} {scenario:52s} quality {m['quality']:.3f}  cost ${m['cost_per_1k']:.2f}  Haiku {m['share_claude-haiku-5-5']:.0%}  "
                  f"Opus {m['share_claude-opus-5-5']:.0%}  {watch} {m[f'share_{watch}']:.0%}  removed in block {info['removed_in_block']}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if chain:
        (OUT / "exp3_chain_check.json").write_text(json.dumps(checks, indent=2) + "\n")
        print(f"wrote {OUT / 'exp3_chain_check.json'}")
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "exp3_attacks.csv", index=False)
    cols = ["quality", "cost_per_1k", "latency", "share_claude-haiku-5-5", "share_claude-opus-5-5", "share_gpt-6.1-sol"]
    print(frame.groupby(["attack", "scenario"], sort=False)[cols].mean().round(4).to_string())
    print(f"wrote {OUT / 'exp3_attacks.csv'}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 7: routers learning from the receipts on the chain.")
    parser.add_argument("--experiment", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--chain", action="store_true", help="repeat one run with the receipts on the real chain")
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args(argv)
    cfg = json.loads(args.config.read_text())
    {1: experiment_1, 2: experiment_2, 3: experiment_3}[args.experiment](cfg, args.chain)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
