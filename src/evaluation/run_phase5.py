"""Phase 5 experiments (improvement of the experimental results).

    python -m src.evaluation.run_phase5 offline        # E1: baselines, AIQ, per-benchmark (both pools)
    python -m src.evaluation.run_phase5 online         # E2: evidence-corrected online router (5-model pool)
    python -m src.evaluation.run_phase5 attacks        # E3-E5: substitution, fake ratings, edited records

Everything is written to results/phase5/*.json. Offline forests are cached per (pool, seed).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from src.evaluation import metrics as M
from src.evaluation.phase5_data import load_fold, seeds_of
from src.routing.evidence_router import EvidenceConfig, route_stream

OUT = Path("results/phase5")
GRID = {"p5": [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 10.0],
        "p11": [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]}
REF = 0  # gpt-4-1106-preview is the first model in both pools
LAMBDAS = np.concatenate([[0.0], np.geomspace(1e-3, 100, 120)])


def offline_picks(A, C, w, scale):
    return (A - w * C / scale).argmax(axis=1)


def rel(Q, C, picks):
    rows = np.arange(len(Q))
    return float(Q[rows, picks].sum() / Q[:, REF].sum()), float(C[rows, picks].sum() / C[:, REF].sum())


def dump(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=1, default=float), encoding="utf-8")
    print(f"wrote {OUT / name}.json")


# ---------------------------------------------------------------------------------------------- E1
def exp_offline(args):
    out = {}
    for pool in ("p5", "p11"):
        per_seed = []
        for seed in seeds_of(pool):
            f = load_fold(pool, seed)
            T = f.test
            singles = M.single_model_points(T.Q, T.C)
            c_min, c_max = float(singles[:, 0].min()), float(singles[:, 0].max())
            router_pts, rows = [], []
            for w in GRID[pool]:
                p = offline_picks(T.A_hat, T.C_hat, w, f.cost_scale)
                c, q = M.picks_point(T.Q, T.C, p)
                rq, rc = rel(T.Q, T.C, p)
                share = np.bincount(p, minlength=len(f.models)) / len(p)
                rows.append({"w_C": w, "quality": q, "cost": c, "quality_vs_gpt4": rq, "cost_vs_gpt4": rc,
                             "share": share.tolist()})
                router_pts.append((c, q))
            orc = M.oracle_choice(T.Q, T.C)
            oc, oq = M.picks_point(T.Q, T.C, orc)
            rc_, rq_ = M.random_router(T.Q, T.C, seed)
            o_curve = M.oracle_curve(T.Q, T.C, LAMBDAS)
            # per-benchmark at the balanced weight
            wb = 0.1 if pool == "p5" else 0.02
            pb = offline_picks(T.A_hat, T.C_hat, wb, f.cost_scale)
            bench = {}
            for b in sorted(set(T.task)):
                m = T.task == b
                qb, cb = rel(T.Q[m], T.C[m], pb[m])
                bench[b] = {"n": int(m.sum()), "quality_vs_gpt4": qb, "cost_vs_gpt4": cb,
                            "gpt4_share": float((pb[m] == REF).mean()),
                            "gpt4_cost_share": float(T.C[m, REF].sum() / T.C[:, REF].sum())}
            per_seed.append({
                "seed": seed, "models": f.models, "n_test": len(T.ids), "fit_seconds": f.fit_seconds,
                "singles": [{"model": mo, "cost": float(c), "quality": float(q),
                             "quality_vs_gpt4": float(q / singles[REF, 1]), "cost_vs_gpt4": float(c / singles[REF, 0])}
                            for mo, (c, q) in zip(f.models, singles)],
                "router": rows,
                "random": {"cost": rc_, "quality": rq_},
                "oracle": {"cost": oc, "quality": oq, "quality_vs_gpt4": oq / singles[REF, 1],
                           "cost_vs_gpt4": oc / singles[REF, 0]},
                "aiq": {"zero_router": M.aiq(singles, c_min, c_max), "router": M.aiq(router_pts, c_min, c_max),
                        "oracle": M.aiq([(oc, oq)], c_min, c_max),            # RouterBench / Stage 1 definition
                        "oracle_curve": M.aiq(o_curve, c_min, c_max)},       # oracle with a cost penalty sweep
                "per_benchmark_w": wb, "per_benchmark": bench,
            })
            a = per_seed[-1]["aiq"]
            print(f"{pool} seed {seed}: AIQ zero {a['zero_router']:.3f} router {a['router']:.3f} oracle {a['oracle']:.3f}")
        out[pool] = per_seed
    dump("e1_offline", out)


# ---------------------------------------------------------------------------------------------- E2
def clusters_for(f, K, seed):
    km = KMeans(K, n_init=3, random_state=seed).fit(f.offline.X.astype(np.float64))
    return km.predict(f.online.X.astype(np.float64)), km.predict(f.test.X.astype(np.float64))


def run_online(f, w, cfg, k_on, k_te, seed, Q_on=None, Q_te=None, rq_on=None, freeze_test=True):
    """Online stream then test. Returns (test picks, stream picks, state)."""
    rng = np.random.default_rng(seed * 1000 + 7)
    Q_on = f.online.Q if Q_on is None else Q_on
    Q_te = f.test.Q if Q_te is None else Q_te
    p_on, st = route_stream(f.online.A_hat, f.online.C_hat, k_on, Q_on, f.online.C, w, f.cost_scale, cfg, rng,
                            reported_quality=rq_on, K=int(max(k_on.max(), k_te.max())) + 1)
    learn = np.zeros(len(k_te), bool) if freeze_test else None
    p_te, st = route_stream(f.test.A_hat, f.test.C_hat, k_te, Q_te, f.test.C, w, f.cost_scale, cfg, rng,
                            learn=learn, state=st)
    return p_te, p_on, st


def exp_online(args):
    res = {"configs": [], "runs": []}
    variants = {
        "ecr_K32": EvidenceConfig(tau=20, alpha=0.0),
        "ecr_K32_ucb": EvidenceConfig(tau=20, alpha=0.02),
        "ecr_K8": EvidenceConfig(tau=20), "ecr_K128": EvidenceConfig(tau=20),
        "ecr_K32_tau5": EvidenceConfig(tau=5), "ecr_K32_tau100": EvidenceConfig(tau=100),
    }
    Ks = {"ecr_K8": 8, "ecr_K128": 128}
    for seed in seeds_of("p5"):
        f = load_fold("p5", seed)
        cl = {K: clusters_for(f, K, seed) for K in (8, 32, 128)}
        for w in GRID["p5"]:
            base = offline_picks(f.test.A_hat, f.test.C_hat, w, f.cost_scale)
            c0, q0 = M.picks_point(f.test.Q, f.test.C, base)
            row = {"seed": seed, "w_C": w, "offline": {"cost": c0, "quality": q0}}
            for name, cfg in variants.items():
                K = Ks.get(name, 32)
                t = time.time()
                p_te, p_on, _ = run_online(f, w, cfg, *cl[K], seed)
                dt = time.time() - t
                c, q = M.picks_point(f.test.Q, f.test.C, p_te)
                cs, qs = M.picks_point(f.online.Q, f.online.C, p_on)
                row[name] = {"cost": c, "quality": q, "stream_cost": cs, "stream_quality": qs,
                             "seconds": dt, "updates": len(p_on)}
            res["runs"].append(row)
            print(f"seed {seed} w {w:<5g} offline {q0:.4f}/{c0*1e3:.3f}m  ecr {row['ecr_K32']['quality']:.4f}/"
                  f"{row['ecr_K32']['cost']*1e3:.3f}m  ucb {row['ecr_K32_ucb']['quality']:.4f}")
    dump("e2_online", res)


# ---------------------------------------------------------------------------------------------- E3-E5
def substituted(Q, start_mask, sub_from=1):
    """GPT-4 (column 0) silently answered by GPT-3.5 (column 1) where start_mask is True; billed cost unchanged."""
    Q = Q.copy()
    Q[start_mask, REF] = Q[start_mask, sub_from]
    return Q


def merkle_root(leaves):
    level = list(leaves)
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [hashlib.sha256(level[i] + level[i + 1]).digest() for i in range(0, len(level), 2)]
    return level[0]


def record_leaf(i, m, q, c):
    return hashlib.sha256(f"{i}|{m}|{q:.6f}|{c:.9f}".encode()).digest()


def exp_attacks(args):
    W = [0.1, 0.3] if not args.w else args.w
    res = {"substitution": [], "poison": [], "tamper": []}
    cfg0 = EvidenceConfig(tau=20, alpha=0.0)
    for seed in seeds_of("p5"):
        f = load_fold("p5", seed)
        k_on, k_te = clusters_for(f, 32, seed)
        n_on = len(f.online.ids)
        for w in W:
            # ---- E3 silent model substitution from 25% of the online stream on (and during the test)
            start = np.zeros(n_on, bool); start[int(0.25 * n_on):] = True
            Q_on_s = substituted(f.online.Q, start)
            Q_te_s = substituted(f.test.Q, np.ones(len(f.test.ids), bool))
            base = offline_picks(f.test.A_hat, f.test.C_hat, w, f.cost_scale)
            honest_ecr, _, _ = run_online(f, w, cfg0, k_on, k_te, seed)
            honest_cont, _, _ = run_online(f, w, cfg0, k_on, k_te, seed, freeze_test=False)
            sub_ecr, sub_on, _ = run_online(f, w, cfg0, k_on, k_te, seed, Q_on=Q_on_s, Q_te=Q_te_s, freeze_test=False)
            row = {"seed": seed, "w_C": w}
            for name, Qx, p in (("offline_honest", f.test.Q, base), ("offline_sub", Q_te_s, base),
                                ("ecr_honest", f.test.Q, honest_cont), ("ecr_sub", Q_te_s, sub_ecr)):
                c, q = M.picks_point(Qx, f.test.C, p)
                row[name] = {"cost": c, "quality": q, "gpt4_share": float((p == REF).mean())}
            oc, oq = M.picks_point(Q_te_s, f.test.C, M.oracle_choice(Q_te_s, f.test.C))
            row["oracle_sub"] = {"cost": oc, "quality": oq}
            # GPT-4 share over the stream in windows of 1,000 queries (adaptation speed)
            row["ecr_sub_gpt4_share_by_1000"] = [float((sub_on[i:i + 1000] == REF).mean()) for i in range(0, n_on, 1000)]
            res["substitution"].append(row)

            # ---- E4 fake ratings: malicious reporters inflate Claude Instant (index 4) and smear GPT-4
            for rho in (0.1, 0.2, 0.3):
                mal = np.zeros(cfg0.reporters, bool); mal[: int(round(rho * cfg0.reporters))] = True
                def rq(i, m, r, Q=f.online.Q):
                    if mal[r] and m == 4:
                        return 1.0
                    if mal[r] and m == REF:
                        return 0.0
                    return Q[i, m]
                prow = {"seed": seed, "w_C": w, "rho": rho}
                for name, cfg in (("none", EvidenceConfig(tau=20)),
                                  ("audit_local", EvidenceConfig(tau=20, audit_rate=0.05, attribution="local")),
                                  ("audit_ledger", EvidenceConfig(tau=20, audit_rate=0.05, attribution="ledger"))):
                    p_te, _, st = run_online(f, w, cfg, k_on, k_te, seed, rq_on=rq)
                    c, q = M.picks_point(f.test.Q, f.test.C, p_te)
                    wts = st.weights(cfg)
                    prow[name] = {"cost": c, "quality": q, "instant_share": float((p_te == 4).mean()),
                                  "malicious_excluded": int(((wts == 0) & mal).sum()),
                                  "honest_excluded": int(((wts == 0) & ~mal).sum())}
                prow["ecr_clean"] = dict(zip(("cost", "quality"), M.picks_point(f.test.Q, f.test.C, honest_ecr)))
                res["poison"].append(prow)

            # ---- E5 edited records: an insider rewrites 50% of GPT-4's stored records to quality 0
            rng = np.random.default_rng(seed + 99)
            _, p_on, _ = run_online(f, w, cfg0, k_on, k_te, seed)
            recs = [(i, int(m), float(f.online.Q[i, m]), float(f.online.C[i, m])) for i, m in enumerate(p_on)]
            B = 256
            roots = [merkle_root([record_leaf(*r) for r in recs[s:s + B]]) for s in range(0, len(recs), B)]
            edited = list(recs)
            hit = [j for j, r in enumerate(recs) if r[1] == REF and rng.random() < 0.5]
            for j in hit:
                i, m, q, c = edited[j]
                edited[j] = (i, m, 0.0, c)
            bad = {s // B for s in range(0, len(edited), B)
                   if merkle_root([record_leaf(*r) for r in edited[s:s + B]]) != roots[s // B]}
            touched = {j // B for j in hit}
            # Rebuild the evidence from (a) the edited local store, (b) only the batches whose roots verify.
            def rebuild(records, keep):
                from src.routing.evidence_router import EvidenceState
                K = int(max(k_on.max(), k_te.max())) + 1
                st = EvidenceState(K, len(f.models), 1)
                for j, (i, m, q, c) in enumerate(records):
                    if keep(j):
                        st.n[0, k_on[i], m] += 1
                        st.sa[0, k_on[i], m] += q - f.online.A_hat[i, m]
                        st.sc[0, k_on[i], m] += c - f.online.C_hat[i, m]
                return st
            trow = {"seed": seed, "w_C": w, "records": len(recs), "edited": len(hit), "batches": len(roots),
                    "batches_touched": len(touched), "batches_detected": len(bad & touched),
                    "false_alarms": len(bad - touched)}
            cfg1 = EvidenceConfig(tau=20, reporters=1)
            for name, st in (("clean", rebuild(recs, lambda j: True)),
                             ("local_edited", rebuild(edited, lambda j: True)),
                             ("ledger_verified", rebuild(edited, lambda j: (j // B) not in bad))):
                p_te, _ = route_stream(f.test.A_hat, f.test.C_hat, k_te, f.test.Q, f.test.C, w, f.cost_scale, cfg1,
                                       np.random.default_rng(1), learn=np.zeros(len(k_te), bool), state=st)
                c, q = M.picks_point(f.test.Q, f.test.C, p_te)
                trow[name] = {"cost": c, "quality": q, "gpt4_share": float((p_te == REF).mean())}
            res["tamper"].append(trow)
            print(f"seed {seed} w {w}: sub offline {row['offline_sub']['quality']:.4f} ecr {row['ecr_sub']['quality']:.4f} | "
                  f"tamper local {trow['local_edited']['quality']:.4f} ledger {trow['ledger_verified']['quality']:.4f} "
                  f"(detected {trow['batches_detected']}/{trow['batches_touched']})")
    dump("e345_attacks", res)


def exp_poison(args):
    """E4 rerun: honest reporters err on 5% of records; two reputation rules (agreement fraction, failure count)."""
    W = args.w or [0.1, 0.3]
    res = []
    for seed in seeds_of("p5"):
        f = load_fold("p5", seed)
        k_on, k_te = clusters_for(f, 32, seed)
        n_on = len(f.online.ids)
        noise = np.random.default_rng(seed + 5).random(n_on) < 0.05
        for w in W:
            clean_te, _, _ = run_online(f, w, EvidenceConfig(tau=20), k_on, k_te, seed)
            cc, cq = M.picks_point(f.test.Q, f.test.C, clean_te)
            for rho in (0.1, 0.2, 0.3):
                R_ = 20
                mal = np.zeros(R_, bool); mal[: int(round(rho * R_))] = True
                def rq(i, m, r, Q=f.online.Q):
                    if mal[r] and m == 4:
                        return 1.0
                    if mal[r] and m == REF:
                        return 0.0
                    return 1.0 - Q[i, m] if noise[i] else Q[i, m]
                row = {"seed": seed, "w_C": w, "rho": rho, "clean": {"quality": cq, "cost": cc}}
                for name, cfg in (("none", EvidenceConfig(tau=20)),
                                  ("audit_local", EvidenceConfig(tau=20, audit_rate=0.05, attribution="local")),
                                  ("ledger_agreement", EvidenceConfig(tau=20, audit_rate=0.05, rule="agreement")),
                                  ("ledger_failures", EvidenceConfig(tau=20, audit_rate=0.05, rule="failures"))):
                    p_te, _, st = run_online(f, w, cfg, k_on, k_te, seed, rq_on=rq)
                    c, q = M.picks_point(f.test.Q, f.test.C, p_te)
                    wts = st.weights(cfg)
                    fail = st.audits - st.agree
                    row[name] = {"cost": c, "quality": q, "instant_share": float((p_te == 4).mean()),
                                 "malicious_excluded": int(((wts == 0) & mal).sum()),
                                 "honest_excluded": int(((wts == 0) & ~mal).sum()),
                                 "fail_rate_malicious": float(fail[mal].sum() / max(1, st.audits[mal].sum())),
                                 "fail_rate_honest": float(fail[~mal].sum() / max(1, st.audits[~mal].sum()))}
                res.append(row)
                print(f"seed {seed} w {w} rho {rho}: none {row['none']['quality']:.4f} local {row['audit_local']['quality']:.4f} "
                      f"agree {row['ledger_agreement']['quality']:.4f} fail {row['ledger_failures']['quality']:.4f} "
                      f"(excl mal {row['ledger_failures']['malicious_excluded']} hon {row['ledger_failures']['honest_excluded']}; "
                      f"fail rates {row['ledger_failures']['fail_rate_malicious']:.2f}/{row['ledger_failures']['fail_rate_honest']:.2f})",
                      flush=True)
    dump("e4_poison", res)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=["offline", "online", "attacks", "poison"])
    ap.add_argument("--w", type=float, nargs="+", default=None)
    args = ap.parse_args(argv)
    {"offline": exp_offline, "online": exp_online, "attacks": exp_attacks, "poison": exp_poison}[args.exp](args)


if __name__ == "__main__":
    main()
