"""Phase 5 router: frozen offline forests + an evidence layer that corrects them online.

Phase 4's online learning retrained the selected models' random forests every 1,000 queries on
offline + online rows. Offline rows dominate those fits, retraining is slow (about 2-4 s per forest),
and there is no exploration, so rarely chosen models never collect new evidence.

Here the forests stay frozen. Queries are grouped into K clusters of their BERT embedding (k-means
on offline queries; the router never sees the benchmark label). For every (cluster k, model m) the
evidence layer keeps the sum of observed-minus-guessed quality and cost of the answers it has seen,
and corrects the guesses with a shrunken mean:

    A~_m(q) = A^_m(q) + S^A_km / (n_km + tau)        C~_m(q) = C^_m(q) + S^C_km / (n_km + tau)

    score_m(q) = A~_m(q) - w_C * C~_m(q) / c  +  alpha * sqrt( ln(1 + N_k) / (1 + n_km) )

tau is the prior strength (pseudo-observations at zero correction), the last term an optional UCB
exploration bonus. An update costs O(1); the aggregates are exactly what an evidence ledger stores.

Reporters: every outcome record is submitted by one reporter. Aggregates are kept per reporter so
that a reporter's weight can be changed retroactively (attributable evidence on a ledger). With
audits, a random fraction of records is re-scored by a trusted evaluator; a reporter's reputation is
(agreements + 1) / (audits + 2) and a reporter below `min_reputation` gets weight 0.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class EvidenceConfig:
    tau: float = 20.0
    alpha: float = 0.0
    correct_cost: bool = True
    reporters: int = 20
    audit_rate: float = 0.0          # fraction of records re-scored by the trusted evaluator
    audit_tolerance: float = 0.25    # |reported - true| above this is a failed audit
    min_reputation: float = 0.5
    rule: str = "agreement"          # "agreement": exclude if (agree+1)/(audits+2) < min_reputation
                                     # "failures": exclude if failed >= min_failures and failed/audits >= max_fail_rate
    min_failures: int = 3
    max_fail_rate: float = 0.15
    attribution: str = "ledger"      # "ledger": reputation applies to all of a reporter's records;
                                     # "local": only the audited record itself is corrected
    block: int = 50                  # queries routed together before their feedback arrives


@dataclass
class EvidenceState:
    K: int
    M: int
    R: int
    n: np.ndarray = field(init=False)
    sa: np.ndarray = field(init=False)
    sc: np.ndarray = field(init=False)
    agree: np.ndarray = field(init=False)
    audits: np.ndarray = field(init=False)

    def __post_init__(self):
        self.n = np.zeros((self.R, self.K, self.M))
        self.sa = np.zeros((self.R, self.K, self.M))
        self.sc = np.zeros((self.R, self.K, self.M))
        self.agree = np.zeros(self.R)
        self.audits = np.zeros(self.R)

    def weights(self, cfg: EvidenceConfig) -> np.ndarray:
        if cfg.audit_rate <= 0 or cfg.attribution != "ledger":
            return np.ones(self.R)
        if cfg.rule == "failures":
            failed = self.audits - self.agree
            rate = failed / np.maximum(self.audits, 1.0)
            return np.where((failed >= cfg.min_failures) & (rate >= cfg.max_fail_rate), 0.0, 1.0)
        rep = (self.agree + 1.0) / (self.audits + 2.0)
        return np.where(rep >= cfg.min_reputation, 1.0, 0.0)

    def aggregate(self, cfg: EvidenceConfig):
        w = self.weights(cfg)[:, None, None]
        return (w * self.n).sum(0), (w * self.sa).sum(0), (w * self.sc).sum(0)


def route_stream(A_hat, C_hat, clusters, Q_true, C_true, w_C, cost_scale, cfg: EvidenceConfig, rng,
                 reported_quality=None, learn: Optional[np.ndarray] = None, state: EvidenceState = None,
                 K: int = None):
    """Route a stream in blocks (each pick before its feedback), learning from the picked model only.

    A_hat, C_hat: frozen forest guesses [rows, M]; clusters: [rows]; Q_true / C_true: the outcome the
    picked model really delivers (attack scenarios may change it); reported_quality(i, m, reporter)
    returns what the reporter submits (default: the truth); learn: bool [rows], False = route only.
    Returns picks [rows] and the final state.
    """
    rows, M = A_hat.shape
    K = K or int(clusters.max()) + 1
    st = state or EvidenceState(K, M, cfg.reporters)
    picks = np.zeros(rows, dtype=int)
    reporter = rng.integers(0, cfg.reporters, rows)
    audit = rng.random(rows) < cfg.audit_rate
    learn = np.ones(rows, bool) if learn is None else learn
    for s in range(0, rows, cfg.block):
        idx = np.arange(s, min(s + cfg.block, rows))
        n, sa, sc = st.aggregate(cfg)
        k = clusters[idx]
        nk = n[k]
        A = A_hat[idx] + sa[k] / (nk + cfg.tau)
        C = C_hat[idx] + (sc[k] / (nk + cfg.tau) if cfg.correct_cost else 0.0)
        score = A - w_C * C / cost_scale
        if cfg.alpha > 0:
            Nk = nk.sum(axis=1, keepdims=True)
            score = score + cfg.alpha * np.sqrt(np.log1p(Nk) / (1.0 + nk))
        p = score.argmax(axis=1)
        picks[idx] = p
        for i, m in zip(idx, p):
            if not learn[i]:
                continue
            r = reporter[i]
            q = Q_true[i, m] if reported_quality is None else reported_quality(i, m, r)
            if audit[i]:
                st.audits[r] += 1
                ok = abs(q - Q_true[i, m]) <= cfg.audit_tolerance
                st.agree[r] += ok
                if not ok and cfg.attribution == "local":
                    q = Q_true[i, m]          # only the audited record is fixed
            kk = clusters[i]
            st.n[r, kk, m] += 1
            st.sa[r, kk, m] += q - A_hat[i, m]
            st.sc[r, kk, m] += C_true[i, m] - C_hat[i, m]
    return picks, st
