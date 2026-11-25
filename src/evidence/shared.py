"""Stage 7: what a router learns from the receipts of the stream (choice C).

From the receipts it can see (alone: only its own; sharing: every router's), a router takes:

* c and l, the cost and latency of a typical model (the mean over models of each model's mean),
  from the shared warm start plus the stream receipts, exactly as router.Scales does locally;
* an accuracy correction per (group, model): the mean score of the visible stream receipts of model
  m in group k, against the router's own forests' mean guess for that group,

      correction[k, m] = n / (n + tau) * (mean score[k, m] - mean guess[k, m]),

  damped while there are few receipts (tau pseudo-answers with no correction, tau = 20 as in Fevzi
  Ege Yurtsevenler's evidence-corrected router). It needs no other router's question.

Receipts are kept in the contract's integer units (score in basis points, cost in nano-USD, latency
in ms), so the same numbers come from memory (Evidence) or from the chain (ChainEvidence), and the
simulation can be repeated on the chain with identical picks.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, NamedTuple, Optional, Sequence

import numpy as np

from src.routing.router import Scales


class StreamReceipt(NamedTuple):
    router: int
    question: int  # row index of the question
    group: int
    model: int
    score: int    # basis points
    cost: int     # nano-USD
    latency: int  # ms


def units(quality: float, cost: float, latency: float) -> tuple:
    """A model's real outcome in the contract's integer units."""
    return int(round(quality * 10_000)), int(round(cost * 1e9)), int(round(latency * 1e3))


class Totals(NamedTuple):
    n_group: np.ndarray       # [groups, models] receipts
    score_group: np.ndarray   # [groups, models] summed score (bp)
    n_model: np.ndarray       # [models]
    cost_model: np.ndarray    # [models] summed cost (nano-USD)
    latency_model: np.ndarray # [models] summed latency (ms)


def totals_of(receipts: Sequence[StreamReceipt], groups: int, models: int) -> Totals:
    t = Totals(np.zeros((groups, models)), np.zeros((groups, models)), np.zeros(models), np.zeros(models), np.zeros(models))
    for r in receipts:
        t.n_group[r.group, r.model] += 1
        t.score_group[r.group, r.model] += r.score
        t.n_model[r.model] += 1
        t.cost_model[r.model] += r.cost
        t.latency_model[r.model] += r.latency
    return t


class Evidence:
    """The stream receipts kept in memory (the simulation)."""

    def __init__(self, groups: int, models: int):
        self.groups, self.models = groups, models
        self.receipts: List[StreamReceipt] = []
        self.excluded: set = set()  # routers removed by the owner: all their receipts are ignored

    def add(self, receipts: Sequence[StreamReceipt]) -> None:
        self.receipts.extend(receipts)

    def exclude(self, router: int) -> None:
        """The owner removes a router: its past receipts are ignored and new ones are not accepted."""
        self.excluded.add(router)

    def visible(self, router: Optional[int]) -> List[StreamReceipt]:
        """Everyone's receipts (router None, sharing) or only one router's (alone), without removed routers."""
        kept = [r for r in self.receipts if r.router not in self.excluded]
        return kept if router is None else [r for r in kept if r.router == router]

    def totals(self, router: Optional[int]) -> Totals:
        return totals_of(self.visible(router), self.groups, self.models)


def scales(warm: Mapping[str, np.ndarray], t: Totals) -> Scales:
    """router.Scales from the shared warm start plus the visible stream receipts."""
    s = Scales(warm)
    s.answers = s.answers + t.n_model
    s.cost = s.cost + t.cost_model / 1e9
    s.latency = s.latency + t.latency_model / 1e3
    return s


def correction(t: Totals, mean_guess: np.ndarray, tau: float) -> np.ndarray:
    """Accuracy correction [groups, models] from the visible stream receipts (0 where there are none)."""
    n = t.n_group
    mean_score = np.divide(t.score_group / 10_000, n, out=np.zeros_like(n), where=n > 0)
    usable = (n > 0) & ~np.isnan(mean_guess)  # no correction for a group the forests have no guess for
    return np.where(usable, n / (n + tau) * (mean_score - np.nan_to_num(mean_guess)), 0.0)


def mean_guess(quality_guess: np.ndarray, groups_of_rows: np.ndarray, groups: int) -> np.ndarray:
    """The forests' mean accuracy guess per (group, model) over the given rows [groups, models];
    NaN for a group with none of the rows."""
    out = np.full((groups, quality_guess.shape[1]), np.nan)
    for k in range(groups):
        rows = groups_of_rows == k
        if rows.any():
            out[k] = quality_guess[rows].mean(axis=0)
    return out


class ChainEvidence(Evidence):
    """The stream receipts on a fresh EvidenceLedger on the chain.

    Each block's receipts are sent by the routers' own accounts (router 0 is the development account,
    the contract's owner; the others are new accounts it approves), and every read takes the receipts
    back from the chain. A memory copy is kept only to compare.
    """

    ADD = "addReceipt(bytes32,uint8,bytes32,uint16,uint64,uint32)"

    def __init__(self, groups: int, models: int, model_names: Sequence[str], prices: Mapping[str, tuple],
                 routers: int, question_hash: Mapping[int, str]):
        import json
        from src.evidence.ledger import RPC, Account, Ledger, dev_account, run

        super().__init__(groups, models)
        self._run, self._Ledger = run, Ledger
        owner = dev_account()
        self.ledger = Ledger.deploy(owner, record=None)
        self.first_block = int(run("cast", "block-number", "--rpc-url", RPC)) - 1
        for name in model_names:
            price_in, price_out = prices[name]
            self.ledger.send("registerModel(string,uint64,uint64)", name, str(int(round(price_in * 1e6))), str(int(round(price_out * 1e6))))
        self.model_ids = [self.ledger.model_id(name) for name in model_names]
        self.model_index = {mid.lower(): j for j, mid in enumerate(self.model_ids)}
        wallets = json.loads(run("cast", "wallet", "new", "--number", str(routers - 1), "--json"))["data"] if routers > 1 else []
        self.accounts = [owner] + [Account(w["address"], w["private_key"]) for w in wallets]
        for account in self.accounts:
            self.ledger.send("addRouter(address)", account.address)
        self.router_of = {a.address.lower(): i for i, a in enumerate(self.accounts)}
        self.question_hash = question_hash
        self._chain: Optional[List[StreamReceipt]] = None

    def add(self, receipts: Sequence[StreamReceipt]) -> None:
        import time

        super().add(receipts)
        for router in sorted({r.router for r in receipts}):
            sender = self._Ledger(self.ledger.address, self.accounts[router], self.ledger.rpc)
            mine = [r for r in receipts if r.router == router]
            for start in range(0, len(mine), 150):  # Besu keeps about 200 pending transactions per account
                chunk, nonce = mine[start:start + 150], sender.nonce()
                for i, r in enumerate(chunk):
                    sender.send(self.ADD, self.question_hash[r.question], str(r.group), self.model_ids[r.model], str(r.score),
                                str(r.cost), str(r.latency), nonce=nonce + i, wait=False)
                waited = time.time()
                while sender.nonce() < nonce + len(chunk):
                    if time.time() - waited > 300:
                        raise TimeoutError("receipts not mined in 5 minutes")
                    time.sleep(0.5)
        self._chain = None

    def exclude(self, router: int) -> None:
        """The owner removes the router on the chain (its new receipts are refused); its past ones are ignored."""
        super().exclude(router)
        self.ledger.send("removeRouter(address)", self.accounts[router].address)

    def chain_receipts(self) -> List[StreamReceipt]:
        if self._chain is None:
            self._chain = [StreamReceipt(self.router_of[r.router.lower()], -1, r.group, self.model_index[r.model_id.lower()],
                                         r.score, r.cost_nano_usd, r.latency_ms) for r in self.ledger.receipts(self.first_block)]
        return self._chain

    def visible(self, router: Optional[int]) -> List[StreamReceipt]:
        receipts = [r for r in self.chain_receipts() if r.router not in self.excluded]
        return receipts if router is None else [r for r in receipts if r.router == router]
