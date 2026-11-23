"""Stage 5: put the live answers on the chain as receipts, then read them back and check them.

    infrastructure/besu/start_besu.sh                  # the 4-validator chain
    python -m src.evidence.upload_live                 # deploy if needed, register, send, check
    python -m src.evidence.upload_live --check-only    # only read back and check

Every one of the 250 live SuperGPQA questions was answered by all 5 models, so there are 1,250
receipts: question hash (keccak256 of the question text), group (the nearest of the 8 centroids in
data/evidence/groups_k8.npz, made from these questions' BERT vectors on the first run), model,
score, cost and latency. The development account deploys the contract, approves itself as the
router, registers the 5 models with their official prices and sends the receipts one by one.
Receipts already on the chain are skipped, so the script can be run again after an interruption.

The check reads every receipt back from the chain and recomputes each model's accuracy, average cost
and latency (and per group) from the chain alone, against the same numbers from the local data.
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.data.collect_live import MODELS
from src.data.tables import read_table
from src.evidence import groups
from src.evidence.ledger import DEPLOYMENT, Ledger, dev_account, keccak, totals
from src.routing.embeddings import cache_path, load_embeddings

DATA = "data/processed/supergpqa_live.csv.gz"
CHUNK = 100  # receipts sent before waiting for them to be in a block


def local_receipts(table: pd.DataFrame, group: np.ndarray, model_ids: dict) -> pd.DataFrame:
    """The 1,250 receipts the local data gives (one per question and model), in the contract's units."""
    rows = []
    hashes = [keccak(text) for text in table["prompt"]]
    for i, (question_hash, g) in enumerate(zip(hashes, group)):
        for model in MODELS:
            rows.append(dict(question_hash=question_hash, group=int(g), model=model, model_id=model_ids[model],
                             score=int(round(table[f"{model}|quality"].iloc[i] * 10_000)),
                             cost_nano_usd=int(round(table[f"{model}|cost"].iloc[i] * 1e9)),
                             latency_ms=int(round(table[f"{model}|latency"].iloc[i] * 1e3))))
    return pd.DataFrame(rows)


def wait_for_nonce(ledger: Ledger, target: int, timeout: float = 120.0) -> None:
    start = time.time()
    while ledger.nonce() < target:
        if time.time() - start > timeout:
            raise TimeoutError(f"transactions not mined: nonce {ledger.nonce()} of {target}")
        time.sleep(1)


def check(ledger: Ledger, local: pd.DataFrame, from_block: int) -> bool:
    chain = ledger.receipts(from_block)
    names = dict(zip(local["model_id"], local["model"]))
    print(f"\nread back {len(chain)} receipts from the chain (contract {ledger.address})")
    key = ["question_hash", "model_id"]
    merged = local.merge(pd.DataFrame(chain), on=key, how="outer", suffixes=("_local", "_chain"), indicator=True)
    same = all((merged[f"{c}_local"] == merged[f"{c}_chain"]).all() for c in ("group", "score", "cost_nano_usd", "latency_ms"))
    print(f"every receipt equal to the local data: {same and (merged['_merge'] == 'both').all()} "
          f"(only local {int((merged['_merge'] == 'left_only').sum())}, only chain {int((merged['_merge'] == 'right_only').sum())})")
    per_model = totals(chain, names)
    expected = local.groupby("model").agg(accuracy=("score", lambda s: s.mean() / 10_000),
                                          cost=("cost_nano_usd", lambda s: s.mean() / 1e9),
                                          latency=("latency_ms", lambda s: s.mean() / 1e3))
    print("\nper model, recomputed from the chain alone (local data in brackets):")
    for model, row in per_model.iterrows():
        e = expected.loc[model]
        print(f"  {model:18s} {int(row.answers):4d} answers  accuracy {row.accuracy:6.1%} ({e.accuracy:6.1%})  "
              f"cost ${row.cost:.6f} (${e.cost:.6f})  latency {row.latency:5.2f} s ({e.latency:5.2f} s)")
    per_group = totals(chain, names, by_group=True)["accuracy"].unstack("model")[list(MODELS)]
    print("\naccuracy per (group, model), from the chain:")
    print((per_group * 100).round(0).astype(int).astype(str).add("%").to_string())
    return same


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Send the live answers to the ledger as receipts and check them.")
    parser.add_argument("--redeploy", action="store_true", help="deploy a new contract even if one exists")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)

    table = read_table(DATA)
    X = load_embeddings(cache_path(DATA, "bert-base-multilingual-cased"), table["sample_id"])
    if not groups.CENTROIDS.exists():
        groups.save(groups.CENTROIDS, groups.fit(X), source=DATA, seed=0)
        print(f"made the 8 groups from the {len(X)} live questions: {groups.CENTROIDS}")
    group = groups.assign(X, groups.load())
    print("questions per group:", np.bincount(group, minlength=8).tolist())

    account = dev_account()
    if args.redeploy or not DEPLOYMENT.exists():
        ledger = Ledger.deploy(account)
        print(f"deployed EvidenceLedger at {ledger.address}")
    else:
        ledger = Ledger.load(account)
    from_block = int(json.loads(DEPLOYMENT.read_text())["block"]) - 5  # receipts can only come after the deployment
    model_ids = {model: ledger.model_id(model) for model in MODELS}
    local = local_receipts(table, group, model_ids)
    if args.check_only:
        return 0 if check(ledger, local, from_block) else 1

    if not ledger.is_router(account.address):
        ledger.send("addRouter(address)", account.address)
    for model, (_, price_in, price_out) in MODELS.items():
        if not ledger.is_registered(model):
            ledger.send("registerModel(string,uint64,uint64)", model, str(int(round(price_in * 1e6))), str(int(round(price_out * 1e6))))
            print(f"registered {model} (${price_in} / ${price_out} per M tokens)")

    done = {(r.question_hash, r.model_id) for r in ledger.receipts(from_block)}
    todo = local[[(h, m) not in done for h, m in zip(local["question_hash"], local["model_id"])]]
    print(f"{len(done)} receipts already on the chain, sending {len(todo)}")
    started = time.time()
    for start in range(0, len(todo), CHUNK):
        nonce = ledger.nonce()
        for offset, r in enumerate(todo.iloc[start:start + CHUNK].itertuples()):
            ledger.send("addReceipt(bytes32,uint8,bytes32,uint16,uint64,uint32)", r.question_hash, str(r.group),
                        r.model_id, str(r.score), str(r.cost_nano_usd), str(r.latency_ms), nonce=nonce + offset, wait=False)
        wait_for_nonce(ledger, nonce + len(todo.iloc[start:start + CHUNK]))
        print(f"  {min(start + CHUNK, len(todo))} / {len(todo)} sent ({time.time() - started:.0f} s)", flush=True)
    return 0 if check(ledger, local, from_block) else 1


if __name__ == "__main__":
    raise SystemExit(main())
