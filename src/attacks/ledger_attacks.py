"""Stage 6: attacks on the evidence ledger (the contract and the 4-validator chain).

    infrastructure/besu/start_besu.sh
    python -m src.attacks.ledger_attacks

Runs against the deployed EvidenceLedger that holds the 1,250 live receipts (Stage 5) and checks
that no attack changes them. Writes results/stage6/ledger_attacks.json.

1. An outsider (an account that is not an approved router) adds a receipt.
2. An approved router adds a receipt for a model that was never registered.
3. An approved router sends one of its receipts again, also with a better score.
   For 1-3 the call is first simulated to read the contract's refusal, then really sent: the
   transaction is mined as failed and the receipts on the chain do not change.
4. Editing past receipts:
   a. the contract has no function that changes or deletes a receipt (receipts are event logs);
   b. an insider edits a local copy (half of Opus's correct answers turned into wrong ones, as in
      Fevzi Ege Yurtsevenler's E5 on the phase5 branch): comparing it with the chain finds every
      edited receipt;
   c. all 4 validators hold the same 1,250 receipts in the same blocks, so one altered copy would
      disagree with the other three (a block needs 3 of the 4 signatures).
Attack 5 (validators down) is the Stage 5 measurement; attack 8 (the owner approving fake routers
or models) is a limitation of the one-owner design and is not run.
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.attacks import ERRORS
from src.data.collect_live import MODELS
from src.data.tables import read_table
from src.evidence import groups
from src.evidence.ledger import CONTRACTS, DEPLOYMENT, RPC, Account, Ledger, dev_account, run
from src.evidence.upload_live import DATA, local_receipts
from src.routing.embeddings import cache_path, load_embeddings

ADD = "addReceipt(bytes32,uint8,bytes32,uint16,uint64,uint32)"
OUT = Path("results/stage6/ledger_attacks.json")
NODES = {f"validator {i}": f"http://127.0.0.1:{8544 + i}" for i in range(1, 5)}


def error_name(revert_data: str) -> str:
    """The contract error a revert carries (its first 4 bytes), e.g. NotRouter."""
    selector = revert_data[:10].lower()
    return next((name for name, sel in ERRORS.items() if sel == selector), f"unknown {selector}")


def simulate(ledger: Ledger, sender: str, signature: str, *args: str) -> str:
    """Run the call without sending it and return the contract's refusal ("accepted" if none)."""
    data = run("cast", "calldata", signature, *args)
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "eth_call",
                       "params": [{"from": sender, "to": ledger.address, "data": data}, "latest"]}).encode()
    with urllib.request.urlopen(urllib.request.Request(ledger.rpc, body, {"Content-Type": "application/json"}), timeout=60) as r:
        reply = json.loads(r.read())
    if "error" not in reply:
        return "accepted"
    return error_name(str(reply["error"].get("data") or ""))


def attempt(ledger: Ledger, name: str, attacker: Ledger, *args: str) -> dict:
    """One refused-receipt attack: the contract's reason, then a real send, then the receipt count."""
    before = len(ledger.receipts(_first_block()))
    reason = simulate(ledger, attacker.account.address, ADD, *args)
    try:
        attacker.send(ADD, *args)
        mined = "succeeded"
    except RuntimeError as error:
        mined = "failed (reverted)" if "failed in block" in str(error) else f"rejected: {str(error)[-120:]}"
    after = len(ledger.receipts(_first_block()))
    result = {"attack": name, "contract_says": reason, "transaction": mined,
              "receipts_before": before, "receipts_after": after, "chain_unchanged": before == after}
    print(f"  {name}: {reason}; transaction {mined}; receipts {before} -> {after}")
    return result


def _first_block() -> int:
    return int(json.loads(DEPLOYMENT.read_text())["block"]) - 5


def contract_functions() -> List[str]:
    """Every function the deployed contract offers (from the compiled ABI)."""
    abi = json.loads((CONTRACTS / "out" / "EvidenceLedger.sol" / "EvidenceLedger.json").read_text())["abi"]
    return sorted(f"{f['name']}({','.join(i['type'] for i in f['inputs'])})" for f in abi if f["type"] == "function")


def find_edits(local: pd.DataFrame, chain: pd.DataFrame, fields: Iterable[str] = ("group", "score", "cost_nano_usd", "latency_ms")) -> pd.DataFrame:
    """Receipts whose local copy differs from the chain (matched by question hash and model)."""
    merged = local.merge(chain, on=["question_hash", "model_id"], how="outer", suffixes=("_local", "_chain"), indicator=True)
    differs = merged["_merge"] != "both"
    for f in fields:
        differs |= merged[f"{f}_local"] != merged[f"{f}_chain"]
    return merged[differs]


def node_copies(ledger: Ledger) -> dict:
    """Each validator's copy of the receipts and of the blocks that hold them."""
    copies = {}
    for name, url in NODES.items():
        receipts = Ledger(ledger.address, ledger.account, url).receipts(_first_block())
        blocks = sorted({r.block for r in receipts})
        hashes = [run("cast", "block", str(b), "--field", "hash", "--rpc-url", url) for b in (blocks[0], blocks[-1])]
        copies[name] = {"receipts": len(receipts), "fingerprint": "0x" + hashlib.sha256(json.dumps(sorted(map(list, receipts))).encode()).hexdigest(),
                        "first_and_last_block_hash": hashes}
    return copies


def main(argv: Optional[Sequence[str]] = None) -> int:
    account = dev_account()
    ledger = Ledger.load(account)
    model_ids = {m: ledger.model_id(m) for m in MODELS}
    chain = pd.DataFrame(ledger.receipts(_first_block()))
    print(f"evidence contract {ledger.address}: {len(chain)} receipts\n\n1-3: refused receipts")
    first = chain[chain["score"] == 0].iloc[0]  # a wrong answer, so a "better score" really is better
    outsider_wallet = json.loads(run("cast", "wallet", "new", "--json"))["data"][0]
    outsider = Ledger(ledger.address, Account(outsider_wallet["address"], outsider_wallet["private_key"]))
    fake_model = run("cast", "keccak", "gpt-7-fake")
    good = lambda q, m, score: [q, str(int(first.group)), m, str(score), str(int(first.cost_nano_usd)), str(int(first.latency_ms))]
    attacks = [
        attempt(ledger, "1. outsider adds a receipt", outsider, *good(first.question_hash, model_ids["claude-opus-5-5"], 10_000)),
        attempt(ledger, "2. receipt for an unregistered model", ledger, *good("0x" + "ab" * 32, fake_model, 10_000)),
        attempt(ledger, "3. the same receipt again", ledger, *good(first.question_hash, first.model_id, int(first.score))),
        attempt(ledger, "3. the same receipt again, with a better score", ledger, *good(first.question_hash, first.model_id, 10_000)),
    ]

    print("\n4a. functions of the contract (none changes or deletes a receipt):")
    functions = contract_functions()
    print("  " + ", ".join(functions))

    print("\n4b. an insider edits a local copy: half of Opus's correct answers become wrong")
    table = read_table(DATA)
    X = load_embeddings(cache_path(DATA, "bert-base-multilingual-cased"), table["sample_id"])
    local = local_receipts(table, groups.assign(X, groups.load()), model_ids)
    edited = local.copy()
    opus_right = edited.index[(edited["model"] == "claude-opus-5-5") & (edited["score"] == 10_000)]
    hit = np.random.default_rng(0).choice(opus_right, size=len(opus_right) // 2, replace=False)
    edited.loc[hit, "score"] = 0
    found = find_edits(edited, chain)
    opus = lambda frame: frame.loc[frame["model_id"] == model_ids["claude-opus-5-5"], "score"].mean() / 10_000
    tamper = {"receipts_edited": int(len(hit)), "edits_found": int(len(found)),
              "all_found_and_only_those": set(found["question_hash"]) == set(edited.loc[hit, "question_hash"]) and len(found) == len(hit),
              "opus_accuracy_true": round(float(opus(chain)), 4), "opus_accuracy_in_edited_copy": round(float(opus(edited)), 4)}
    print(f"  edited {tamper['receipts_edited']}, found {tamper['edits_found']} by comparing with the chain "
          f"(all of them and only them: {tamper['all_found_and_only_those']}); Opus accuracy {tamper['opus_accuracy_true']:.1%} on the chain, "
          f"{tamper['opus_accuracy_in_edited_copy']:.1%} in the edited copy")

    print("\n4c. the 4 validators' copies")
    copies = node_copies(ledger)
    for name, c in copies.items():
        print(f"  {name}: {c['receipts']} receipts, fingerprint {c['fingerprint'][:14]}..., blocks {[h[:12] for h in c['first_and_last_block_hash']]}")
    identical = len({json.dumps(c, sort_keys=True) for c in copies.values()}) == 1

    results = {"contract": ledger.address, "receipts": len(chain), "refused_receipts": attacks,
               "contract_functions": functions, "edited_local_copy": tamper,
               "validator_copies": copies, "validator_copies_identical": identical}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nall 4 copies identical: {identical}\nwrote {OUT}")
    ok = all(a["chain_unchanged"] and a["contract_says"] != "accepted" for a in attacks) and tamper["all_found_and_only_those"] and identical
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
