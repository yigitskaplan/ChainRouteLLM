"""Phase 5, Stage 4: measure the evidence ledger on a real node.

    python contracts/measure_ledger.py --rpc http://127.0.0.1:8560 --label besu-qbft   # Hyperledger Besu (QBFT)
    python contracts/measure_ledger.py --rpc http://127.0.0.1:8547 --label anvil        # reference EVM

Deploys EvidenceLedger (contracts/src) and measures, from real transaction receipts:
  * gas per routing record in three storage modes: contract storage, event log, Merkle batch root;
  * inclusion latency (submit -> receipt) and sustained records per second for a burst;
  * Merkle inclusion proofs: gas of verify() for batch sizes 16..1024, a valid proof is accepted and a
    proof for an edited record is rejected.
The key used is a throw-away key of the local development chain (infrastructure/besu/dev_wallet.json,
or anvil's public default account). Writes results/phase5/ledger_<label>.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

from eth_account import Account
from web3 import Web3

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "contracts/out/EvidenceLedger.sol/EvidenceLedger.json"
ANVIL_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"  # anvil default account 0


def sha(a, b):
    return hashlib.sha256(a + b).digest()


def tree(leaves):
    levels = [list(leaves)]
    while len(levels[-1]) > 1:
        lv = levels[-1]
        levels.append([sha(lv[i], lv[i + 1]) for i in range(0, len(lv), 2)])
    return levels


def proof(levels, idx):
    out = []
    for lv in levels[:-1]:
        out.append(lv[idx ^ 1])
        idx >>= 1
    return out


def leaf(i, model, q, c):
    return hashlib.sha256(f"{i}|{model}|{q}|{c}".encode()).digest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rpc", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--burst", type=int, default=200)
    args = ap.parse_args()
    w3 = Web3(Web3.HTTPProvider(args.rpc, request_kwargs={"timeout": 30}))
    try:  # QBFT blocks carry a long extraData (validators, seals)
        from web3.middleware import ExtraDataToPOAMiddleware
        w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
    except ImportError:
        from web3.middleware import geth_poa_middleware
        w3.middleware_onion.inject(geth_poa_middleware, layer=0)
    if "anvil" in args.label:
        key = ANVIL_KEY
    else:
        key = json.loads((ROOT / "infrastructure/besu/dev_wallet.json").read_text())["data"][0]["private_key"]
    acct = Account.from_key(key)
    chain = w3.eth.chain_id
    client = w3.client_version
    art = json.loads(ART.read_text())
    abi, bytecode = art["abi"], art["bytecode"]["object"]
    nonce = w3.eth.get_transaction_count(acct.address)

    def fees():
        base = w3.eth.get_block("latest").get("baseFeePerGas", 0) or 0
        return {"maxFeePerGas": int(base * 2 + 10**9), "maxPriorityFeePerGas": 10**9} if base else {"gasPrice": 0}

    def send(tx, wait=True):
        nonlocal nonce
        tx = dict(tx, nonce=nonce, chainId=chain, **fees())
        tx.setdefault("gas", 3_000_000)
        signed = acct.sign_transaction(tx)
        h = w3.eth.send_raw_transaction(signed.raw_transaction)
        nonce += 1
        if not wait:
            return h
        t = time.time()
        r = w3.eth.wait_for_transaction_receipt(h, timeout=120, poll_latency=0.05)
        return r, time.time() - t

    r, _ = send({"data": bytecode, "gas": 3_000_000})
    assert r.status == 1, "deploy failed"
    ledger = w3.eth.contract(address=r.contractAddress, abi=abi)
    out = {"label": args.label, "client": client, "chain_id": chain, "contract": r.contractAddress,
           "deploy_gas": r.gasUsed}
    model = Web3.keccak(text="gpt-4-1106-preview")
    receipt = Web3.keccak(text="router-receipt")

    def call(fn, *a):
        return fn(*a).build_transaction({"from": acct.address, "gas": 500_000, "nonce": 0, "chainId": chain, **fees()})

    for name, fn in (("store_record", ledger.functions.storeRecord), ("log_record", ledger.functions.logRecord)):
        gas, lat = [], []
        for i in range(args.n):
            r, dt = send(call(fn, model, i % 9, 7500, 123456 + i, 900 + i, receipt))
            assert r.status == 1
            gas.append(r.gasUsed); lat.append(dt)
        out[name] = {"gas_mean": statistics.mean(gas), "gas_first": gas[0], "gas_rest_mean": statistics.mean(gas[1:]),
                     "latency_s_mean": statistics.mean(lat), "latency_s_p95": sorted(lat)[int(0.95 * len(lat)) - 1]}
        print(name, out[name], flush=True)
    gas, lat = [], []
    for i in range(args.n):
        root = hashlib.sha256(f"root{i}".encode()).digest()
        r, dt = send(call(ledger.functions.commitBatch, root, 256))
        gas.append(r.gasUsed); lat.append(dt)
    out["commit_batch"] = {"gas_mean": statistics.mean(gas), "gas_first": gas[0], "gas_rest_mean": statistics.mean(gas[1:]),
                           "latency_s_mean": statistics.mean(lat)}
    print("commit_batch", out["commit_batch"], flush=True)

    # Merkle proofs: a real batch committed on chain, valid proof accepted, edited record rejected.
    ver = {}
    for B in (16, 64, 256, 1024):
        leaves = [leaf(i, "m%d" % (i % 5), 0.75, 0.00123) for i in range(B)]
        lv = tree(leaves)
        r, _ = send(call(ledger.functions.commitBatch, lv[-1][0], B))
        bid = ledger.functions.recordCount().call() * 0 + len(ver) + args.n  # batches committed before: args.n
        idx = B // 3
        pf = proof(lv, idx)
        ok = ledger.functions.verify(bid, leaves[idx], pf, idx).call()
        bad_leaf = leaf(idx, "m%d" % (idx % 5), 1.0, 0.00123)  # the same record with an edited quality
        rejected = not ledger.functions.verify(bid, bad_leaf, pf, idx).call()
        g = ledger.functions.verify(bid, leaves[idx], pf, idx).estimate_gas({"from": acct.address})
        ver[str(B)] = {"depth": len(pf), "verify_gas_estimate": g, "valid_proof_accepted": bool(ok),
                       "edited_record_rejected": bool(rejected)}
        print("verify", B, ver[str(B)], flush=True)
    out["verify_inclusion"] = ver

    # Burst: records/s with the log mode (the cheapest per-record on-chain mode).
    b0 = w3.eth.block_number
    t = time.time()
    hashes = [send(call(ledger.functions.logRecord, model, 1, 5000, 1000 + i, 800, receipt), wait=False)
              for i in range(args.burst)]
    rs = [w3.eth.wait_for_transaction_receipt(h, timeout=300, poll_latency=0.1) for h in hashes]
    dt = time.time() - t
    blocks = sorted({x.blockNumber for x in rs})
    out["burst_log_record"] = {"records": args.burst, "seconds": dt, "records_per_s": args.burst / dt,
                               "blocks_used": len(blocks), "first_block_after_start": blocks[0] - b0,
                               "all_ok": all(x.status == 1 for x in rs)}
    print("burst", out["burst_log_record"], flush=True)
    path = ROOT / f"results/phase5/ledger_{args.label}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
