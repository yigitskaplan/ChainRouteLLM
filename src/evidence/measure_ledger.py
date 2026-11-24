"""Stage 5: measure the evidence ledger on the 4-validator Besu chain.

    infrastructure/besu/start_besu.sh
    python -m src.evidence.measure_ledger

On a fresh EvidenceLedger (not the one holding the live receipts) this measures:

* gas of each action: deploying, approving a router, registering a model, adding a receipt;
* confirmation time: receipts sent one after another, each timed from sending until it is in a block;
* throughput: receipts signed beforehand by several routers (Besu allows about 200 pending per
  account) and then all sent at once, timed until the last is in a block, so the chain is measured
  rather than how fast a client can sign;
* both again with 1 of the 4 validators stopped (it is started again afterwards).

Gas does not depend on the number of validators (it is the fixed cost of the contract's code with the
same compiler settings), so Fevzi Ege Yurtsevenler's gas figures for his three write modes (storage,
event, Merkle batch; phase5 branch, 1 validator) hold on this chain too; time and throughput depend
on the network and are measured here. Writes results/stage5/ledger_measurements.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from src.evidence.ledger import RPC, Account, Ledger, dev_account, run

ADD = "addReceipt(bytes32,uint8,bytes32,uint16,uint64,uint32)"
OUT = Path("results/stage5/ledger_measurements.json")
PER_ROUTER = 200  # Besu refuses more than about 200 pending transactions ahead of one account


def rpc(method: str, params: list, url: str = RPC):
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
    with urllib.request.urlopen(urllib.request.Request(url, body, {"Content-Type": "application/json"}), timeout=60) as r:
        reply = json.loads(r.read())
    if "error" in reply:
        raise RuntimeError(f"{method}: {reply['error']}")
    return reply["result"]


def rpc_batch(calls: list, url: str = RPC, size: int = 200) -> list:
    """Several JSON-RPC calls, sent in batches of ``size`` (Besu limits the calls per request)."""
    replies = []
    for start in range(0, len(calls), size):
        chunk = calls[start:start + size]
        body = json.dumps([{"jsonrpc": "2.0", "id": i, "method": m, "params": p} for i, (m, p) in enumerate(chunk)]).encode()
        with urllib.request.urlopen(urllib.request.Request(url, body, {"Content-Type": "application/json"}), timeout=120) as r:
            reply = json.loads(r.read())
        if isinstance(reply, dict):  # the whole batch was refused
            raise RuntimeError(f"batch refused: {reply.get('error', reply)}")
        replies += sorted(reply, key=lambda x: x["id"])
    return replies


def question(label: str) -> str:
    """A made-up question hash for the measurements (the contract only needs 32 distinct bytes)."""
    return "0x" + hashlib.sha256(label.encode()).hexdigest()


def receipt_args(ledger: Ledger, label: str, model_id: str) -> List[str]:
    return [question(label), "3", model_id, "10000", "7424000", "3835"]


def confirmation(ledger: Ledger, model_id: str, n: int, tag: str) -> dict:
    """n receipts one after another; seconds from sending to being in a block, and gas."""
    seconds, gas = [], []
    for i in range(n):
        start = time.time()
        receipt = ledger.send(ADD, *receipt_args(ledger, f"{tag}-{i}", model_id))
        seconds.append(time.time() - start)
        gas.append(int(receipt["gasUsed"], 16))
    s = np.array(seconds)
    return {"receipts": n, "seconds_mean": round(float(s.mean()), 3), "seconds_median": round(float(np.median(s)), 3),
            "seconds_p95": round(float(np.percentile(s, 95)), 3), "seconds_max": round(float(s.max()), 3),
            "gas_per_receipt": int(np.median(gas))}


def new_routers(ledger: Ledger, count: int) -> List[Ledger]:
    """``count`` new throw-away router accounts, approved by the owner (gas is free, so no funds needed)."""
    wallets = json.loads(run("cast", "wallet", "new", "--number", str(count), "--json"))["data"]
    routers = []
    for w in wallets:
        ledger.send("addRouter(address)", w["address"])
        routers.append(Ledger(ledger.address, Account(w["address"], w["private_key"]), ledger.rpc))
    return routers


def throughput(ledger: Ledger, model_id: str, n: int, tag: str) -> dict:
    """n receipts signed beforehand by several routers, then all sent at once; timed until the last is in a block.

    Besu keeps at most about 200 not-yet-mined transactions ahead of one account's confirmed nonce
    ("nonce too distant"), so the load comes from ceil(n / 200) routers sending in parallel, as many
    routers would; each signs at most 200 receipts.
    """
    routers = new_routers(ledger, -(-n // PER_ROUTER))
    raw, owners = [], []
    for k, router in enumerate(routers):
        count = min(PER_ROUTER, n - k * PER_ROUTER)
        raw += [router.sign(ADD, *receipt_args(router, f"{tag}-{k}-{i}", model_id), nonce=i) for i in range(count)]
        owners += [(router, count)]
    order = np.argsort([i % PER_ROUTER for i in range(len(raw))], kind="stable")  # interleave the routers
    raw = [raw[i] for i in order]
    first_block = int(rpc("eth_blockNumber", []), 16)
    start = time.time()
    hashes = []
    for i in range(0, n, 200):
        replies = rpc_batch([("eth_sendRawTransaction", [tx]) for tx in raw[i:i + 200]])
        bad = [r["error"] for r in replies if "error" in r]
        if bad:
            raise RuntimeError(f"sending failed: {bad[:2]}")
        hashes += [r["result"] for r in replies]
    sent = time.time() - start
    while any(int(rpc("eth_getTransactionCount", [r.account.address, "latest"]), 16) < count for r, count in owners):
        time.sleep(0.2)
        if time.time() - start > 600:
            raise TimeoutError("receipts not mined in 10 minutes")
    seconds = time.time() - start
    receipts = [r["result"] for r in rpc_batch([("eth_getTransactionReceipt", [h]) for h in hashes])]
    ok = sum(r["status"] == "0x1" for r in receipts)
    blocks = sorted({int(r["blockNumber"], 16) for r in receipts})
    return {"receipts": n, "routers_sending": len(routers), "all_ok": ok == n, "ok": ok, "seconds_to_send": round(sent, 2), "seconds_until_all_in_blocks": round(seconds, 2),
            "receipts_per_second": round(n / seconds, 1), "blocks_used": len(blocks),
            "blocks_waited_from_start": blocks[-1] - first_block, "max_receipts_in_one_block": max(
                sum(int(r["blockNumber"], 16) == b for r in receipts) for b in blocks)}


def wait_for_block(after: int, timeout: float = 120.0) -> None:
    start = time.time()
    while int(rpc("eth_blockNumber", []), 16) <= after:
        if time.time() - start > timeout:
            raise TimeoutError("no new block")
        time.sleep(0.5)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Measure gas, confirmation time and throughput of the evidence ledger.")
    parser.add_argument("--confirm", type=int, default=30, help="receipts timed one by one")
    parser.add_argument("--burst", type=int, default=2000, help="receipts sent at once")
    parser.add_argument("--burst-faulty", type=int, default=1000, help="receipts sent at once with a validator down")
    args = parser.parse_args(argv)

    client = run("cast", "rpc", "web3_clientVersion", "--rpc-url", RPC).strip('"')
    account = dev_account()
    ledger = Ledger.deploy(account, record=None)
    gas = {"deploy": int(run("cast", "receipt", ledger.deploy_tx, "gasUsed", "--rpc-url", RPC))}
    gas["add_router"] = int(ledger.send("addRouter(address)", account.address)["gasUsed"], 16)
    gas["register_model"] = int(ledger.send("registerModel(string,uint64,uint64)", "claude-opus-5-5", "4000000", "20000000")["gasUsed"], 16)
    model_id = ledger.model_id("claude-opus-5-5")
    print(f"measuring on {client}, contract {ledger.address}\ngas: {gas}")

    results = {"client": client, "validators": 4, "block_period_s": 2, "contract": ledger.address, "gas": gas}
    results["healthy"] = {"confirmation": confirmation(ledger, model_id, args.confirm, "ok"),
                          "throughput": throughput(ledger, model_id, args.burst, "ok-burst")}
    print("4 of 4 validators:", json.dumps(results["healthy"]))

    subprocess.run(["docker", "stop", "besu-node4"], check=True, capture_output=True)
    try:
        wait_for_block(int(rpc("eth_blockNumber", []), 16))
        results["one_validator_down"] = {"confirmation": confirmation(ledger, model_id, args.confirm, "down"),
                                         "throughput": throughput(ledger, model_id, args.burst_faulty, "down-burst")}
        print("3 of 4 validators:", json.dumps(results["one_validator_down"]))
    finally:
        subprocess.run(["docker", "start", "besu-node4"], check=True, capture_output=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
