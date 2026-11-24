"""Stage 5: talk to the EvidenceLedger contract on the local Besu chain through Foundry.

The chain runs from infrastructure/besu (start_besu.sh); the contract is in contracts/. Transactions
are signed with the throw-away development account in infrastructure/besu/dev_wallet.json, which is
the contract's owner and its first approved router. The deployed address is kept in
infrastructure/besu/deployment.json. Both files stay out of Git.

A receipt is one routed question: its keccak256 hash, its group, the model, the score in basis
points (10,000 = 100%), the cost in nano-USD and the latency in ms. Every router reads all receipts
from the chain and adds them up itself (totals per model and per (group, model)).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence

import pandas as pd

RPC = "http://127.0.0.1:8545"
BESU = Path("infrastructure/besu")
CONTRACTS = Path("contracts")
DEPLOYMENT = BESU / "deployment.json"
RECEIPT = "Receipt(bytes32 indexed,bytes32 indexed,address indexed,uint8,uint16,uint64,uint32)"


def _tool(name: str) -> str:
    return shutil.which(name) or str(Path.home() / ".foundry" / "bin" / name)


def run(tool: str, *args: str, cwd: Optional[Path] = None) -> str:
    """Run a Foundry tool (cast or forge) and return what it prints; fail loudly on an error."""
    result = subprocess.run([_tool(tool), *args], cwd=cwd, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"{tool} {args[0]} failed: {result.stderr.strip()[-400:]}")
    return result.stdout.strip()


class Account(NamedTuple):
    address: str
    key: str


def dev_account() -> Account:
    wallet = json.loads((BESU / "dev_wallet.json").read_text())["data"][0]
    return Account(wallet["address"], wallet["private_key"])


class Receipt(NamedTuple):
    question_hash: str
    model_id: str
    router: str
    group: int
    score: int
    cost_nano_usd: int
    latency_ms: int
    block: int


def decode_receipt(log: dict) -> Receipt:
    """A Receipt event log (as eth_getLogs returns it) as a Receipt."""
    topics, data = log["topics"], log["data"][2:]
    group, score, cost, latency = (int(data[i * 64:(i + 1) * 64], 16) for i in range(4))
    return Receipt(topics[1], topics[2], "0x" + topics[3][-40:], group, score, cost, latency, int(log["blockNumber"], 16))


def keccak(text: str) -> str:
    """keccak256 of the UTF-8 text, as the contract and every router compute a question hash."""
    return run("cast", "keccak", text)


class Ledger:
    """The deployed EvidenceLedger, used by one account."""

    def __init__(self, address: str, account: Account, rpc: str = RPC):
        self.address, self.account, self.rpc = address, account, rpc

    @classmethod
    def deploy(cls, account: Account, rpc: str = RPC, record: Optional[Path] = DEPLOYMENT) -> "Ledger":
        """Deploy a new contract; its address is saved to ``record`` (None: not saved, e.g. for measurements)."""
        out = run("forge", "create", "src/EvidenceLedger.sol:EvidenceLedger", "--rpc-url", rpc,
                  "--private-key", account.key, "--broadcast", cwd=CONTRACTS)
        field = lambda name: next(line.split(":", 1)[1].strip() for line in out.splitlines() if line.startswith(name))
        ledger = cls(field("Deployed to:"), account, rpc)
        ledger.deploy_tx = field("Transaction hash:")
        if record is not None:
            block = int(run("cast", "block-number", "--rpc-url", rpc))
            record.write_text(json.dumps({"address": ledger.address, "block": block}, indent=2) + "\n")
        return ledger

    @classmethod
    def load(cls, account: Account, rpc: str = RPC) -> "Ledger":
        return cls(json.loads(DEPLOYMENT.read_text())["address"], account, rpc)

    def call(self, signature: str, *args: str) -> str:
        return run("cast", "call", self.address, signature, *args, "--rpc-url", self.rpc)

    def send(self, signature: str, *args: str, nonce: Optional[int] = None, wait: bool = True,
             gas_limit: int = 200_000, tries: int = 4):
        """Send a transaction and return its receipt (a dict) once it is in a block; with wait=False
        return its hash at once (the nonce must then be given).

        Gas is free on our chain (zero base fee, minimum gas price 0), so every transaction is sent at
        gas price 0 with a fixed gas limit: no fee or gas estimation calls, which are slower and which
        Besu occasionally answers with a passing internal error. Such errors are retried.
        """
        extra = ["--legacy", "--gas-price", "0", "--gas-limit", str(gas_limit)]
        extra += ["--nonce", str(nonce)] if nonce is not None else []
        base = ["send", self.address, signature, *args, "--rpc-url", self.rpc, "--private-key", self.account.key]
        for attempt in range(tries):
            try:
                if not wait:
                    return run("cast", *base, "--async", *extra)
                receipt = json.loads(run("cast", *base, "--json", *extra))
                break
            except RuntimeError as error:
                if attempt == tries - 1 or "-32603" not in str(error):
                    raise
                time.sleep(1 + attempt)
        if receipt["status"] != "0x1":
            raise RuntimeError(f"{signature} failed in block {int(receipt['blockNumber'], 16)}")
        return receipt

    def sign(self, signature: str, *args: str, nonce: int, gas_limit: int = 200_000, chain_id: int = 1337) -> str:
        """A signed raw transaction, made offline, to be sent later (used to measure the chain's throughput)."""
        return run("cast", "mktx", self.address, signature, *args, "--private-key", self.account.key, "--nonce", str(nonce),
                   "--legacy", "--gas-price", "0", "--gas-limit", str(gas_limit), "--chain", str(chain_id), "--rpc-url", self.rpc)

    def nonce(self) -> int:
        return int(run("cast", "nonce", self.account.address, "--rpc-url", self.rpc))

    def is_router(self, address: str) -> bool:
        return self.call("isRouter(address)(bool)", address) == "true"

    def model_id(self, name: str) -> str:
        return self.call("modelIdOf(string)(bytes32)", name)

    def is_registered(self, name: str) -> bool:
        return self.call("models(bytes32)(bool,uint64,uint64,uint64,string)", self.model_id(name)).splitlines()[0] == "true"

    def receipts(self, from_block: int = 0, window: int = 4_000) -> List[Receipt]:
        """Every receipt from ``from_block`` on, searched in windows of blocks (Besu answers a log
        search over at most 5,000 blocks, and the chain keeps making empty blocks)."""
        last = int(run("cast", "block-number", "--rpc-url", self.rpc))
        found = []
        for start in range(max(from_block, 0), last + 1, window):
            end = min(start + window - 1, last)
            found += json.loads(run("cast", "logs", "--address", self.address, "--from-block", str(start),
                                    "--to-block", str(end), RECEIPT, "--rpc-url", self.rpc, "--json"))
        return [decode_receipt(log) for log in found]


def totals(receipts: Sequence[Receipt], names: Dict[str, str], by_group: bool = False) -> pd.DataFrame:
    """Running totals and averages per model (and per group): what every router computes from the chain.

    ``names`` maps model ids to model names. Accuracy is the mean score (0-1), cost the mean in USD,
    latency the mean in seconds.
    """
    frame = pd.DataFrame(receipts)
    frame["model"] = frame["model_id"].map(names)
    keys = ["group", "model"] if by_group else ["model"]
    out = frame.groupby(keys).agg(answers=("score", "size"), score_sum=("score", "sum"),
                                  cost_sum=("cost_nano_usd", "sum"), latency_sum=("latency_ms", "sum"))
    out["accuracy"] = out["score_sum"] / out["answers"] / 10_000
    out["cost"] = out["cost_sum"] / out["answers"] / 1e9
    out["latency"] = out["latency_sum"] / out["answers"] / 1e3
    return out
