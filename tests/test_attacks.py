"""Stage 6: reading a contract refusal and finding edited receipts (no chain needed)."""
import pandas as pd

from src.attacks import ERRORS
from src.attacks.ledger_attacks import error_name, find_edits


def test_refusal_is_named_from_its_selector():
    assert error_name(ERRORS["NotRouter"] + "000000000000000000000000" + "ab" * 20) == "NotRouter"
    assert error_name(ERRORS["AlreadyReported"] + "00" * 64) == "AlreadyReported"
    assert error_name("0xdeadbeef").startswith("unknown")


def test_edited_receipts_are_found_against_the_chain():
    chain = pd.DataFrame({"question_hash": ["q1", "q2", "q3"], "model_id": ["m", "m", "m"], "group": [0, 1, 2],
                          "score": [10_000, 10_000, 0], "cost_nano_usd": [5, 5, 5], "latency_ms": [9, 9, 9]})
    local = chain.copy()
    assert find_edits(local, chain).empty
    local.loc[1, "score"] = 0              # a right answer made wrong
    local.loc[2, "latency_ms"] = 1         # a slow answer made fast
    found = find_edits(local, chain)
    assert sorted(found["question_hash"]) == ["q2", "q3"]


def test_missing_or_extra_receipts_count_as_edits():
    chain = pd.DataFrame({"question_hash": ["q1"], "model_id": ["m"], "group": [0], "score": [1], "cost_nano_usd": [1], "latency_ms": [1]})
    extra = pd.concat([chain, chain.assign(question_hash="q9")], ignore_index=True)
    assert list(find_edits(extra, chain)["question_hash"]) == ["q9"]
