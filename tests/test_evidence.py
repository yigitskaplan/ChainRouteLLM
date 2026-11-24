"""Stage 5: reading receipts from event logs, the question groups and the totals (no chain needed)."""
import numpy as np
import pytest

from src.evidence import groups
from src.evidence.ledger import Receipt, decode_receipt, totals


def word(value: int) -> str:
    return f"{value:064x}"


def test_receipt_log_is_decoded():
    log = {
        "topics": ["0x" + "aa" * 32, "0x" + "11" * 32, "0x" + "22" * 32, "0x" + "0" * 24 + "ab" * 20],
        "data": "0x" + word(3) + word(10_000) + word(7_424_000) + word(3_835),
        "blockNumber": "0x8a44",
    }
    r = decode_receipt(log)
    assert r == Receipt("0x" + "11" * 32, "0x" + "22" * 32, "0x" + "ab" * 20, 3, 10_000, 7_424_000, 3_835, 0x8A44)


def test_group_is_the_nearest_centroid():
    centroids = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]], dtype=np.float32)
    X = np.array([[1.0, 1.0], [9.0, 1.0], [1.0, 8.0], [6.0, 0.0]])
    assert list(groups.assign(X, centroids)) == [0, 1, 2, 1]


def test_groups_are_reproducible(tmp_path):
    X = np.random.default_rng(0).normal(size=(200, 5))
    a, b = groups.fit(X, k=4, seed=0), groups.fit(X, k=4, seed=0)
    assert np.array_equal(a, b)
    path = groups.save(tmp_path / "g.npz", a, source="test", seed=0)
    assert np.array_equal(groups.load(path), a)


def test_totals_per_model_and_group():
    r = lambda g, m, s, c, l: Receipt("0xq", m, "0xr", g, s, c, l, 1)
    receipts = [r(0, "0xa", 10_000, 2_000, 1_000), r(0, "0xa", 0, 4_000, 3_000), r(1, "0xa", 10_000, 0, 2_000), r(1, "0xb", 5_000, 9_000, 500)]
    names = {"0xa": "opus", "0xb": "sol"}
    per_model = totals(receipts, names)
    assert per_model.loc["opus", "answers"] == 3
    assert per_model.loc["opus", "accuracy"] == pytest.approx(2 / 3)
    assert per_model.loc["opus", "cost"] == pytest.approx(2e-6)       # (2,000 + 4,000 + 0) / 3 nano-USD
    assert per_model.loc["opus", "latency"] == pytest.approx(2.0)     # (1 + 3 + 2) / 3 s
    assert per_model.loc["sol", "accuracy"] == pytest.approx(0.5)
    per_group = totals(receipts, names, by_group=True)
    assert per_group.loc[(0, "opus"), "accuracy"] == pytest.approx(0.5)
    assert per_group.loc[(1, "opus"), "accuracy"] == pytest.approx(1.0)
