"""Stage 7: what a router takes from the receipts (alone or sharing), without a chain."""
import numpy as np
import pytest

from src.evidence.shared import Evidence, StreamReceipt, correction, mean_guess, scales, totals_of, units


def receipt(router, group, model, score, cost=1_000, latency=2_000):
    return StreamReceipt(router, -1, group, model, score, cost, latency)


def test_units_match_the_contract():
    assert units(1.0, 0.011773, 5.99) == (10_000, 11_773_000, 5_990)
    assert units(0.0, 0.000369, 3.77) == (0, 369_000, 3_770)


def test_alone_sees_only_its_own_receipts_sharing_sees_all():
    ev = Evidence(groups=2, models=2)
    ev.add([receipt(0, 0, 0, 10_000), receipt(1, 0, 0, 0), receipt(1, 1, 1, 10_000)])
    assert len(ev.visible(0)) == 1 and len(ev.visible(1)) == 2 and len(ev.visible(None)) == 3
    t = ev.totals(None)
    assert t.n_group[0, 0] == 2 and t.score_group[0, 0] == 10_000 and t.n_model.tolist() == [2, 1]


def test_correction_is_damped_by_the_number_of_receipts():
    t = totals_of([receipt(0, 0, 0, 10_000)] * 20, groups=1, models=2)
    guess = np.array([[0.5, 0.5]])
    corr = correction(t, guess, tau=20)
    assert corr[0, 0] == pytest.approx(20 / 40 * (1.0 - 0.5))  # half way at n = tau
    assert corr[0, 1] == 0.0                                    # no receipts, no correction


def test_no_correction_for_a_group_without_a_guess():
    t = totals_of([receipt(0, 1, 0, 10_000)], groups=2, models=1)
    guess = mean_guess(np.array([[0.4]]), np.array([0]), groups=2)  # warm rows only in group 0
    assert np.isnan(guess[1, 0])
    assert correction(t, guess, tau=20)[1, 0] == 0.0


def test_scales_add_the_stream_to_the_warm_start():
    warm = {"cost": np.array([[1.0, 3.0], [1.0, 3.0]]), "latency": np.array([[2.0, 4.0], [2.0, 4.0]])}
    t = totals_of([receipt(0, 0, 0, 10_000, cost=4_000_000_000, latency=8_000)], groups=1, models=2)
    s = scales(warm, t)
    assert s.answers.tolist() == [3, 2]
    assert s.c == pytest.approx(((1 + 1 + 4) / 3 + 3) / 2)   # mean over models of each model's mean cost
    assert s.l == pytest.approx(((2 + 2 + 8) / 3 + 4) / 2)
