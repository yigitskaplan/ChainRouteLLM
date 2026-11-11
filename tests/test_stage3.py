import numpy as np
import pandas as pd
import pytest

from src.data.live_table import build
from src.routing.run_stage3 import measure, oracle
from src.routing.predictors import Predictors
from src.routing.router import Router, Scales

FOREST = {"n_estimators": 20, "min_samples_leaf": 2, "max_features": None}


def test_predictors_also_learn_latency_when_it_is_given():
    # Model a is slow on long questions (x0 = 1), model b is always fast.
    rng = np.random.default_rng(0)
    long = rng.integers(0, 2, 300)
    X = np.column_stack([long + rng.normal(0, 0.05, 300), rng.normal(size=300)])
    outcomes = {
        "quality": np.column_stack([np.ones(300), np.zeros(300)]),
        "cost": np.column_stack([np.full(300, 2.0), np.full(300, 1.0)]),
        "latency": np.column_stack([1.0 + 9.0 * long, np.full(300, 1.0)]),
    }
    guess = Predictors(["a", "b"], FOREST, seed=0).fit(X, outcomes).predict(np.array([[0.0, 0.0], [1.0, 0.0]]))

    assert set(guess) == {"quality", "cost", "latency"}
    assert guess["latency"][:, 0] == pytest.approx([1.0, 10.0], abs=0.5)
    assert guess["latency"][:, 1] == pytest.approx([1.0, 1.0])


def test_scales_are_the_mean_of_each_models_mean():
    offline = {"cost": np.array([[1.0, 4.0], [3.0, 4.0]]), "latency": np.array([[2.0, 6.0], [2.0, 6.0]])}
    scales = Scales(offline)
    assert scales.c == pytest.approx((2.0 + 4.0) / 2)
    assert scales.l == pytest.approx((2.0 + 6.0) / 2)

    scales.add(0, cost=5.0, latency=2.0)  # model 0 now: 3 answers, total cost 9 -> mean 3
    assert scales.c == pytest.approx((3.0 + 4.0) / 2)
    assert scales.l == pytest.approx((2.0 + 6.0) / 2)


def test_scales_do_not_follow_the_routers_own_picks():
    # The router keeps picking the cheap model at its usual cost: a plain mean of the answers
    # seen would fall towards 1, but the typical model's cost does not change.
    scales = Scales({"cost": np.array([[1.0, 9.0]] * 10), "latency": np.ones((10, 2))})
    for _ in range(1000):
        scales.add(0, cost=1.0, latency=1.0)
    assert scales.c == pytest.approx(5.0)


def test_latency_weight_moves_picks_to_the_faster_model():
    guess = {"quality": np.array([[0.8, 0.7]]), "cost": np.array([[1.0, 1.0]]), "latency": np.array([[10.0, 2.0]])}
    scales = Scales({"cost": np.ones((1, 2)), "latency": np.array([[10.0, 2.0]])})  # l = 6
    assert Router(0.0, scales, w_L=0.0).route(guess).tolist() == [0]
    assert Router(0.0, scales, w_L=0.5).route(guess).tolist() == [1]
    # Score = accuracy - w_C * cost / c - w_L * latency / l
    assert Router(0.1, scales, w_L=0.5).scores(guess)[0] == pytest.approx([0.8 - 0.1 - 0.5 * 10 / 6, 0.7 - 0.1 - 0.5 * 2 / 6])


def test_router_without_scales_is_stage_1_and_refuses_a_latency_weight():
    guess = {"quality": np.array([[0.9, 0.6]]), "cost": np.array([[2.0, 0.5]])}
    assert Router(0.1, 2.0).scores(guess)[0] == pytest.approx([0.9 - 0.1, 0.6 - 0.025])
    with pytest.raises(ValueError):
        Router(0.1, 2.0, w_L=0.5)


def test_oracle_picks_the_cheapest_right_model():
    real = {
        "quality": np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "cost": np.array([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0], [3.0, 2.0, 1.0]]),
    }
    assert oracle(real).tolist() == [1, 2, 0]


def test_measure_reports_shares_of_the_reference_model():
    real = {
        "quality": np.array([[1.0, 1.0], [1.0, 0.0]]),
        "cost": np.array([[4.0, 1.0], [4.0, 1.0]]),
        "latency": np.array([[6.0, 2.0], [6.0, 2.0]]),
    }
    out = measure(real, np.array([1, 0]), reference=0, models=["big", "small"])
    assert out["quality_vs_ref"] == pytest.approx(1.0)
    assert out["cost_vs_ref"] == pytest.approx(5 / 8)
    assert out["latency_vs_ref"] == pytest.approx(8 / 12)
    assert out["picks_small"] == pytest.approx(0.5)


def test_live_table_keeps_questions_every_model_answered(tmp_path):
    live, data = tmp_path / "live", tmp_path / "supergpqa"
    live.mkdir(), data.mkdir()
    questions = pd.DataFrame({"uuid": ["q1", "q2"], "question": ["One?", "Two?"], "options": [["x", "y"], ["x", "y"]],
                              "discipline": ["Science", "Law"], "difficulty": ["easy", "hard"]})
    questions.to_json(data / "SuperGPQA-all.jsonl", orient="records", lines=True)
    answer = lambda uuid, model, correct, error=None: dict(uuid=uuid, model=model, correct=correct, cost=0.5,
                                                           seconds=2.0, error=error)
    pd.DataFrame([
        answer("q1", "a", True), answer("q1", "b", False),
        answer("q2", "a", False, error="RateLimitError: 429"),  # failed calls are left out
        answer("q2", "b", True),
    ]).to_csv(live / "batch.csv", index=False)

    table = build(live, data, ["a", "b"])
    assert table["sample_id"].tolist() == ["q1"]  # q2 has no answer from model a
    assert table[["a|quality", "b|quality", "a|cost", "b|latency"]].iloc[0].tolist() == [1.0, 0.0, 0.5, 2.0]
    assert table["prompt"].iloc[0].endswith("One?\n\nA) x\nB) y")
