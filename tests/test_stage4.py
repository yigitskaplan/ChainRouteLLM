"""Stage 4: LLMRouterBench table, its own split, its metrics, and the router without latency."""
import random

import numpy as np
import pandas as pd
import pytest

from src.data.llmrouterbench import MODELS, Record, build_table, make_split, paper_train_prompts
from src.routing.router import Router, Scales
from src.routing.run_stage4 import measure, oracle


def records(dataset, n, models=("a", "b")):
    return [Record(dataset, m, i, f"{dataset} question {i}", float(i % 2), 0.01 * (j + 1)) for i in range(n) for j, m in enumerate(models)]


def test_split_is_the_papers_procedure():
    data = records("x", 10) + records("y", 7)
    train = paper_train_prompts(data, seed=42)
    # One random.seed, then per dataset: prompts sorted by index, shuffled, first int(n * 0.7) kept.
    rng = random.Random(42)
    expected = set()
    for name, n in (("x", 10), ("y", 7)):
        positions = list(range(n))
        rng.shuffle(positions)
        expected |= {(name, f"{name} question {i}") for i in positions[: int(n * 0.7)]}
    assert train == expected
    assert paper_train_prompts(data, seed=42) == train  # same seed, same split
    assert paper_train_prompts(data, seed=999) != train


def test_a_second_prompt_text_counts_as_an_extra_prompt():
    data = records("x", 10) + [Record("x", "b", 3, "x question 3 (other text)", 1.0, 0.02)]
    assert len(paper_train_prompts(data, seed=1)) == int(11 * 0.7)  # 11 prompt texts, not 10 questions


def test_table_fills_missing_answers_with_zero_and_keeps_the_common_text():
    data = [Record("tau2", MODELS[0], 0, "task", 1.0, 0.5), Record("tau2", MODELS[1], 0, "task", 0.0, 0.2),
            Record("tau2", MODELS[2], 0, "task!", 1.0, 0.1)]
    table = build_table(data)
    assert table.loc[0, "prompt"] == "task"
    assert table.loc[0, f"{MODELS[3]}|quality"] == 0.0 and table.loc[0, f"{MODELS[3]}|cost"] == 0.0
    split = make_split(table, {("tau2", "task")}, seed=0)
    assert split.loc[0, "split"] == "offline_fit"


def test_avg_acc_is_the_mean_over_datasets_and_cost_the_total():
    real = {"quality": np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), "cost": np.array([[1.0, 2.0]] * 4)}
    datasets = np.array(["big", "big", "big", "small"])
    out = measure(real, np.array([0, 0, 0, 0]), datasets, ["m0", "m1"])
    assert out["avg_acc"] == pytest.approx(50.0)  # big 100, small 0: per dataset, not per question
    assert out["cost"] == pytest.approx(4.0)
    assert out["picks_m0"] == 1.0
    assert list(oracle(real)) == [0, 0, 0, 1]


def test_router_with_dynamic_c_works_without_latency():
    offline = {"quality": np.array([[0.9, 0.5]]), "cost": np.array([[4.0, 1.0]])}
    scales = Scales(offline)
    assert scales.c == pytest.approx(2.5)
    picks = Router(0.5, scales).route({"quality": np.array([[0.9, 0.6]]), "cost": np.array([[4.0, 1.0]])})
    assert list(picks) == [1]  # 0.9 - 0.5*4/2.5 = 0.1 < 0.6 - 0.5*1/2.5 = 0.4
    with pytest.raises(KeyError):
        Router(0.5, scales, w_L=0.1).route({"quality": np.zeros((1, 2)), "cost": np.zeros((1, 2))})
