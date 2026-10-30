import json

import numpy as np
import pandas as pd
import pytest

from src.data.splits import make_split
from src.data.tables import write_table
from src.routing.archive import run_stage1
from src.routing.embeddings import cache_path, load_embeddings, save_embeddings
from src.routing.predictors import Predictors
from src.routing.router import Router

MODELS = ["big", "mid", "small"]
FOREST = {"n_estimators": 10, "min_samples_leaf": 2, "max_features": "sqrt"}
FRACTIONS = dict(
    offline_fraction=0.3, online_fraction=0.6, test_fraction=0.1, validation_fraction_of_offline=0.2
)


def test_embeddings_are_saved_and_loaded_by_question_id(tmp_path):
    path = save_embeddings(tmp_path / "e.npz", ["q1", "q2", "q3"], np.arange(6, dtype=np.float32).reshape(3, 2))
    assert load_embeddings(path, ["q3", "q1"]).tolist() == [[4.0, 5.0], [0.0, 1.0]]
    with pytest.raises(KeyError):
        load_embeddings(path, ["q4"])
    assert cache_path("data/processed/x.csv.gz", "org/model", tmp_path) == tmp_path / "x.org__model.npz"


def test_predictors_learn_which_model_is_good_at_what():
    # Two kinds of questions: model a is right on kind 0, model b on kind 1.
    rng = np.random.default_rng(0)
    kind = rng.integers(0, 2, 400)
    X = np.column_stack([kind + rng.normal(0, 0.1, 400), rng.normal(size=400)])
    quality = np.column_stack([kind == 0, kind == 1]).astype(float)
    cost = np.column_stack([np.full(400, 2.0), np.full(400, 1.0)])
    predictors = Predictors(["a", "b"], FOREST, seed=0).fit(X, {"quality": quality, "cost": cost})

    assert predictors.cost_scale_ == pytest.approx(1.5)  # average cost over both models
    guess = predictors.predict(np.array([[0.0, 0.0], [1.0, 0.0]]))
    assert np.argmax(guess["quality"], axis=1).tolist() == [0, 1]
    assert guess["cost"][0].tolist() == pytest.approx([2.0, 1.0])


def test_score_is_accuracy_minus_weighted_cost():
    guess = {"quality": np.array([[0.9, 0.6]]), "cost": np.array([[2.0, 0.5]])}
    assert np.allclose(Router(w_C=0.1, cost_scale=1.0).scores(guess), [[0.9 - 0.2, 0.6 - 0.05]])
    # Dividing by the cost scale: cost 2.0 is "1 average answer" when the scale is 2.
    assert np.allclose(Router(w_C=0.1, cost_scale=2.0).scores(guess), [[0.9 - 0.1, 0.6 - 0.025]])


def test_higher_cost_weight_picks_cheaper_models():
    guess = {"quality": np.tile([0.9, 0.8, 0.7], (5, 1)), "cost": np.tile([1.0, 0.1, 0.01], (5, 1))}
    picks = [Router(w_C, 1.0).route(guess)[0] for w_C in (0, 0.5, 10)]
    assert picks == [0, 1, 2]


def _synthetic_table(n=300, seed=0):
    """Three topics; model j is usually right on topic j. 'big' costs the most."""
    rng = np.random.default_rng(seed)
    topic = rng.integers(0, 3, n)
    table = pd.DataFrame(
        {
            "sample_id": [f"synthetic.{i}" for i in range(n)],
            "group_id": [f"g{i}" for i in range(n)],
            "eval_name": "synthetic",
            "benchmark": "synthetic",
            "n_turns": 1,
            "prompt": [f"topic {t} question {i}" for i, t in enumerate(topic)],
            "topic": topic,
        }
    )
    table["prompt_chars"] = table["prompt"].str.len()
    for j, (model, price) in enumerate(zip(MODELS, (1e-5, 2e-6, 5e-7))):
        table[f"{model}|quality"] = (rng.random(n) < np.where(topic == j, 0.9, 0.3)).astype(float)
        table[f"{model}|cost"] = price * rng.integers(100, 1000, n)
        table[f"{model}|resp_chars"] = rng.integers(1, 400, n)
    return table


def test_run_end_to_end(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    table = _synthetic_table()
    data = write_table(table, tmp_path / "synthetic.csv.gz")
    write_table(make_split(table, seed=11, **FRACTIONS), tmp_path / "splits" / "synthetic_seed11.csv.gz")
    # Stand-in for BERT: the topic as one-hot numbers plus a little noise.
    numbers = np.eye(3)[table["topic"]] + np.random.default_rng(1).normal(0, 0.05, (len(table), 3))
    save_embeddings(cache_path(data, "fake-bert"), table["sample_id"], numbers.astype(np.float32))
    config = {
        "seeds": [11],
        "data": {"model_keys_primary": MODELS},
        "router": {"encoder_checkpoint": "fake-bert", "w_C_grid": [0.0, 10.0], "predictors": {"random_forest": FOREST}},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))

    picks = run_stage1.route_test(data, tmp_path / "splits" / "synthetic_seed11.csv.gz", config, seed=11)
    topic = table.set_index("sample_id").loc[picks["sample_id"], "topic"].to_numpy()
    # Ignoring cost, the router sends each topic to the model that is good at it.
    assert (picks["w_C=0"].to_numpy() == np.asarray(MODELS)[topic]).mean() > 0.8
    # With a huge cost weight it always picks the cheapest model.
    assert (picks["w_C=10"] == "small").all()

    assert run_stage1.main(["--data", str(data), "--config", "config.json", "--splits-dir", "splits"]) == 0
    assert "seed 11" in capsys.readouterr().out
