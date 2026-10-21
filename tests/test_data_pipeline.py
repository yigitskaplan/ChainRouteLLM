import json

import numpy as np
import pandas as pd
import pytest

from src.data import build_dataset, splits
from src.data.build_dataset import COST_SUFFIX, RESPONSE_SUFFIX, benchmark_of, build, join_turns
from src.data.splits import SPLITS, load_split, make_split
from src.data.tables import read_table, sha256_file, write_table

MODELS = ["gpt-4-1106-preview", "gpt-3.5-turbo-1106", "claude-instant-v1"]
FRACTIONS = dict(
    offline_fraction=0.3, online_fraction=0.6, test_fraction=0.1, validation_fraction_of_offline=0.2
)


def _row(sample_id, eval_name, prompt, scores, costs=(0.01, 0.001, 0.0005), answers=None):
    row = {"sample_id": sample_id, "eval_name": eval_name, "prompt": prompt}
    answers = answers or [["an answer"]] * len(MODELS)
    for model, score, cost, answer in zip(MODELS, scores, costs, answers):
        row[model] = score
        row[model + COST_SUFFIX] = cost
        row[model + RESPONSE_SUFFIX] = answer
    return row


@pytest.fixture
def wide():
    """RouterBench-shaped table with one example of each problem."""
    rows = [
        _row("mmlu-anatomy.val.0", "mmlu-anatomy", ["Q0"], [1.0, 0.0, 1.0]),
        _row("arc-challenge.val.1", "arc-challenge", ["Q1"], [1.0, 1.0, 0.0]),
        # Empty copy of a repeated id: the scored copy above must survive.
        _row("arc-challenge.val.1", "arc-challenge", ["Q1"], [np.nan, np.nan, np.nan]),
        # Same prompt under two ids (spacing and case differ): one group.
        _row("consensus_summary.dev.1", "consensus_summary", ["Same text"], [0.75, 0.0, 1.0]),
        _row("consensus_summary.dev.2", "consensus_summary", ["same   TEXT"], [1.0, 0.0, 1.0]),
        # Two turns, stored as text the way a CSV round trip leaves them.
        _row(
            "grade-school-math.dev.3",
            "grade-school-math",
            "['Q3 turn 1', 'Q3 turn 2']",
            [0.25, 0.5, 0.0],
            answers=[["a", "b"], "['x', 'yy']", "plain"],
        ),
        _row("mbpp.dev.4", "mbpp", ["Q4"], [1.0, 1.0, 1.0], costs=(0.01, np.nan, 0.001)),
        _row("mtbench.dev.5", "mtbench", ["Q5"], [1.2, 0.5, 0.5]),
    ]
    rows += [
        _row(f"hellaswag.dev.{i}", "hellaswag", [f"Q{i}"], [1.0, float(i % 2), 0.0]) for i in range(6, 60)
    ]
    return pd.DataFrame(rows)


def test_build_removes_bad_rows_and_keeps_scored_copy(wide):
    table, report = build(wide, MODELS)
    assert report["rows_in"] == 62
    assert report["rows_out"] == len(table) == 59
    reasons = {step["reason"]: step["rows"] for step in report["removed"]}
    assert reasons == {
        "missing prompt, score or cost for a selected model": 2,
        "score outside [0, 1] or negative cost": 1,
        "repeated sample_id (first complete copy kept)": 0,
    }
    assert table["sample_id"].is_unique
    arc = table.set_index("sample_id").loc["arc-challenge.val.1"]
    assert arc[[f"{m}|quality" for m in MODELS]].tolist() == [1.0, 1.0, 0.0]
    assert "mbpp.dev.4" not in set(table["sample_id"])
    assert "mtbench.dev.5" not in set(table["sample_id"])


def test_build_groups_shared_prompts_and_parses_turns(wide):
    table, report = build(wide, MODELS)
    by_id = table.set_index("sample_id")
    assert by_id.loc["consensus_summary.dev.1", "group_id"] == by_id.loc["consensus_summary.dev.2", "group_id"]
    assert table["group_id"].nunique() == len(table) - 1
    assert report["shared_prompts"] == {
        "groups": 1,
        "rows": 2,
        "groups_with_different_scores": 1,
        "example_sample_ids": ["consensus_summary.dev.1", "consensus_summary.dev.2"],
    }
    gsm = by_id.loc["grade-school-math.dev.3"]
    assert gsm["prompt"] == "Q3 turn 1\n\nQ3 turn 2"
    assert gsm["n_turns"] == 2
    assert gsm["prompt_chars"] == len("Q3 turn 1\n\nQ3 turn 2")
    assert [gsm[f"{m}|resp_chars"] for m in MODELS] == [len("a\n\nb"), len("x\n\nyy"), len("plain")]
    assert gsm["benchmark"] == "gsm8k"


def test_build_needs_cost_and_answer_columns(wide):
    with pytest.raises(KeyError, match=r"claude-instant-v1\|total_cost"):
        build(wide.drop(columns=["claude-instant-v1" + COST_SUFFIX]), MODELS)


@pytest.mark.parametrize(
    "eval_name, expected",
    [
        ("mmlu-professional-law", "mmlu"),
        ("mtbench-math", "mt-bench"),
        ("grade-school-math", "gsm8k"),
        ("hellaswag", "hellaswag"),
        ("Chinese_character_riddles", "other"),
    ],
)
def test_benchmark_of(eval_name, expected):
    assert benchmark_of(eval_name) == expected


def test_join_turns_handles_missing_and_plain_text():
    assert join_turns(np.nan) is None
    assert join_turns(None) is None
    assert join_turns("[INST] not a list [/INST]") == "[INST] not a list [/INST]"
    assert join_turns(("a", "b")) == "a\n\nb"


def test_split_fractions_groups_and_order(wide):
    table, _ = build(wide, MODELS)
    manifest = make_split(table, seed=11, **FRACTIONS)
    assert len(manifest) == len(table)
    assert set(manifest["sample_id"]) == set(table["sample_id"])
    assert set(manifest["split"]) <= set(SPLITS)
    assert manifest.groupby("group_id")["split"].nunique().max() == 1
    groups = manifest.drop_duplicates("group_id")["split"].value_counts()
    n_groups = table["group_id"].nunique()
    assert groups["online"] == round(n_groups * 0.6)
    assert groups["offline_fit"] + groups["offline_val"] == round(n_groups * 0.3)
    assert groups["offline_val"] == round(round(n_groups * 0.3) * 0.2)
    online = manifest[manifest["split"] == "online"]
    assert online["order"].tolist() == list(range(len(online)))


def test_split_is_seeded_and_ignores_row_order(wide):
    table, _ = build(wide, MODELS)
    first = make_split(table, seed=11, **FRACTIONS)
    shuffled = make_split(table.sample(frac=1.0, random_state=0), seed=11, **FRACTIONS)
    other_seed = make_split(table, seed=23, **FRACTIONS)
    pd.testing.assert_frame_equal(first, shuffled)
    by_seed_11 = first.set_index("sample_id")["split"]
    by_seed_23 = other_seed.set_index("sample_id")["split"].reindex(by_seed_11.index)
    assert not by_seed_11.equals(by_seed_23)


def test_split_rejects_bad_fractions(wide):
    table, _ = build(wide, MODELS)
    with pytest.raises(ValueError):
        make_split(table, 11, 0.5, 0.6, 0.1, 0.2)


def test_scripts_end_to_end(tmp_path, wide):
    pickle_path = tmp_path / "routerbench_0shot.pkl"
    wide.to_pickle(pickle_path)
    out = tmp_path / "processed" / "routerbench_0shot.csv.gz"
    assert build_dataset.main(["--pickle", str(pickle_path), "--out", str(out), "--models", *MODELS]) == 0

    report = json.loads((tmp_path / "processed" / "routerbench_0shot.report.json").read_text())
    assert report["input"]["sha256"] == sha256_file(pickle_path)
    assert report["output"]["sha256"] == sha256_file(out)
    table = read_table(out)
    assert len(table) == report["rows_out"] == 59
    assert pd.api.types.is_string_dtype(table["sample_id"])
    assert table["group_id"].str.startswith("g").all()

    # Same input -> byte-identical output.
    first_hash = sha256_file(out)
    build_dataset.main(["--pickle", str(pickle_path), "--out", str(out), "--models", *MODELS])
    assert sha256_file(out) == first_hash

    config = tmp_path / "experiment.json"
    config.write_text(json.dumps({"experiment": "test", "seeds": [11, 23], "data": FRACTIONS}))
    split_dir = tmp_path / "splits"
    assert splits.main(["--data", str(out), "--config", str(config), "--out-dir", str(split_dir)]) == 0
    summary = json.loads((split_dir / "routerbench_0shot_splits.json").read_text())
    assert set(summary["seeds"]) == {"11", "23"}
    assert sum(summary["seeds"]["11"]["rows"].values()) == 59

    parts = load_split(out, split_dir / "routerbench_0shot_seed11.csv.gz")
    assert set(parts) == set(SPLITS)
    assert sum(len(part) for part in parts.values()) == 59
    assert parts["online"]["order"].tolist() == list(range(len(parts["online"])))
    assert "prompt" in parts["test"].columns


def test_load_split_rejects_other_dataset(tmp_path, wide):
    table, _ = build(wide, MODELS)
    data_path = write_table(table, tmp_path / "data.csv.gz")
    manifest = make_split(table.iloc[:-1], seed=11, **FRACTIONS)
    manifest_path = write_table(manifest, tmp_path / "manifest.csv.gz")
    with pytest.raises(ValueError):
        load_split(data_path, manifest_path)
