"""Stage 3: one table of the live answers, shaped like the RouterBench table.

    python -m src.data.live_table

Reads every answer in data/live/ (failed calls left out) and the SuperGPQA
questions, and writes data/processed/supergpqa_live.csv.gz with one row per
question that every chosen model has answered:

* ``sample_id`` (the SuperGPQA uuid), ``group_id``, ``discipline``, ``difficulty``;
* ``prompt``: the text the models were sent;
* ``<model>|quality`` (1 right, 0 wrong), ``<model>|cost`` (USD) and
  ``<model>|latency`` (seconds) for every chosen model.

By default the models are the three Claude models; the GPT models join once
their answers are complete.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from src.data.collect_live import prompt
from src.data.tables import PathLike, write_table

CLAUDE = ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"]


def build(live_dir: PathLike, data_dir: PathLike, models: Sequence[str]) -> pd.DataFrame:
    answers = pd.concat([pd.read_csv(path) for path in sorted(Path(live_dir).glob("*.csv"))], ignore_index=True)
    answers = answers[answers.error.isna() & answers.model.isin(models)]
    wide = answers.pivot(index="uuid", columns="model", values=["correct", "cost", "seconds"]).dropna().sort_index()
    questions = pd.read_json(Path(data_dir) / "SuperGPQA-all.jsonl", lines=True).set_index("uuid").loc[wide.index]
    table = pd.DataFrame(
        {
            "sample_id": wide.index,
            "group_id": wide.index,
            "discipline": questions.discipline.to_numpy(),
            "difficulty": questions.difficulty.to_numpy(),
            "prompt": [prompt(q, o) for q, o in zip(questions.question, questions.options)],
        }
    )
    for model in models:
        table[f"{model}|quality"] = wide[("correct", model)].astype(float).to_numpy()
        table[f"{model}|cost"] = wide[("cost", model)].astype(float).to_numpy()
        table[f"{model}|latency"] = wide[("seconds", model)].astype(float).to_numpy()
    return table


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build the table of live answers for Stage 3.")
    parser.add_argument("--live-dir", type=Path, default=Path("data/live"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/supergpqa"))
    parser.add_argument("--models", nargs="+", default=CLAUDE)
    parser.add_argument("--out", type=Path, default=Path("data/processed/supergpqa_live.csv.gz"))
    args = parser.parse_args(argv)
    table = build(args.live_dir, args.data_dir, args.models)
    path = write_table(table, args.out)
    print(f"{len(table)} questions answered by all of {', '.join(args.models)} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
