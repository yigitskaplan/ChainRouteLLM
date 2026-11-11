"""SuperGPQA question batches for the Stage 3 live-model runs.

    python -m src.data.supergpqa

Downloads SuperGPQA (m-a-p/SuperGPQA on Hugging Face, ODC-BY licence) into
data/supergpqa/ if it is not there, then writes the questions the live models
answer, one file per batch, to data/supergpqa/batches/:

* batch_0_10.csv: the 10-question test run (3 easy, 4 middle, 3 hard, seed 0);
* batch_1_90.csv and batch_2_150.csv: a seeded random draw of the rest.

Questions whose text is the same apart from case and punctuation count as one
and only the first is kept, so no question is asked twice. No question is in
two batches. Each row has the SuperGPQA uuid, its discipline, field, subfield,
difficulty and correct letter; the question text stays in the download.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from src.data.tables import write_table

REPO_ID = "m-a-p/SuperGPQA"
FILE = "SuperGPQA-all.jsonl"
# How the 10-question test run picked its questions: a fixed number per difficulty.
TEST_RUN = {"easy": 3, "middle": 4, "hard": 3}
TEST_SEED = 0
COLUMNS = ["uuid", "discipline", "field", "subfield", "difficulty", "answer_letter"]


def load_questions(data_dir: Path) -> pd.DataFrame:
    path = data_dir / FILE
    if not path.exists():
        from huggingface_hub import hf_hub_download

        hf_hub_download(repo_id=REPO_ID, filename=FILE, repo_type="dataset", local_dir=str(data_dir))
    return pd.read_json(path, lines=True)


def same_text(question: pd.Series) -> pd.Series:
    """Question text without case and punctuation, to find near-identical copies."""
    return question.str.lower().str.replace(r"\W+", " ", regex=True).str.strip()


def make_batches(questions: pd.DataFrame, sizes: Sequence[int], seed: int) -> list:
    """The test-run batch, then one batch per size from a seeded random draw."""
    first = pd.concat(
        questions[questions.difficulty == level].sample(n=n, random_state=TEST_SEED) for level, n in TEST_RUN.items()
    )
    text = same_text(questions.question)
    rest = questions[~text.duplicated() & ~text.isin(same_text(first.question))]
    drawn = rest.sample(n=sum(sizes), random_state=seed)
    ends = pd.Series(sizes).cumsum().tolist()
    return [first] + [drawn.iloc[end - size : end] for size, end in zip(sizes, ends)]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Write the SuperGPQA question batches for the live runs.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/supergpqa"))
    parser.add_argument("--sizes", type=int, nargs="+", default=[90, 150], help="batches after the test run")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args(argv)

    questions = load_questions(args.data_dir)
    for number, batch in enumerate(make_batches(questions, args.sizes, args.seed)):
        path = write_table(batch[COLUMNS], args.data_dir / "batches" / f"batch_{number}_{len(batch)}.csv")
        print(f"{path}: {len(batch)} questions, " + ", ".join(f"{k} {v}" for k, v in batch.difficulty.value_counts().items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
