"""Stage 4: LLMRouterBench's performance-cost data as one table, with its own splits.

    python -m src.data.llmrouterbench

LLMRouterBench (Li et al., arXiv 2601.07206, by the authors of Avengers-Pro)
stores every model's answer, score and cost (USD) on every question; the data is
the OpenRouterBench release on Hugging Face (NPULH/OpenRouterBench,
bench-release.tar.gz, 1.28 GB). This downloads it to data/llmrouterbench/ if it
is not there, unpacks the ten datasets of the paper's performance-cost setting
(Table 9, 12,446 questions) and keeps its 13 flagship models (Table 8). It writes

* data/processed/llmrouterbench_perf_cost.csv.gz: one row per question
  (sample_id = dataset:index, dataset, index, prompt) with every model's score
  (``<model>|quality``) and cost (``<model>|cost``). A model without an answer
  (GPT-5-Chat on tau2, it cannot call tools) gets score 0 and cost 0, as in the
  paper's adaptors (fill_missing_models_scores);
* data/splits/llmrouterbench/llmrouterbench_perf_cost_seed<seed>.csv.gz for the
  paper's five seeds: its own 70/30 split (baselines/data_loader.py,
  split_by_dataset_then_prompt) redone step by step. The 70% is the warm start
  (``offline_fit``); the 30% test part is the stream (``online``) in a seeded
  arrival order.

The paper splits prompt texts, not questions. Two questions are sent with a
slightly different text to one model (arenahard 103 to Claude Sonnet 4, hle 1494
to GPT-5), which the paper counts as extra prompts; here such a question follows
the text most models got.
"""
from __future__ import annotations

import argparse
import json
import random
import tarfile
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, List, NamedTuple, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from src.data.tables import table_stem, write_table

REPO_ID = "NPULH/OpenRouterBench"
ARCHIVE = "bench-release.tar.gz"
# Dataset -> split, in the order of the paper's files (folder names sorted).
DATASETS = {
    "aime": "hybrid", "arenahard": "test", "gpqa": "test", "hle": "test", "livecodebench": "test",
    "livemathbench": "test", "mmlupro": "test_3000", "simpleqa": "test", "swe-bench": "verified", "tau2": "test",
}
MODELS = (
    "claude-sonnet-4", "deepseek-r1-0528", "deepseek-v3-0324", "deepseek-v3.1-terminus", "gemini-2.5-flash",
    "gemini-2.5-pro", "glm-4.6", "gpt-5", "gpt-5-chat", "intern-s1", "kimi-k2-0905", "qwen3-235b-a22b-2507",
    "qwen3-235b-a22b-thinking-2507",
)
SEEDS = (42, 999, 2024, 2025, 3407)
TRAIN_RATIO = 0.7
TABLE = Path("data/processed/llmrouterbench_perf_cost.csv.gz")


class Record(NamedTuple):
    dataset: str
    model: str
    index: int
    prompt: str
    score: float
    cost: float


def download(data_dir: Path) -> Path:
    """The release archive, downloaded once; the needed datasets unpacked next to it."""
    archive = data_dir / ARCHIVE
    if not archive.exists():
        from huggingface_hub import hf_hub_download

        hf_hub_download(repo_id=REPO_ID, filename=ARCHIVE, repo_type="dataset", local_dir=str(data_dir))
    root = data_dir / "bench-release"
    missing = [name for name in DATASETS if not (root / name).exists()]
    if missing:
        with tarfile.open(archive) as tar:
            wanted = tuple(f"bench-release/{name}/" for name in missing)
            tar.extractall(data_dir, members=[m for m in tar.getmembers() if m.name.startswith(wanted)], filter="data")
    return root


def load_records(root: Path) -> Iterator[Record]:
    """Every answer of the 13 models on the ten datasets, file by file in the paper's order.

    Like the paper's loader, files are read in sorted path order and, for each
    dataset, split and model, only the newest file (by its time-stamped name) counts.
    """
    newest: Dict[Tuple[str, str, str], Path] = {}
    for path in sorted(p for name in DATASETS for p in (root / name).rglob("*.json")):
        with open(path, encoding="utf-8") as f:
            head = json.load(f)
        key = (head["dataset_name"], head["split"], head["model_name"])
        if DATASETS.get(key[0]) == key[1] and key[2] in MODELS and (key not in newest or path.name > newest[key].name):
            newest[key] = path
    for path in sorted(newest.values()):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for r in data["records"]:
            score = 0.0 if r.get("score") is None else float(r["score"])
            yield Record(data["dataset_name"], data["model_name"], int(r["index"]), str(r.get("prompt") or ""),
                         score, float(r.get("cost") or 0.0))


def paper_train_prompts(records: Sequence[Record], seed: int, train_ratio: float = TRAIN_RATIO) -> Set[Tuple[str, str]]:
    """(dataset, prompt text) pairs in the paper's training part.

    LLMRouterBench's split_by_dataset_then_prompt: one random.seed(seed); for each
    dataset in order of appearance, its distinct prompt texts are sorted by their
    smallest index (ties keep the order of appearance), shuffled, and the first
    int(n * train_ratio) of the shuffled positions are the training part.
    """
    rng = random.Random(seed)
    prompts: Dict[str, Dict[str, int]] = {}
    for r in records:
        seen = prompts.setdefault(r.dataset, {})
        seen[r.prompt] = min(seen.get(r.prompt, r.index), r.index)
    train = set()
    for dataset, smallest in prompts.items():
        unique = sorted(smallest, key=smallest.get)
        positions = list(range(len(unique)))
        rng.shuffle(positions)
        train.update((dataset, unique[i]) for i in positions[: int(len(unique) * train_ratio)])
    return train


def build_table(records: Sequence[Record]) -> pd.DataFrame:
    """One row per question with every model's score and cost (0 and 0 when it has no answer)."""
    rows: Dict[Tuple[str, int], dict] = {}
    texts: Dict[Tuple[str, int], Counter] = {}
    for r in records:
        key = (r.dataset, r.index)
        row = rows.setdefault(key, {"sample_id": f"{r.dataset}:{r.index}", "dataset": r.dataset, "index": r.index})
        row[f"{r.model}|quality"], row[f"{r.model}|cost"] = r.score, r.cost
        texts.setdefault(key, Counter())[r.prompt] += 1
    for key, row in rows.items():
        row["prompt"] = texts[key].most_common(1)[0][0]
    columns = ["sample_id", "dataset", "index", "prompt"] + [f"{m}|{k}" for m in MODELS for k in ("quality", "cost")]
    table = pd.DataFrame(list(rows.values())).reindex(columns=columns)
    order = {name: i for i, name in enumerate(DATASETS)}
    table = table.sort_values(["dataset", "index"], key=lambda s: s.map(order) if s.name == "dataset" else s)
    return table.fillna({c: 0.0 for c in columns[4:]}).reset_index(drop=True)


def make_split(table: pd.DataFrame, train: Set[Tuple[str, str]], seed: int) -> pd.DataFrame:
    """The paper's split of every question: warm start (offline_fit) or stream (online, in arrival order)."""
    in_train = np.array([(d, p) in train for d, p in zip(table["dataset"], table["prompt"])])
    split = np.where(in_train, "offline_fit", "online")
    order = np.zeros(len(table), dtype=int)
    order[in_train] = np.arange(in_train.sum())
    order[~in_train] = np.random.default_rng(seed).permutation((~in_train).sum())  # datasets mixed in the stream
    return pd.DataFrame({"sample_id": table["sample_id"], "group_id": table["sample_id"], "split": split, "order": order})


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build the LLMRouterBench performance-cost table and its splits.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/llmrouterbench"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits/llmrouterbench"))
    args = parser.parse_args(argv)

    records: List[Record] = list(load_records(download(args.data_dir)))
    table = build_table(records)
    path = write_table(table, TABLE)
    print(f"{path}: {len(table)} questions, {len(MODELS)} models; " + ", ".join(f"{d} {n}" for d, n in table["dataset"].value_counts(sort=False).items()))
    for seed in SEEDS:
        split = make_split(table, paper_train_prompts(records, seed), seed)
        out = write_table(split, args.splits_dir / f"{table_stem(TABLE)}_seed{seed}.csv.gz")
        print(f"{out}: warm start {int((split['split'] == 'offline_fit').sum())}, stream {int((split['split'] == 'online').sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
