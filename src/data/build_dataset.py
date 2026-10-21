"""Build the cleaned RouterBench table used by the routing experiments.

Input: RouterBench's wide pickle, one row per query
(``data/routerbench_0shot.pkl`` or ``data/routerbench_5shot.pkl``).
For every selected model the output keeps:

* ``<model>|quality``    RouterBench score in [0, 1]
* ``<model>|cost``       billed cost of the query in USD (``<model>|total_cost``)
* ``<model>|resp_chars`` length of the model's answer in characters

plus ``sample_id``, ``group_id`` (same prompt text -> same group),
``eval_name``, ``benchmark``, ``n_turns``, ``prompt_chars`` and the prompt.

All cleaning happens here, in code. A JSON report next to the output lists
what was removed and why, with SHA-256 checksums of the input and output:

1. drop rows with a missing prompt, score or cost for any selected model;
2. drop rows with a score outside [0, 1] or a negative cost;
3. keep one row per sample_id (empty copies are already gone after step 1);
4. give rows with the same prompt text the same group_id, so the splits keep
   them together (no rows are dropped for this).

Run from the repository root:

    python -m src.data.build_dataset --pickle data/routerbench_0shot.pkl

Do not open and re-save the output in Excel: it changes the delimiter and the
text encoding, and non-English prompts get garbled.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.data.tables import sha256_file, table_stem, write_table

# RouterBench column keys of the proprietary models in this study.
DEFAULT_MODELS: List[str] = [
    "gpt-4-1106-preview",  # GPT-4
    "gpt-3.5-turbo-1106",  # GPT-3.5
    "claude-v1",  # Claude v1
    "claude-v2",  # Claude v2
    "claude-instant-v1",  # Claude Instant
]
COST_SUFFIX = "|total_cost"
RESPONSE_SUFFIX = "|model_response"
MAX_EXAMPLE_IDS = 20

_BENCHMARKS = {
    "hellaswag": "hellaswag",
    "grade-school-math": "gsm8k",
    "arc-challenge": "arc-challenge",
    "winogrande": "winogrande",
    "mbpp": "mbpp",
}


def benchmark_of(eval_name: str) -> str:
    """Map a RouterBench eval_name to its parent benchmark ('other' if none)."""
    name = str(eval_name)
    if name.startswith("mmlu-"):
        return "mmlu"
    if name.startswith("mtbench"):
        return "mt-bench"
    return _BENCHMARKS.get(name, "other")


def as_turns(value: Any) -> Optional[List[str]]:
    """Return a prompt or answer as a list of turns (None if missing).

    RouterBench stores turns as a list. After a CSV round trip the list becomes
    its text form "['turn 1', 'turn 2']", which is parsed back here.
    """
    if isinstance(value, (list, tuple, np.ndarray)):
        return [str(item) for item in value]
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    text = str(value)
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            parsed = None
        if isinstance(parsed, (list, tuple)):
            return [str(item) for item in parsed]
    return [text]


def join_turns(value: Any) -> Optional[str]:
    """Join turns with a blank line (None if missing)."""
    turns = as_turns(value)
    return None if turns is None else "\n\n".join(turns)


def text_length(value: Any) -> float:
    """Number of characters in a prompt or answer (NaN if missing)."""
    text = join_turns(value)
    return float("nan") if text is None else float(len(text))


def group_id_of(prompt: str) -> str:
    """Stable id of a prompt; case and spacing differences are ignored."""
    normalized = " ".join(prompt.split()).lower()
    return "g" + hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


def _examples(ids: pd.Series) -> List[str]:
    return sorted(map(str, ids.unique()))[:MAX_EXAMPLE_IDS]


def _ids_with_different_scores(frame: pd.DataFrame, key: str, score_cols: List[str]) -> int:
    if frame.empty:
        return 0
    return int(frame.groupby(key)[score_cols].nunique().gt(1).any(axis=1).sum())


def build(
    wide: pd.DataFrame, models: Sequence[str] = DEFAULT_MODELS
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Return the cleaned table and a report of what was removed and why."""
    models = list(models)
    required = ["sample_id", "eval_name", "prompt"]
    for model in models:
        required += [model, model + COST_SUFFIX, model + RESPONSE_SUFFIX]
    missing = [column for column in required if column not in wide.columns]
    if missing:
        raise KeyError(f"columns not in the RouterBench table: {missing}")

    wide = wide.reset_index(drop=True)
    table = pd.DataFrame(
        {
            "sample_id": wide["sample_id"].astype(str),
            "eval_name": wide["eval_name"].astype(str),
        }
    )
    table["benchmark"] = table["eval_name"].map(benchmark_of)
    table["prompt"] = wide["prompt"].map(join_turns)
    table["n_turns"] = wide["prompt"].map(lambda value: len(as_turns(value) or []))
    quality_cols, cost_cols = [], []
    for model in models:
        table[f"{model}|quality"] = pd.to_numeric(wide[model], errors="coerce")
        table[f"{model}|cost"] = pd.to_numeric(wide[model + COST_SUFFIX], errors="coerce")
        table[f"{model}|resp_chars"] = wide[model + RESPONSE_SUFFIX].map(text_length)
        quality_cols.append(f"{model}|quality")
        cost_cols.append(f"{model}|cost")

    removed: List[Dict[str, Any]] = []

    def record(mask: pd.Series, reason: str) -> None:
        removed.append(
            {
                "reason": reason,
                "rows": int(mask.sum()),
                "example_sample_ids": _examples(table.loc[mask, "sample_id"]),
            }
        )

    # 1. Missing values.
    mask = (
        table["prompt"].isna()
        | table[quality_cols].isna().any(axis=1)
        | table[cost_cols].isna().any(axis=1)
    )
    record(mask, "missing prompt, score or cost for a selected model")
    table = table.loc[~mask]

    # 2. Values outside their range.
    mask = ((table[quality_cols] < 0) | (table[quality_cols] > 1)).any(axis=1) | (
        table[cost_cols] < 0
    ).any(axis=1)
    record(mask, "score outside [0, 1] or negative cost")
    table = table.loc[~mask]

    # 3. Repeated sample ids: keep the first complete copy.
    repeated = table[table["sample_id"].duplicated(keep=False)]
    mask = table["sample_id"].duplicated(keep="first")
    record(mask, "repeated sample_id (first complete copy kept)")
    removed[-1]["ids_with_different_scores"] = _ids_with_different_scores(
        repeated, "sample_id", quality_cols
    )
    table = table.loc[~mask].copy()

    # 4. Same prompt under different ids: same group, kept together by the splits.
    table["group_id"] = table["prompt"].map(group_id_of)
    table["prompt_chars"] = table["prompt"].str.len()
    group_size = table.groupby("group_id")["sample_id"].transform("size")
    shared = table[group_size > 1]

    columns = ["sample_id", "group_id", "eval_name", "benchmark", "n_turns", "prompt_chars", "prompt"]
    for model in models:
        columns += [f"{model}|quality", f"{model}|cost", f"{model}|resp_chars"]
    table = table[columns].reset_index(drop=True)

    report: Dict[str, Any] = {
        "rows_in": int(len(wide)),
        "rows_out": int(len(table)),
        "models": models,
        "removed": removed,
        "shared_prompts": {
            "groups": int(shared["group_id"].nunique()),
            "rows": int(len(shared)),
            "groups_with_different_scores": _ids_with_different_scores(
                shared, "group_id", quality_cols
            ),
            "example_sample_ids": _examples(shared["sample_id"]),
        },
        "rows_per_benchmark": {
            name: int(count) for name, count in table["benchmark"].value_counts().sort_index().items()
        },
        "per_model": {
            model: {
                "mean_quality": round(float(table[f"{model}|quality"].mean()), 4),
                "total_cost_usd": round(float(table[f"{model}|cost"].sum()), 4),
                "mean_resp_chars": round(float(table[f"{model}|resp_chars"].mean()), 1),
                "rows_without_answer": int(table[f"{model}|resp_chars"].isna().sum()),
            }
            for model in models
        },
        "oracle_mean_quality": round(float(table[quality_cols].max(axis=1).mean()), 4),
    }
    return table, report


def summary(report: Dict[str, Any]) -> str:
    """Short text version of the report for the terminal."""
    lines = [f"rows: {report['rows_in']} in -> {report['rows_out']} out"]
    for step in report["removed"]:
        lines.append(f"  removed {step['rows']:>6}  {step['reason']}")
    shared = report["shared_prompts"]
    lines.append(
        f"  shared prompts: {shared['groups']} groups ({shared['rows']} rows), "
        f"{shared['groups_with_different_scores']} with different scores"
    )
    lines.append(f"{'model':<22}{'quality':>9}{'cost $':>11}{'answer chars':>14}")
    for model, stats in report["per_model"].items():
        lines.append(
            f"{model:<22}{stats['mean_quality']:>9.4f}{stats['total_cost_usd']:>11.2f}"
            f"{stats['mean_resp_chars']:>14.1f}"
        )
    lines.append(f"oracle (best model per query) quality: {report['oracle_mean_quality']:.4f}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build the cleaned RouterBench table.")
    parser.add_argument(
        "--pickle", required=True, type=Path, help="RouterBench pickle, e.g. data/routerbench_0shot.pkl"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output table (.csv.gz, .csv or .parquet); default data/processed/<pickle name>.csv.gz",
    )
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS, help="RouterBench model keys to keep"
    )
    args = parser.parse_args(argv)

    out = args.out or Path("data/processed") / f"{args.pickle.stem}.csv.gz"
    # A pickle can run code when it is loaded: only load the official RouterBench files.
    wide = pd.read_pickle(args.pickle)
    table, report = build(wide, args.models)
    write_table(table, out)
    report = {
        "input": {"path": str(args.pickle), "sha256": sha256_file(args.pickle)},
        "output": {"path": str(out), "sha256": sha256_file(out)},
        **report,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "pandas": pd.__version__,
    }
    report_path = out.with_name(f"{table_stem(out)}.report.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(summary(report))
    print(f"wrote {out} and {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
