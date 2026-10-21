"""Seeded offline / online / test splits by prompt group.

Rows with the same prompt (same ``group_id``) always go to the same split, so
the router is never evaluated on a prompt it was trained on. Fractions and
seeds come from the experiment config (``seeds`` and ``data.offline_fraction``,
``data.online_fraction``, ``data.test_fraction``,
``data.validation_fraction_of_offline``).

    python -m src.data.splits --data data/processed/routerbench_0shot.csv.gz \
        --config configs/experiment_historical.json

For each seed this writes ``data/splits/<data name>_seed<seed>.csv.gz`` with the
columns ``sample_id, group_id, split, order``, plus one summary
``data/splits/<data name>_splits.json`` with checksums and counts.

Splits: ``offline_fit`` and ``offline_val`` (the offline data; the validation
part is for choosing the score weights), ``online`` (the query stream;
``order`` is the arrival order) and ``test`` (held out, no updates).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from src.data.tables import read_table, sha256_file, table_stem, write_table

SPLITS = ("offline_fit", "offline_val", "online", "test")
FRACTION_KEYS = (
    "offline_fraction",
    "online_fraction",
    "test_fraction",
    "validation_fraction_of_offline",
)


def make_split(
    table: pd.DataFrame,
    seed: int,
    offline_fraction: float,
    online_fraction: float,
    test_fraction: float,
    validation_fraction_of_offline: float,
    group_col: str = "group_id",
) -> pd.DataFrame:
    """Assign every row to a split; groups never cross splits."""
    total = offline_fraction + online_fraction + test_fraction
    if not np.isclose(total, 1.0):
        raise ValueError(f"split fractions add up to {total}, not 1")
    if not 0 <= validation_fraction_of_offline < 1:
        raise ValueError("validation_fraction_of_offline must be in [0, 1)")

    # Sorting first makes the result independent of the row order of the table.
    groups = np.sort(table[group_col].astype(str).unique())
    groups = groups[np.random.default_rng(seed).permutation(len(groups))]
    n = len(groups)
    n_offline = int(round(n * offline_fraction))
    n_online = min(int(round(n * online_fraction)), n - n_offline)
    n_val = int(round(n_offline * validation_fraction_of_offline))

    labels = np.full(n, "test", dtype=object)
    labels[:n_val] = "offline_val"
    labels[n_val:n_offline] = "offline_fit"
    labels[n_offline : n_offline + n_online] = "online"

    manifest = pd.DataFrame(
        {
            "sample_id": table["sample_id"].astype(str).to_numpy(),
            "group_id": table[group_col].astype(str).to_numpy(),
        }
    )
    manifest["split"] = manifest["group_id"].map(pd.Series(labels, index=groups))
    manifest["_rank"] = manifest["group_id"].map(pd.Series(np.arange(n), index=groups))
    manifest = (
        manifest.sort_values(["_rank", "sample_id"], kind="mergesort")
        .drop(columns="_rank")
        .reset_index(drop=True)
    )
    manifest["order"] = manifest.groupby("split").cumcount()
    return manifest


def load_split(data_path: Path, manifest_path: Path) -> Dict[str, pd.DataFrame]:
    """Return {split name: its rows of the dataset, in order} for one manifest."""
    data = read_table(data_path)
    manifest = read_table(manifest_path)
    if len(manifest) != len(data):
        raise ValueError(f"{manifest_path} has {len(manifest)} rows, {data_path} has {len(data)}")
    merged = manifest.merge(
        data, on=["sample_id", "group_id"], how="left", validate="one_to_one", indicator=True
    )
    if (merged["_merge"] != "both").any():
        raise ValueError(f"{manifest_path} does not match {data_path}; rebuild the splits")
    merged = merged.drop(columns="_merge")
    return {
        name: merged[merged["split"] == name].sort_values("order").reset_index(drop=True)
        for name in SPLITS
    }


def _counts(series: pd.Series) -> Dict[str, int]:
    return {name: int(series.get(name, 0)) for name in SPLITS}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Write seeded offline/online/test split manifests.")
    parser.add_argument("--data", required=True, type=Path, help="table from src.data.build_dataset")
    parser.add_argument("--config", required=True, type=Path, help="experiment config with seeds and fractions")
    parser.add_argument("--out-dir", type=Path, default=Path("data/splits"))
    args = parser.parse_args(argv)

    config = json.loads(args.config.read_text(encoding="utf-8"))
    fractions = {key: float(config["data"][key]) for key in FRACTION_KEYS}
    seeds = [int(seed) for seed in config["seeds"]]
    table = read_table(args.data, columns=["sample_id", "group_id"])
    stem = table_stem(args.data)

    summary: Dict[str, Any] = {
        "data": {
            "path": str(args.data),
            "sha256": sha256_file(args.data),
            "rows": int(len(table)),
            "groups": int(table["group_id"].nunique()),
        },
        "config": str(args.config),
        "experiment": config.get("experiment"),
        **fractions,
        "seeds": {},
    }
    for seed in seeds:
        manifest = make_split(table, seed, **fractions)
        path = write_table(manifest, args.out_dir / f"{stem}_seed{seed}.csv.gz")
        summary["seeds"][str(seed)] = {
            "manifest": str(path),
            "sha256": sha256_file(path),
            "rows": _counts(manifest["split"].value_counts()),
            "groups": _counts(manifest.groupby("split")["group_id"].nunique()),
        }
        rows = summary["seeds"][str(seed)]["rows"]
        print(f"seed {seed}: " + ", ".join(f"{name} {rows[name]}" for name in SPLITS) + f" -> {path}")

    summary_path = args.out_dir / f"{stem}_splits.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
