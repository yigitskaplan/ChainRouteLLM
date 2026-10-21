"""Small file helpers shared by the data scripts."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Sequence, Union

import pandas as pd

PathLike = Union[str, Path]
_TABLE_SUFFIXES = (".csv.gz", ".csv", ".parquet")
# Read id columns as text so ids are never turned into numbers.
_TEXT_COLUMNS = {"sample_id": str, "group_id": str}


def sha256_file(path: PathLike) -> str:
    """Return the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def table_stem(path: PathLike) -> str:
    """File name without its table suffix: data/x.csv.gz -> x."""
    name = Path(path).name
    for suffix in _TABLE_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(path).stem


def read_table(path: PathLike, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Read a .csv, .csv.gz or .parquet table."""
    path = Path(path)
    if path.name.endswith(".parquet"):
        return pd.read_parquet(path, columns=None if columns is None else list(columns))
    return pd.read_csv(
        path,
        usecols=None if columns is None else list(columns),
        dtype=_TEXT_COLUMNS,
        keep_default_na=True,
    )


def write_table(df: pd.DataFrame, path: PathLike) -> Path:
    """Write a .csv, .csv.gz or .parquet table and return its path.

    Gzip output is written without a timestamp, so the same table always gives
    the same file and the same checksum.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.name.endswith(".parquet"):
        df.to_parquet(path, index=False)
    elif path.name.endswith(".gz"):
        df.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
    else:
        df.to_csv(path, index=False)
    return path
