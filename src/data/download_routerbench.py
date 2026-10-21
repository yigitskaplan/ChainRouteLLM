"""Download RouterBench's wide pickles from Hugging Face into data/.

    python -m src.data.download_routerbench
    python -m src.data.download_routerbench --files routerbench_0shot.pkl

Source: https://huggingface.co/datasets/withmartian/routerbench
The pickles are large and stay out of Git (see .gitignore).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

REPO_ID = "withmartian/routerbench"
DEFAULT_FILES = ["routerbench_0shot.pkl", "routerbench_5shot.pkl"]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Download RouterBench pickles into data/.")
    parser.add_argument("--files", nargs="+", default=DEFAULT_FILES)
    parser.add_argument("--dest", type=Path, default=Path("data"))
    args = parser.parse_args(argv)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise SystemExit("Install it first: pip install huggingface_hub") from exc

    args.dest.mkdir(parents=True, exist_ok=True)
    for name in args.files:
        path = hf_hub_download(
            repo_id=REPO_ID, filename=name, repo_type="dataset", local_dir=str(args.dest)
        )
        print(f"{name} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
