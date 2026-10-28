"""Step 1: turn every question into 768 numbers with BERT.

    python -m src.routing.embeddings --data data/processed/routerbench_0shot.csv.gz \
        --config configs/experiment_historical.json

BERT (``router.encoder_checkpoint``) is Google's pre-trained model, loaded
through Hugging Face transformers and never retrained. Each question is cut to
``router.encoder_max_tokens`` tokens; its numbers are the average of BERT's last
layer over its tokens. All questions are saved once to
``data/embeddings/<data name>.<checkpoint>.npz``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from src.data.tables import PathLike, read_table, table_stem


def cache_path(data_path: PathLike, checkpoint: str, out_dir: PathLike = "data/embeddings") -> Path:
    """data/processed/x.csv.gz + model -> data/embeddings/x.model.npz"""
    return Path(out_dir) / f"{table_stem(data_path)}.{checkpoint.replace('/', '__')}.npz"


def encode_bert(texts: Sequence[str], checkpoint: str, max_tokens: int = 512, batch_size: int = 32) -> np.ndarray:
    """768 numbers per text: the average of BERT's last layer over the text's tokens."""
    # Use the Mac's graphics chip when there is one; it is much faster.
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModel.from_pretrained(checkpoint).to(device).eval()
    # Encode texts of similar length together so batches need little padding.
    order = np.argsort([len(text) for text in texts], kind="stable")
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            rows = order[start : start + batch_size]
            batch = tokenizer(
                [texts[i] for i in rows], padding=True, truncation=True, max_length=max_tokens, return_tensors="pt"
            ).to(device)
            hidden = model(**batch).last_hidden_state
            # Average over the real tokens only, not the padding.
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            out[rows] = ((hidden * mask).sum(dim=1) / mask.sum(dim=1)).cpu().numpy()
    return out


def save_embeddings(path: PathLike, sample_ids: Sequence[str], embeddings: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, sample_id=np.asarray(sample_ids, dtype=str), embedding=embeddings)
    return path


def load_embeddings(path: PathLike, sample_ids: Sequence[str]) -> np.ndarray:
    """The saved numbers of the given questions, in that order."""
    with np.load(path, allow_pickle=False) as saved:
        ids, embeddings = saved["sample_id"], saved["embedding"]
    positions = pd.Index(ids).get_indexer(list(sample_ids))
    if (positions < 0).any():
        raise KeyError(f"{int((positions < 0).sum())} questions are not in {path}; rebuild it")
    return embeddings[positions]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Encode every question with BERT and save the numbers.")
    parser.add_argument("--data", required=True, type=Path, help="table from src.data.build_dataset")
    parser.add_argument("--config", required=True, type=Path, help="experiment config (router.encoder_*)")
    args = parser.parse_args(argv)

    router = json.loads(args.config.read_text(encoding="utf-8"))["router"]
    checkpoint = router["encoder_checkpoint"]
    table = read_table(args.data, columns=["sample_id", "prompt"])
    print(f"encoding {len(table)} questions with {checkpoint}")
    embeddings = encode_bert(table["prompt"].tolist(), checkpoint, int(router["encoder_max_tokens"]))
    path = save_embeddings(cache_path(args.data, checkpoint), table["sample_id"], embeddings)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
