"""Stage 5: automatic question groups for the ledger.

A receipt says which group its question falls in, never the question. The groups come from k-means
on the questions' BERT vectors (no human topics): a question's group is its nearest centroid. K = 8
for now: in Fevzi Ege Yurtsevenler's ablation (phase5 branch) K = 8 helped most (+0.48 pts, against
+0.25 for K = 32 and -0.06 for K = 128), and our 250 live questions give about 30 per group at K = 8
but only about 8 at K = 32. More groups once there is more data.

The centroids are saved to data/evidence/groups_k8.npz, so every run gives the same groups.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from sklearn.cluster import KMeans

CENTROIDS = Path("data/evidence/groups_k8.npz")


def fit(X: np.ndarray, k: int = 8, seed: int = 0) -> np.ndarray:
    """k centroids [k, dim] of the rows of X (k-means++ start, 10 restarts, fixed seed)."""
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X).cluster_centers_.astype(np.float32)


def assign(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Each row's group: the index of its nearest centroid (squared Euclidean distance)."""
    distances = ((X[:, None, :].astype(np.float64) - centroids[None, :, :].astype(np.float64)) ** 2).sum(axis=2)
    return distances.argmin(axis=1)


def save(path: Union[str, Path], centroids: np.ndarray, source: str, seed: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, centroids=centroids, source=np.asarray(source), seed=np.asarray(seed))
    return path


def load(path: Union[str, Path] = CENTROIDS) -> np.ndarray:
    with np.load(path, allow_pickle=False) as saved:
        return saved["centroids"]
