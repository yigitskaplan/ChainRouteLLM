"""Phase 5 metrics: baselines on the cost-quality plane and RouterBench's AIQ.

Quality is the mean score of the chosen answers, cost the mean billed USD per query, both on the
same test queries. A router with a parameter (w_C) gives one (cost, quality) point per value.

AIQ (Hu et al., RouterBench 2024): the points plus (0, 0) are joined into the non-decreasing upper
convex hull; AIQ is the average height of that curve between the cheapest and the most expensive
single model's mean cost. Mixing two operating points at random reaches any point on the segment
between them, which is why the hull, not the raw points, is the fair curve.
"""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np


def upper_hull(points: Iterable[Tuple[float, float]]) -> np.ndarray:
    """Non-decreasing upper convex hull of (cost, quality) points, (0, 0) included, sorted by cost."""
    pts = sorted(set((float(c), float(q)) for c, q in points) | {(0.0, 0.0)})
    hull = []
    for p in pts:
        # Keep only a concave (upper) chain.
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) >= 0:
                hull.pop()
            else:
                break
        hull.append(p)
    hull = np.asarray(hull)
    # Non-decreasing: past the best quality the curve stays flat.
    top = int(np.argmax(hull[:, 1]))
    return hull[: top + 1]


def curve_at(hull: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Hull height at costs c (flat beyond the last hull point)."""
    return np.interp(c, hull[:, 0], hull[:, 1], left=0.0, right=float(hull[-1, 1]))


def aiq(points: Iterable[Tuple[float, float]], c_min: float, c_max: float, n: int = 20001) -> float:
    hull = upper_hull(points)
    grid = np.linspace(c_min, c_max, n)
    return float(np.trapezoid(curve_at(hull, grid), grid) / (c_max - c_min))


def single_model_points(Q: np.ndarray, C: np.ndarray) -> np.ndarray:
    """[models, 2] mean (cost, quality) of always using each model."""
    return np.column_stack([C.mean(axis=0), Q.mean(axis=0)])


def oracle_choice(Q: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Best model per query, cheapest on ties (the 'cheater' with full knowledge)."""
    best = Q.max(axis=1, keepdims=True)
    masked = np.where(Q >= best - 1e-12, C, np.inf)
    return masked.argmin(axis=1)


def oracle_curve(Q: np.ndarray, C: np.ndarray, lambdas: Sequence[float]) -> np.ndarray:
    """Oracle trade-off curve: per query argmax(q - lambda * c / mean cost), cheapest on ties."""
    scale = float(C.mean())
    rows = np.arange(len(Q))
    out = []
    for lam in lambdas:
        s = Q - lam * C / scale
        best = s.max(axis=1, keepdims=True)
        pick = np.where(s >= best - 1e-12, C, np.inf).argmin(axis=1)
        out.append((C[rows, pick].mean(), Q[rows, pick].mean()))
    return np.asarray(out)


def random_router(Q: np.ndarray, C: np.ndarray, seed: int) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, Q.shape[1], len(Q))
    rows = np.arange(len(Q))
    return float(C[rows, pick].mean()), float(Q[rows, pick].mean())


def picks_point(Q: np.ndarray, C: np.ndarray, picks: np.ndarray) -> Tuple[float, float]:
    rows = np.arange(len(Q))
    return float(C[rows, picks].mean()), float(Q[rows, picks].mean())
