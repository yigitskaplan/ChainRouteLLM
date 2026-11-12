"""Router: send each query to the model with the best guessed quality for its cost and latency.

Score of model m for a query:

    s_m = A_m - w_C * C_m / c - w_L * L_m / l + w_R * R_m

A, C and L are the guessed quality, cost and latency (predictors.py). c and l
are the cost and latency of a typical model, so w_C and w_L are in quality
points per "one typical answer's cost" and "one typical answer's wait". The
router picks argmax_m s_m. Stages 1 and 2 use no latency (w_L = 0) and a fixed
c, the mean offline cost over all models. With a ``Scales`` object c and l are
the mean over the models of each model's own mean and follow every answer seen
(Stage 3). R_m is the public reputation of model m's provider
(src/data/reputation.py); w_R is set by the user and expresses the user's
values, not answer quality.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence, Union

import numpy as np


class Scales:
    """c and l from running totals per model: answers seen, total cost, total latency.

    Averaging per model first keeps them from following the router's own picks:
    if it mostly picked the cheapest model, a plain mean of the answers seen
    would fall and every model would look more expensive. The totals are what
    the ledger keeps per model.
    """

    def __init__(self, offline: Mapping[str, np.ndarray]):
        """Start from the offline answers, where every model answered every question."""
        rows, models = offline["cost"].shape
        self.answers = np.full(models, float(rows))
        self.cost = offline["cost"].sum(axis=0).astype(float)
        self.latency = offline["latency"].sum(axis=0).astype(float)

    def add(self, model: int, cost: float, latency: float) -> None:
        """Take in one answer seen online (only the picked model answers)."""
        self.answers[model] += 1
        self.cost[model] += cost
        self.latency[model] += latency

    @property
    def c(self) -> float:
        return float((self.cost / self.answers).mean())

    @property
    def l(self) -> float:
        return float((self.latency / self.answers).mean())


class Router:
    def __init__(
        self,
        w_C: float,
        cost_scale: Union[float, Scales],
        w_L: float = 0.0,
        w_R: float = 0.0,
        reputation: Optional[Sequence[float]] = None,
    ):
        """``cost_scale`` is a fixed c, or a ``Scales`` that gives c and l and follows the evidence.

        ``reputation`` holds R for every model (its provider's R), in model order.
        """
        self.w_C = float(w_C)
        self.w_L = float(w_L)
        self.w_R = float(w_R)
        self.cost_scale = cost_scale
        self.reputation = None if reputation is None else np.asarray(reputation, dtype=float)
        if self.w_L and not isinstance(cost_scale, Scales):
            raise ValueError("a latency weight needs Scales for l")
        if self.w_R and self.reputation is None:
            raise ValueError("a reputation weight needs each model's R")

    def scores(self, predicted: Mapping[str, np.ndarray]) -> np.ndarray:
        """Scores [rows, models]."""
        scale = self.cost_scale
        if isinstance(scale, Scales):
            score = predicted["quality"] - self.w_C * predicted["cost"] / scale.c - self.w_L * predicted["latency"] / scale.l
        else:
            score = predicted["quality"] - self.w_C * predicted["cost"] / scale
        if self.w_R:
            score = score + self.w_R * self.reputation
        return score

    def route(self, predicted: Mapping[str, np.ndarray]) -> np.ndarray:
        return np.argmax(self.scores(predicted), axis=1)
