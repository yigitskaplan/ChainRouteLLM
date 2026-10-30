"""Router: send each query to the model with the best guessed quality for its cost.

Score of model m for a query:

    s_m = A_m - w_C * C_m / c

A and C are the guessed quality and cost (predictors.py); c is the mean offline
cost over all models, so w_C is in quality points per "one average answer's
cost". The router picks argmax_m s_m. It does not learn online yet.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np


class Router:
    def __init__(self, w_C: float, cost_scale: float):
        self.w_C = float(w_C)
        self.cost_scale = cost_scale

    def scores(self, predicted: Mapping[str, np.ndarray]) -> np.ndarray:
        """Scores [rows, models]."""
        return predicted["quality"] - self.w_C * predicted["cost"] / self.cost_scale

    def route(self, predicted: Mapping[str, np.ndarray]) -> np.ndarray:
        return np.argmax(self.scores(predicted), axis=1)
