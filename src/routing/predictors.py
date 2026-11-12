"""Per-model guesses of quality, cost and latency from the query embedding.

For each model, one random forest guesses the quality score and one guesses the
cost (USD); when the data has latency (the live answers of Stage 3; RouterBench
has none), one more guesses the latency (seconds). They are fitted on offline
rows, where every model's outcome is known, and stay fixed online. MixLLM
(Wang et al., NAACL 2025) also guesses quality with a random forest.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np
from sklearn.ensemble import RandomForestRegressor


class Predictors:
    def __init__(self, models: Sequence[str], forest: Mapping[str, Any], seed: int):
        self.models = list(models)
        self.forest = dict(forest)
        self.seed = seed

    def _fit_forest(self, X: np.ndarray, y: np.ndarray) -> RandomForestRegressor:
        forest = RandomForestRegressor(random_state=self.seed, n_jobs=-1, **self.forest).fit(X, y)
        # Predict on one thread: routing time is measured per query, and starting
        # worker threads would cost more than the prediction itself.
        forest.n_jobs = 1
        return forest

    def fit(self, X: np.ndarray, outcomes: Mapping[str, np.ndarray]) -> "Predictors":
        """Fit on offline rows; ``outcomes`` holds the real quality, cost and, if known, latency, each [rows, models]."""
        self.quality_ = [self._fit_forest(X, outcomes["quality"][:, j]) for j in range(len(self.models))]
        self.cost_ = [self._fit_forest(X, outcomes["cost"][:, j]) for j in range(len(self.models))]
        self.latency_ = (
            [self._fit_forest(X, outcomes["latency"][:, j]) for j in range(len(self.models))]
            if "latency" in outcomes
            else []
        )
        # The router divides cost by this: the average cost over all models.
        self.cost_scale_ = float(outcomes["cost"].mean())
        return self

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Guessed quality, cost (USD) and, if learned, latency (seconds), each [rows, models]."""
        guess = {
            "quality": np.column_stack([forest.predict(X) for forest in self.quality_]),
            "cost": np.column_stack([forest.predict(X) for forest in self.cost_]),
        }
        if self.latency_:
            guess["latency"] = np.column_stack([forest.predict(X) for forest in self.latency_])
        return guess
