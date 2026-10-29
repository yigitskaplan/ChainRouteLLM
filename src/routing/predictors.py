"""Per-model guesses of quality and cost from the query embedding.

For each model, one random forest guesses the quality score and one guesses the
cost (USD). They are fitted on offline rows, where every model's outcome is
known; online, online.py retrains them with the answers the router sees.
MixLLM (Wang et al., NAACL 2025) also guesses quality with a random forest.
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
        """Fit on offline rows; ``outcomes`` holds the real quality and cost, each [rows, models]."""
        self.quality_ = [self._fit_forest(X, outcomes["quality"][:, j]) for j in range(len(self.models))]
        self.cost_ = [self._fit_forest(X, outcomes["cost"][:, j]) for j in range(len(self.models))]
        # The router divides cost by this: the average cost over all models.
        self.cost_scale_ = float(outcomes["cost"].mean())
        return self

    def refit_model(self, j: int, X: np.ndarray, quality: np.ndarray, cost: np.ndarray) -> None:
        """Retrain model j's two forests, e.g. on the offline answers plus its new online ones."""
        self.quality_[j] = self._fit_forest(X, quality)
        self.cost_[j] = self._fit_forest(X, cost)

    def copy(self) -> "Predictors":
        """A copy whose forests can be retrained without changing this one's."""
        other = Predictors(self.models, self.forest, self.seed)
        other.quality_, other.cost_, other.cost_scale_ = list(self.quality_), list(self.cost_), self.cost_scale_
        return other

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Guessed quality and cost (USD), each [rows, models]."""
        return {
            "quality": np.column_stack([forest.predict(X) for forest in self.quality_]),
            "cost": np.column_stack([forest.predict(X) for forest in self.cost_]),
        }
