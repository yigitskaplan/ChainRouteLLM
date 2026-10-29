"""Online learning: the router keeps learning from the answers it sees.

Online questions arrive one by one. The router picks a model for each, and only
that model's answer is seen: its real accuracy and cost (MixLLM's "refined
feedback" from the selected LLM). Every ``update_interval`` questions, each
model that got new answers has its two forests retrained on the offline answers
plus all its new ones. Between retrainings the guessers do not change, so a
block of questions can be routed at once.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from src.routing.predictors import Predictors
from src.routing.router import Router


def learn_online(
    predictors: Predictors,
    router: Router,
    X_offline: np.ndarray,
    offline: Mapping[str, np.ndarray],
    X_online: np.ndarray,
    online: Mapping[str, np.ndarray],
    update_interval: int,
) -> Predictors:
    """Route the online questions in order, learning from each pick; returns the updated guessers.

    ``offline`` and ``online`` hold the real quality and cost, each [rows, models];
    of the online ones only the picked model's are used. ``predictors`` itself is not changed.
    """
    predictors = predictors.copy()
    seen = [[] for _ in predictors.models]  # online rows answered by each model
    for start in range(0, len(X_online), update_interval):
        rows = np.arange(start, min(start + update_interval, len(X_online)))
        picks = router.route(predictors.predict(X_online[rows]))
        for j in np.unique(picks):
            seen[j].extend(rows[picks == j])
            new = np.asarray(seen[j])
            predictors.refit_model(
                j,
                np.vstack([X_offline, X_online[new]]),
                np.concatenate([offline["quality"][:, j], online["quality"][new, j]]),
                np.concatenate([offline["cost"][:, j], online["cost"][new, j]]),
            )
    return predictors
