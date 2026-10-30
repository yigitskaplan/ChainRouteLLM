"""Stage 2 online learning: the router keeps learning from the answers it sees.

Online questions arrive one by one. The router picks a model for each, and only
that model's answer is seen: its real accuracy and cost (MixLLM's "refined
feedback" from the selected LLM). Every ``update_interval`` questions, each
model that got new answers has its two forests retrained on the offline answers
plus all its new ones, with the same forest settings as offline. Between
retrainings the guessers do not change, so a block of questions can be routed
at once, and every pick is made before its answer is seen.
"""
from __future__ import annotations

import copy
from typing import Mapping, Tuple

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
) -> Tuple[Predictors, np.ndarray]:
    """Route the online questions in order, learning from each pick.

    Returns the updated guessers and the model picked for every online question.
    ``offline`` and ``online`` hold the real quality and cost, each [rows, models];
    of the online ones only the picked model's are used. ``predictors`` itself is not changed.
    """
    # Work on a copy with its own forest lists, so the offline guessers stay as they are.
    learned = copy.copy(predictors)
    learned.quality_, learned.cost_ = list(predictors.quality_), list(predictors.cost_)
    picks = np.zeros(len(X_online), dtype=int)
    seen = [[] for _ in predictors.models]  # online rows answered by each model
    for start in range(0, len(X_online), update_interval):
        rows = np.arange(start, min(start + update_interval, len(X_online)))
        picks[rows] = router.route(learned.predict(X_online[rows]))
        for j in np.unique(picks[rows]):
            seen[j].extend(rows[picks[rows] == j])
            new = np.asarray(seen[j])
            # Retrain model j's two forests on its offline answers plus all its online ones.
            retrained = Predictors([predictors.models[j]], predictors.forest, predictors.seed).fit(
                np.vstack([X_offline, X_online[new]]),
                {key: np.concatenate([offline[key][:, j], online[key][new, j]])[:, None] for key in ("quality", "cost")},
            )
            learned.quality_[j], learned.cost_[j] = retrained.quality_[0], retrained.cost_[0]
    return learned, picks
