"""Decision-threshold selection from explicit retention economics."""

import numpy as np
from numpy.typing import ArrayLike


def profit_at_threshold(
    y_true: ArrayLike,
    probabilities: ArrayLike,
    threshold: float,
    save_value: float,
    contact_cost: float,
    missed_churn_cost: float,
) -> float:
    """Compute utility when a positive prediction triggers a retention offer."""
    actual = np.asarray(y_true, dtype=int)
    predicted = np.asarray(probabilities, dtype=float) >= threshold
    if actual.shape != predicted.shape:
        raise ValueError("y_true and probabilities must have the same shape")

    true_positive = (predicted & (actual == 1)).sum()
    false_positive = (predicted & (actual == 0)).sum()
    false_negative = (~predicted & (actual == 1)).sum()
    return float(
        true_positive * save_value
        - false_positive * contact_cost
        - false_negative * missed_churn_cost
    )


def best_profit_threshold(
    y_true: ArrayLike,
    probabilities: ArrayLike,
    save_value: float,
    contact_cost: float,
    missed_churn_cost: float,
) -> tuple[float, float]:
    """Return the maximizing threshold and utility on the supplied validation set."""
    scores = np.asarray(probabilities, dtype=float)
    if scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("probabilities must be a non-empty finite array")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("probabilities must be between 0 and 1")

    candidates = np.unique(np.concatenate(([0.0, 1.0], scores)))
    outcomes = [
        profit_at_threshold(y_true, scores, threshold, save_value, contact_cost, missed_churn_cost)
        for threshold in candidates
    ]
    best_index = int(np.argmax(outcomes))
    return float(candidates[best_index]), float(outcomes[best_index])
