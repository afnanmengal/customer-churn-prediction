import pytest

from customer_churn.thresholds import best_profit_threshold, profit_at_threshold


def test_profit_threshold_accounts_for_missed_churn():
    threshold, profit = best_profit_threshold(
        [1, 1, 0, 0],
        [0.9, 0.4, 0.6, 0.1],
        save_value=100,
        contact_cost=10,
        missed_churn_cost=50,
    )

    assert threshold == pytest.approx(0.4)
    assert profit == 190


def test_profit_rejects_mismatched_inputs():
    with pytest.raises(ValueError, match="same shape"):
        profit_at_threshold([1], [0.2, 0.3], 0.5, 10, 1, 2)
