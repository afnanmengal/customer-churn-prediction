"""Train churn models and select business thresholds on validation data."""

from typing import Any

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from customer_churn.model import build_models
from customer_churn.thresholds import best_profit_threshold, profit_at_threshold


def run_churn_experiment(
    features: pd.DataFrame,
    labels: pd.Series,
    save_value: float,
    contact_cost: float,
    missed_churn_cost: float,
    random_state: int = 42,
    cv_folds: int = 3,
) -> dict[str, Any]:
    """Compare class weighting and SMOTENC without validation/test leakage."""
    if features.empty or len(features) != len(labels):
        raise ValueError("Features and labels must be non-empty and row-aligned.")
    if set(labels.unique()) != {0, 1}:
        raise ValueError("Labels must contain both binary classes 0 and 1.")
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2.")
    if min(save_value, contact_cost, missed_churn_cost) < 0:
        raise ValueError("Business values and costs must be non-negative.")
    if save_value == contact_cost == missed_churn_cost == 0:
        raise ValueError("At least one business value or cost must be positive.")

    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in features.columns if column not in numeric]
    x_train_validation, x_test, y_train_validation, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=random_state,
    )
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train_validation,
        y_train_validation,
        test_size=0.25,
        stratify=y_train_validation,
        random_state=random_state,
    )
    if y_train.value_counts().min() <= 3:
        raise ValueError("SMOTENC needs at least four training examples in each class.")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    rows: list[dict[str, float | str]] = []
    fitted: dict[str, Any] = {}
    thresholds: dict[str, float] = {}
    validation_profits: dict[str, float] = {}
    test_probabilities: dict[str, pd.Series] = {}
    for name, model in build_models(features.columns, numeric, categorical, random_state).items():
        cv_pr_auc = cross_val_score(
            model,
            x_train,
            y_train,
            cv=cv,
            scoring="average_precision",
            n_jobs=-1,
        )
        model.fit(x_train, y_train)
        validation_scores = model.predict_proba(x_validation)[:, 1]
        threshold, validation_profit = best_profit_threshold(
            y_validation,
            validation_scores,
            save_value,
            contact_cost,
            missed_churn_cost,
        )
        test_scores = model.predict_proba(x_test)[:, 1]
        predictions = test_scores >= threshold
        test_profit = profit_at_threshold(
            y_test,
            test_scores,
            threshold,
            save_value,
            contact_cost,
            missed_churn_cost,
        )
        rows.append(
            {
                "model": name,
                "cv_pr_auc": float(cv_pr_auc.mean()),
                "test_pr_auc": float(average_precision_score(y_test, test_scores)),
                "test_roc_auc": float(roc_auc_score(y_test, test_scores)),
                "test_precision": float(precision_score(y_test, predictions, zero_division=0)),
                "test_recall": float(recall_score(y_test, predictions, zero_division=0)),
                "test_f1": float(f1_score(y_test, predictions, zero_division=0)),
                "threshold": threshold,
                "validation_profit": validation_profit,
                "test_profit": test_profit,
            }
        )
        fitted[name] = model
        thresholds[name] = threshold
        validation_profits[name] = validation_profit
        test_probabilities[name] = pd.Series(test_scores, index=y_test.index, name=name)

    return {
        "benchmark": pd.DataFrame(rows).sort_values("test_pr_auc", ascending=False),
        "models": fitted,
        "thresholds": thresholds,
        "validation_profits": validation_profits,
        "test_probabilities": test_probabilities,
        "features": features,
        "x_test": x_test,
        "y_test": y_test,
    }
