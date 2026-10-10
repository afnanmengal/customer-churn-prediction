import pandas as pd
import pytest

from customer_churn.data_loader import load_churn_data, split_features_target


def test_churn_loader_converts_blank_total_charges_and_labels(tmp_path) -> None:
    path = tmp_path / "telco.csv"
    pd.DataFrame(
        {
            "customerID": ["a", "b", "c", "d"],
            "tenure": [0, 3, 5, 7],
            "TotalCharges": ["", "10.5", "20", "30"],
            "Churn": ["No", "Yes", "No", "Yes"],
        }
    ).to_csv(path, index=False)

    frame = load_churn_data(path)
    features, labels = split_features_target(frame)

    assert labels.tolist() == [0, 1, 0, 1]
    assert features["TotalCharges"].isna().sum() == 1
    assert "customerID" not in features


def test_churn_loader_rejects_unknown_target_label(tmp_path) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame({"feature": [1, 2], "Churn": ["No", "Maybe"]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="Yes/No"):
        load_churn_data(path)
