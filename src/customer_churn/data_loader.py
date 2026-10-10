"""Validated ingestion for Telco customer churn data."""

from pathlib import Path
from typing import IO

import pandas as pd

TARGET = "Churn"
IDENTIFIER = "customerID"


def load_churn_data(source: str | Path | IO[bytes] | IO[str]) -> pd.DataFrame:
    """Load the Telco CSV, normalize labels, and coerce TotalCharges blanks to missing."""
    try:
        frame = pd.read_csv(source)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, ValueError) as error:
        raise ValueError(f"Could not read churn CSV: {error}") from error
    if frame.empty or TARGET not in frame.columns:
        raise ValueError(f"Churn data must be non-empty and contain {TARGET!r}.")
    if frame[TARGET].isna().any():
        raise ValueError("Churn target contains missing labels.")
    labels = set(frame[TARGET].astype(str).str.strip().unique())
    if labels != {"Yes", "No"}:
        raise ValueError(f"Churn labels must be Yes/No; found {sorted(labels)}.")
    if "TotalCharges" in frame.columns:
        frame["TotalCharges"] = pd.to_numeric(
            frame["TotalCharges"].astype(str).str.strip().replace("", pd.NA),
            errors="coerce",
        )
    frame[TARGET] = frame[TARGET].astype(str).str.strip().map({"No": 0, "Yes": 1}).astype("int8")
    return frame


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Remove the customer identifier and return a binary churn label."""
    if TARGET not in frame.columns or frame.empty:
        raise ValueError(f"A non-empty frame with {TARGET!r} is required.")
    features = frame.drop(columns=[TARGET, IDENTIFIER], errors="ignore").copy()
    if features.empty:
        raise ValueError("No churn predictors remain after excluding target and identifier.")
    return features, frame[TARGET].astype("int8").copy()
