# Customer Churn Prediction

Rank customers for retention outreach under class imbalance. The Streamlit app compares class weighting with SMOTENC, tunes a decision cutoff against explicit business assumptions, plots held-out precision-recall curves, and scores an example customer profile.

## Dataset

- [Telco Customer Churn on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- IBM sample data; 7,043 customers; target `Churn` (`Yes` = 1).
- Kaggle lists the data files as © original authors. The CSV is never committed; check original terms before redistribution or reuse.
- `customerID` is excluded. Empty `TotalCharges` entries are parsed as missing and imputed within training folds.

## Method and Results

Compare class-weighted logistic regression with SMOTENC performed inside imbalanced-learn training/CV folds. Thresholds maximize validation utility; held-out test data is reserved for final comparison. PR-AUC is emphasized for the imbalanced target.

Seed `42`, stratified 60/20/20 split, 3-fold CV. Example economics: retained customer value 250, contact cost 10, missed-churn cost 200. These are illustrative assumptions, not measured causal intervention effects.

| Model | CV PR-AUC | Test PR-AUC | Test ROC-AUC | Precision | Recall | Test utility |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Class-weighted logistic | 0.662 | 0.633 | 0.843 | 0.312 | 0.995 | 84,380 |
| SMOTENC logistic | 0.661 | 0.619 | 0.836 | 0.306 | 0.995 | 84,150 |

Validate calibration, subgroup behavior, and intervention efficacy before operational use. Scores are not proof that an offer will save a customer.

## Setup and Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/download_dataset.py
streamlit run app.py
```

The app opens at `http://localhost:8501`. Run tests with `python -m pytest`; lint with `python -m ruff check .`.

## Repository Layout

`app.py` is the Streamlit interface; `src/customer_churn/` contains validation, SMOTENC/class-weighted pipelines, threshold selection, and experiment logic; `tests/` covers behavior; `configs/` stores experiment settings. Data and model files are excluded from Git.