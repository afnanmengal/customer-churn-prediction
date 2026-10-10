"""Interactive churn benchmark and retention-priority explorer."""

import hashlib
import sys
from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve

PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from customer_churn.data_loader import load_churn_data, split_features_target
from customer_churn.experiment import run_churn_experiment


st.set_page_config(page_title="Churn Retention Lab", layout="wide")
st.title("Churn Retention Lab")
st.caption(
    "Compare imbalance strategies and prioritize outreach using explicit retention economics. "
    "Predicted risk does not establish that an intervention will prevent churn."
)

default_dataset = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
uploaded_file = st.file_uploader("Customer CSV", type=["csv"])
if uploaded_file is not None:
    contents = uploaded_file.getvalue()
    data_signature = hashlib.sha256(contents).hexdigest()
    try:
        data = load_churn_data(BytesIO(contents))
        source_label = uploaded_file.name
    except ValueError as error:
        st.error(str(error))
        st.stop()
elif default_dataset.exists():
    data_signature = f"{default_dataset.stat().st_mtime_ns}:{default_dataset.stat().st_size}"
    try:
        data = load_churn_data(default_dataset)
        source_label = "Kaggle Telco Customer Churn"
    except ValueError as error:
        st.error(str(error))
        st.stop()
else:
    st.info("Upload a customer CSV with a Yes/No Churn column to begin.")
    st.stop()

features, labels = split_features_target(data)
st.sidebar.header("Retention economics")
save_value = st.sidebar.number_input(
    "Value of a retained churn-risk customer",
    min_value=0.0,
    value=250.0,
    step=25.0,
)
contact_cost = st.sidebar.number_input(
    "Cost per retention contact", min_value=0.0, value=10.0, step=1.0
)
missed_churn_cost = st.sidebar.number_input(
    "Cost of missing a churner", min_value=0.0, value=200.0, step=25.0
)
cv_folds = st.sidebar.select_slider("Stratified CV folds", options=[3, 4, 5], value=3)
experiment_signature = (
    f"{data_signature}:{save_value}:{contact_cost}:{missed_churn_cost}:{cv_folds}"
)
if st.session_state.get("churn_experiment_signature") != experiment_signature:
    st.session_state.pop("churn_results", None)
    st.session_state["churn_experiment_signature"] = experiment_signature

summary = st.columns(4)
summary[0].metric("Customers", f"{len(data):,}")
summary[1].metric("Predictors", f"{features.shape[1]:,}")
summary[2].metric("Churn rate", f"{labels.mean():.1%}")
summary[3].metric("Missing predictor cells", f"{int(features.isna().sum().sum()):,}")
st.caption(
    f"Dataset: {source_label}. `customerID` is excluded; blank `TotalCharges` values "
    "are imputed within training folds."
)

with st.expander("Dataset preview", expanded=False):
    st.dataframe(data.head(12), use_container_width=True)

if st.button("Compare class weighting and SMOTENC", type="primary"):
    if save_value == 0 and contact_cost == 0 and missed_churn_cost == 0:
        st.error("Set at least one non-zero retention value or cost.")
        st.stop()
    with st.spinner("Fitting both models with stratified cross-validation…"):
        try:
            st.session_state["churn_results"] = run_churn_experiment(
                features,
                labels,
                save_value=save_value,
                contact_cost=contact_cost,
                missed_churn_cost=missed_churn_cost,
                random_state=42,
                cv_folds=cv_folds,
            )
        except ValueError as error:
            st.error(str(error))
            st.stop()

results = st.session_state.get("churn_results")
if results:
    benchmark = results["benchmark"]
    st.subheader("Held-out benchmark")
    st.dataframe(
        benchmark.style.format(
            {
                "cv_pr_auc": "{:.3f}",
                "test_pr_auc": "{:.3f}",
                "test_roc_auc": "{:.3f}",
                "test_precision": "{:.3f}",
                "test_recall": "{:.3f}",
                "test_f1": "{:.3f}",
                "threshold": "{:.3f}",
                "validation_profit": "{:,.0f}",
                "test_profit": "{:,.0f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Thresholds are chosen on the validation split. Test utility is reported only "
        "for comparison and does not select the model or threshold."
    )

    curve = go.Figure()
    for name, probabilities in results["test_probabilities"].items():
        precision, recall, _ = precision_recall_curve(results["y_test"], probabilities)
        curve.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=name))
    curve.update_layout(
        title="Held-out precision-recall curves",
        xaxis_title="Recall",
        yaxis_title="Precision",
        yaxis_range=[0, 1],
        xaxis_range=[0, 1],
        legend_title="Training strategy",
    )
    st.plotly_chart(curve, use_container_width=True)

    st.subheader("Prioritize a customer")
    selected_model = st.selectbox("Scoring model", list(results["models"]))
    numeric_features = features.select_dtypes(include="number").columns.tolist()
    categorical_features = [column for column in features.columns if column not in numeric_features]
    numeric_medians = features[numeric_features].median(numeric_only=True)
    categorical_modes = {
        column: features[column].mode(dropna=True).iloc[0]
        for column in categorical_features
        if not features[column].mode(dropna=True).empty
    }
    customer_values: dict[str, float | int | str] = {}
    with st.form("customer_prediction"):
        input_columns = st.columns(3)
        for index, column in enumerate(features.columns):
            with input_columns[index % 3]:
                if column in categorical_features:
                    choices = sorted(features[column].dropna().astype(str).unique().tolist())
                    mode = str(categorical_modes.get(column, choices[0]))
                    default_index = choices.index(mode) if mode in choices else 0
                    customer_values[column] = st.selectbox(
                        column,
                        choices,
                        index=default_index,
                        key=f"churn_input_{column}",
                    )
                else:
                    customer_values[column] = st.number_input(
                        column,
                        value=float(numeric_medians[column]),
                        key=f"churn_input_{column}",
                    )
        score_clicked = st.form_submit_button("Estimate churn risk")

    if score_clicked:
        customer = pd.DataFrame([customer_values], columns=features.columns)
        probability = float(results["models"][selected_model].predict_proba(customer)[0, 1])
        threshold = results["thresholds"][selected_model]
        decision = "Prioritize for review" if probability >= threshold else "Below selected cutoff"
        score_columns = st.columns(3)
        score_columns[0].metric("Estimated churn probability", f"{probability:.1%}")
        score_columns[1].metric("Validation cutoff", f"{threshold:.1%}")
        score_columns[2].metric("Outreach queue", decision)

st.divider()
st.caption(
    "Retention values are user assumptions, not observed causal effects. Validate "
    "calibration, subgroup behavior, and intervention efficacy before operational use."
)
