"""Class-weighted and SMOTENC churn model pipelines."""

from collections.abc import Sequence

from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def build_models(
    feature_names: Sequence[str],
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    random_state: int = 42,
) -> dict[str, Pipeline | ImbalancedPipeline]:
    """Create a class-weighted baseline and train-fold SMOTENC comparison.

    SMOTENC operates on imputed numeric/categorical columns before one-hot
    encoding, avoiding synthetic fractional one-hot vectors.
    """
    columns = list(feature_names)
    numeric = list(numeric_features)
    categorical = list(categorical_features)
    if not columns or set(numeric) | set(categorical) != set(columns):
        raise ValueError("Numeric and categorical columns must partition the features.")

    weighted_preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                        ),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ],
        sparse_threshold=1.0,
    )
    weighted = Pipeline(
        [
            ("preprocess", weighted_preprocessor),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced", max_iter=2_000, random_state=random_state
                ),
            ),
        ]
    )

    numeric_count = len(numeric)
    categorical_indices = list(range(numeric_count, numeric_count + len(categorical)))
    smote_preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                        ),
                        (
                            "ordinal",
                            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        sparse_threshold=0.0,
    )
    post_smote_encoder = ColumnTransformer(
        [
            ("numeric", "passthrough", list(range(numeric_count))),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_indices),
        ],
        sparse_threshold=1.0,
    )
    smotenc = SMOTENC(
        categorical_features=categorical_indices,
        random_state=random_state,
        k_neighbors=3,
    )
    smote_model = ImbalancedPipeline(
        [
            ("preprocess", smote_preprocessor),
            ("smote", smotenc),
            ("encode", post_smote_encoder),
            (
                "classifier",
                LogisticRegression(max_iter=2_000, random_state=random_state),
            ),
        ]
    )
    return {"class_weighted": weighted, "smotenc": smote_model}
