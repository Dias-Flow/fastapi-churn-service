import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from app.ml.training import build_churn_pipeline, train_churn_model


def test_build_churn_pipeline_returns_pipeline():
    """
    build_churn_pipeline should return sklearn Pipeline and hyperparameters.
    """

    pipeline, hyperparameters = build_churn_pipeline(
        model_type="logreg",
        hyperparameters={},
        random_state=42,
    )

    assert isinstance(pipeline, Pipeline)
    assert "preprocessor" in pipeline.named_steps
    assert "model" in pipeline.named_steps
    assert hyperparameters["max_iter"] == 1000
    assert hyperparameters["random_state"] == 42


def test_build_churn_pipeline_contains_imputers():
    """
    The preprocessing pipeline should contain imputers for missing values.
    """

    pipeline, _ = build_churn_pipeline(
        model_type="logreg",
        hyperparameters={},
        random_state=42,
    )

    preprocessor = pipeline.named_steps["preprocessor"]

    numeric_pipeline = preprocessor.transformers[0][1]
    categorical_pipeline = preprocessor.transformers[1][1]

    assert "imputer" in numeric_pipeline.named_steps
    assert "scaler" in numeric_pipeline.named_steps

    assert "imputer" in categorical_pipeline.named_steps
    assert "encoder" in categorical_pipeline.named_steps


def test_build_churn_pipeline_supports_random_forest():
    """
    build_churn_pipeline should support random_forest model type.
    """

    pipeline, hyperparameters = build_churn_pipeline(
        model_type="random_forest",
        hyperparameters={"n_estimators": 10},
        random_state=42,
    )

    assert isinstance(pipeline, Pipeline)
    assert "preprocessor" in pipeline.named_steps
    assert "model" in pipeline.named_steps
    assert hyperparameters["n_estimators"] == 10
    assert hyperparameters["random_state"] == 42


def test_build_churn_pipeline_rejects_unknown_model_type():
    """
    Unknown model types should raise ValueError.
    """

    try:
        build_churn_pipeline(
            model_type="unknown_model",
            hyperparameters={},
            random_state=42,
        )
    except ValueError as error:
        assert "Unsupported model_type" in str(error)
    else:
        raise AssertionError("Expected ValueError for unsupported model_type")


def test_train_churn_model_returns_metrics():
    """
    train_churn_model should train a model and return metrics.
    """

    result = train_churn_model(
        test_size=0.2,
        random_state=42,
        model_type="logreg",
        hyperparameters={},
    )

    assert "model" in result
    assert "accuracy" in result
    assert "f1" in result
    assert "trained_at" in result
    assert "train_rows" in result
    assert "test_rows" in result

    assert 0 <= result["accuracy"] <= 1
    assert 0 <= result["f1"] <= 1
    assert result["train_rows"] > 0
    assert result["test_rows"] > 0

def test_churn_pipeline_handles_missing_feature_values():
    """
    The training pipeline should handle missing feature values.

    Missing values are processed by SimpleImputer inside the preprocessing step.
    """

    X = pd.DataFrame(
        [
            {
                "monthly_fee": 29.99,
                "usage_hours": 12.5,
                "support_requests": 2,
                "account_age_months": 8,
                "failed_payments": 1,
                "autopay_enabled": 1,
                "region": "europe",
                "device_type": "mobile",
                "payment_method": "card",
            },
            {
                "monthly_fee": np.nan,
                "usage_hours": 20.0,
                "support_requests": np.nan,
                "account_age_months": 12,
                "failed_payments": 0,
                "autopay_enabled": 0,
                "region": np.nan,
                "device_type": "desktop",
                "payment_method": "paypal",
            },
            {
                "monthly_fee": 55.0,
                "usage_hours": np.nan,
                "support_requests": 0,
                "account_age_months": 24,
                "failed_payments": 0,
                "autopay_enabled": 1,
                "region": "asia",
                "device_type": np.nan,
                "payment_method": "card",
            },
            {
                "monthly_fee": 10.0,
                "usage_hours": 2.0,
                "support_requests": 4,
                "account_age_months": np.nan,
                "failed_payments": 2,
                "autopay_enabled": np.nan,
                "region": "america",
                "device_type": "tablet",
                "payment_method": np.nan,
            },
        ]
    )

    y = pd.Series([0, 1, 0, 1])

    pipeline, _ = build_churn_pipeline(
        model_type="logreg",
        hyperparameters={},
        random_state=42,
    )

    pipeline.fit(X, y)
    predictions = pipeline.predict(X)

    assert len(predictions) == len(X)