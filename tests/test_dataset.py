import pandas as pd
import numpy as np
import pytest

from app.ml.data import (
    create_train_test_split,
    get_dataset_info,
    get_dataset_preview,
    load_churn_dataset,
    validate_churn_dataset,
)
from app.ml.features import FEATURE_COLUMNS, TARGET_COLUMN


VALID_DATASET_ROW = {
    "monthly_fee": 29.99,
    "usage_hours": 12.5,
    "support_requests": 2,
    "account_age_months": 8,
    "failed_payments": 1,
    "autopay_enabled": 1,
    "region": "europe",
    "device_type": "mobile",
    "payment_method": "card",
    "churn": 0,
}


def test_load_churn_dataset_returns_dataframe():
    """
    Dataset should load successfully from data/churn_dataset.csv.
    """

    dataframe = load_churn_dataset()

    assert dataframe.empty is False
    assert TARGET_COLUMN in dataframe.columns

    for column in FEATURE_COLUMNS:
        assert column in dataframe.columns


def test_get_dataset_preview_returns_requested_number_of_rows():
    """
    Dataset preview should return exactly the requested number of rows.
    """

    preview = get_dataset_preview(limit=3)

    assert isinstance(preview, list)
    assert len(preview) == 3


def test_get_dataset_info_contains_basic_information():
    """
    Dataset info should contain rows, columns, features, target and distribution.
    """

    info = get_dataset_info()

    assert info["rows"] > 0
    assert info["columns"] == len(FEATURE_COLUMNS) + 1
    assert info["feature_names"] == FEATURE_COLUMNS
    assert info["target_name"] == TARGET_COLUMN
    assert 0 in info["churn_distribution"]
    assert 1 in info["churn_distribution"]


def test_create_train_test_split_returns_expected_shapes():
    """
    Train/test split should preserve total number of rows.
    """

    dataframe = load_churn_dataset()

    X_train, X_test, y_train, y_test = create_train_test_split(
        dataframe=dataframe,
        test_size=0.2,
        random_state=42,
    )

    assert len(X_train) + len(X_test) == len(dataframe)
    assert len(y_train) + len(y_test) == len(dataframe)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)


def test_validate_churn_dataset_accepts_valid_dataframe():
    """
    Valid dataset rows should pass validation.
    """

    dataframe = pd.DataFrame([VALID_DATASET_ROW])

    validate_churn_dataset(dataframe)


def test_validate_churn_dataset_rejects_missing_column():
    """
    Dataset without a required column should be rejected.
    """

    row = VALID_DATASET_ROW.copy()
    row.pop("monthly_fee")

    dataframe = pd.DataFrame([row])

    with pytest.raises(ValueError, match="missing required columns"):
        validate_churn_dataset(dataframe)


def test_validate_churn_dataset_rejects_unexpected_column():
    """
    Dataset with unexpected columns should be rejected.
    """

    row = VALID_DATASET_ROW.copy()
    row["unexpected_column"] = "not allowed"

    dataframe = pd.DataFrame([row])

    with pytest.raises(ValueError, match="unexpected columns"):
        validate_churn_dataset(dataframe)


def test_validate_churn_dataset_rejects_invalid_category():
    """
    Dataset rows with invalid categorical values should be rejected.
    """

    row = VALID_DATASET_ROW.copy()
    row["region"] = "moon"

    dataframe = pd.DataFrame([row])

    with pytest.raises(ValueError) as error:
        validate_churn_dataset(dataframe)

    assert "Column region contains invalid categories" in str(error.value)


def test_validate_churn_dataset_rejects_invalid_target():
    """
    Dataset rows with invalid churn values should be rejected.
    """

    row = VALID_DATASET_ROW.copy()
    row["churn"] = 7

    dataframe = pd.DataFrame([row])

    with pytest.raises(ValueError) as error:
        validate_churn_dataset(dataframe)

    assert "Target column must contain only 0 and 1 values" in str(error.value)


def test_validate_churn_dataset_rejects_invalid_numeric_value():
    """
    Dataset rows with invalid numeric constraints should be rejected.
    """

    row = VALID_DATASET_ROW.copy()
    row["monthly_fee"] = -10

    dataframe = pd.DataFrame([row])

    with pytest.raises(ValueError) as error:
        validate_churn_dataset(dataframe)

    assert "Column monthly_fee must contain positive values" in str(error.value)

def test_validate_churn_dataset_allows_missing_feature_values():
    """
    Missing feature values should be allowed.

    They are handled later by SimpleImputer in the preprocessing pipeline.
    """

    row = VALID_DATASET_ROW.copy()
    row["monthly_fee"] = np.nan
    row["region"] = np.nan
    row["support_requests"] = np.nan

    dataframe = pd.DataFrame([row])

    validate_churn_dataset(dataframe)

def test_validate_churn_dataset_rejects_missing_target():
    """
    Missing target values should be rejected.

    The target column is required for supervised training.
    """

    row = VALID_DATASET_ROW.copy()
    row["churn"] = np.nan

    dataframe = pd.DataFrame([row])

    with pytest.raises(ValueError, match="Target column contains missing values"):
        validate_churn_dataset(dataframe)

def test_validate_churn_dataset_rejects_fractional_target():
    """
    Dataset rows with fractional churn values should be rejected.
    """

    row = VALID_DATASET_ROW.copy()
    row["churn"] = 0.5

    dataframe = pd.DataFrame([row])

    with pytest.raises(ValueError) as error:
        validate_churn_dataset(dataframe)

    assert "Target column must contain only 0 and 1 values" in str(error.value)