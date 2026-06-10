import pandas as pd
from sklearn.model_selection import train_test_split

from app.core.config import DATASET_PATH
from app.ml.features import (
    ALLOWED_CATEGORIES,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
)


def load_churn_dataset() -> pd.DataFrame:
    """
    Load the churn dataset from a CSV file.

    The dataset is validated structurally before training.
    Missing feature values are allowed because they are handled later
    by SimpleImputer inside the sklearn preprocessing pipeline.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If the dataset has invalid structure or invalid values.
    """

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset file was not found at path: {DATASET_PATH}"
        )

    dataframe = pd.read_csv(DATASET_PATH)
    validate_churn_dataset(dataframe)
    dataframe = normalize_churn_dataset(dataframe)

    return dataframe


def validate_churn_dataset(dataframe: pd.DataFrame) -> None:
    """
    Validate churn dataset structure and non-missing values.

    This function checks:
    - dataset is not empty
    - all required columns exist
    - there are no unexpected columns
    - target column has no missing values
    - target values are only 0 or 1
    - numeric feature values are numeric when present
    - numeric feature values satisfy basic constraints when present
    - categorical feature values are allowed when present

    Important:
        Missing values in feature columns are allowed here.
        They are handled by SimpleImputer in the preprocessing pipeline.
    """

    if dataframe.empty:
        raise ValueError("Dataset is empty.")

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    unexpected_columns = [
        column
        for column in dataframe.columns
        if column not in REQUIRED_COLUMNS
    ]

    if unexpected_columns:
        raise ValueError(
            f"Dataset contains unexpected columns: {unexpected_columns}"
        )

    if dataframe[TARGET_COLUMN].isna().any():
        raise ValueError("Target column contains missing values.")

    target_values = pd.to_numeric(
        dataframe[TARGET_COLUMN],
        errors="coerce",
    )

    if target_values.isna().any():
        raise ValueError("Target column must contain numeric values.")

    if not target_values.isin([0, 1]).all():
        raise ValueError("Target column must contain only 0 and 1 values.")

    for column in NUMERIC_FEATURES:
        non_missing_values = dataframe[column].dropna()

        numeric_values = pd.to_numeric(
            non_missing_values,
            errors="coerce",
        )

        if numeric_values.isna().any():
            raise ValueError(f"Column {column} must contain numeric values.")

        if column == "monthly_fee" and (numeric_values <= 0).any():
            raise ValueError("Column monthly_fee must contain positive values.")

        if column != "monthly_fee" and (numeric_values < 0).any():
            raise ValueError(f"Column {column} must contain non-negative values.")

        if column in {
            "support_requests",
            "account_age_months",
            "failed_payments",
            "autopay_enabled",
        }:
            has_fractional_values = (numeric_values % 1 != 0).any()

            if has_fractional_values:
                raise ValueError(f"Column {column} must contain integer values.")

        if column == "autopay_enabled":
            invalid_autopay_values = set(numeric_values.astype(int).unique()) - {0, 1}

            if invalid_autopay_values:
                raise ValueError("Column autopay_enabled must contain only 0 and 1 values.")

    for column in CATEGORICAL_FEATURES:
        non_missing_values = dataframe[column].dropna()
        allowed_values = set(ALLOWED_CATEGORIES[column])
        actual_values = set(non_missing_values.unique())

        invalid_values = actual_values - allowed_values

        if invalid_values:
            raise ValueError(
                f"Column {column} contains invalid categories: {sorted(invalid_values)}"
            )

def normalize_churn_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize dataset column types after validation.

    Missing feature values are preserved for SimpleImputer.
    Target values are converted to integers because classifiers and
    prediction logic expect classes 0 and 1.
    """

    normalized_dataframe = dataframe.copy()

    for column in NUMERIC_FEATURES:
        normalized_dataframe[column] = pd.to_numeric(
            normalized_dataframe[column],
            errors="coerce",
        )

    normalized_dataframe[TARGET_COLUMN] = pd.to_numeric(
        normalized_dataframe[TARGET_COLUMN],
        errors="coerce",
    ).astype(int)

    return normalized_dataframe

def get_dataset_preview(limit: int = 5) -> list[dict]:
    """
    Return the first N rows from the churn dataset.
    """

    dataframe = load_churn_dataset()

    return dataframe.head(limit).to_dict(orient="records")


def get_dataset_info() -> dict:
    """
    Return basic information about the churn dataset.
    """

    dataframe = load_churn_dataset()

    return {
        "rows": int(dataframe.shape[0]),
        "columns": int(dataframe.shape[1]),
        "feature_names": FEATURE_COLUMNS,
        "target_name": TARGET_COLUMN,
        "churn_distribution": get_class_distribution(dataframe[TARGET_COLUMN]),
    }


def split_features_and_target(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split the dataset into input features X and target variable y.
    """

    X = dataframe[FEATURE_COLUMNS]
    y = dataframe[TARGET_COLUMN]

    return X, y


def create_train_test_split(
    dataframe: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into train and test parts.

    Stratification is used to keep churn class proportions similar
    in both train and test datasets.
    """

    X, y = split_features_and_target(dataframe)

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def get_class_distribution(target: pd.Series) -> dict[int, int]:
    """
    Count how many examples belong to each target class.
    """

    return {
        int(class_value): int(count)
        for class_value, count in target.value_counts().sort_index().items()
    }


def get_split_info(
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Return information about train/test split.
    """

    dataframe = load_churn_dataset()
    X_train, X_test, y_train, y_test = create_train_test_split(
        dataframe=dataframe,
        test_size=test_size,
        random_state=random_state,
    )

    return {
        "total_rows": int(dataframe.shape[0]),
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "test_size": test_size,
        "random_state": random_state,
        "train_churn_distribution": get_class_distribution(y_train),
        "test_churn_distribution": get_class_distribution(y_test),
    }