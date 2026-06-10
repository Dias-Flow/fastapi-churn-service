import pandas as pd
from pydantic import ValidationError
from sklearn.model_selection import train_test_split

from app.core.config import DATASET_PATH
from app.ml.features import FEATURE_COLUMNS, REQUIRED_COLUMNS, TARGET_COLUMN
from app.schemas.churn import DatasetRowChurn


def load_churn_dataset() -> pd.DataFrame:
    """
    Load the churn dataset from a CSV file.

    Returns:
        A pandas DataFrame with the dataset.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If the dataset has invalid structure or invalid rows.
    """

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset file was not found at path: {DATASET_PATH}"
        )

    dataframe = pd.read_csv(DATASET_PATH)
    validate_churn_dataset(dataframe)

    return dataframe


def validate_churn_dataset(dataframe: pd.DataFrame) -> None:
    """
    Validate that the dataset has the required structure.

    The validation checks:
    - dataset is not empty
    - all required columns exist
    - there are no unexpected columns
    - every row matches DatasetRowChurn schema

    Raises:
        ValueError: If the dataset is invalid.
    """

    if dataframe.empty:
        raise ValueError("Dataset is empty.")

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    unexpected_columns = [
        column for column in dataframe.columns
        if column not in REQUIRED_COLUMNS
    ]

    if unexpected_columns:
        raise ValueError(
            f"Dataset contains unexpected columns: {unexpected_columns}"
        )

    try:
        for row in dataframe[REQUIRED_COLUMNS].to_dict(orient="records"):
            DatasetRowChurn.model_validate(row)

    except ValidationError as error:
        raise ValueError(
            "Dataset contains rows that do not match DatasetRowChurn schema."
        ) from error


def get_dataset_preview(limit: int = 5) -> list[dict]:
    """
    Return the first N rows from the churn dataset.

    Args:
        limit: Number of rows to return.

    Returns:
        A list of dataset rows represented as dictionaries.
    """

    dataframe = load_churn_dataset()

    return dataframe.head(limit).to_dict(orient="records")


def get_dataset_info() -> dict:
    """
    Return basic information about the churn dataset.

    Returns:
        A dictionary with rows count, columns count, feature names,
        target name, and churn class distribution.
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

    Args:
        dataframe: Full churn dataset.

    Returns:
        X: Feature matrix.
        y: Target vector.
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