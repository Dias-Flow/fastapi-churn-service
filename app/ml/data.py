import pandas as pd
from sklearn.model_selection import train_test_split

from app.core.config import DATASET_PATH
from app.ml.features import FEATURE_COLUMNS, REQUIRED_COLUMNS, TARGET_COLUMN


def load_churn_dataset() -> pd.DataFrame:
    """
    Load the churn dataset from a CSV file.

    Returns:
        A pandas DataFrame with the dataset.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If the dataset has invalid structure.
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
    Validate that the dataset contains all required columns and is not empty.

    Args:
        dataframe: Dataset loaded as a pandas DataFrame.

    Raises:
        ValueError: If the dataset is empty or required columns are missing.
    """

    if dataframe.empty:
        raise ValueError("Dataset is empty.")

    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )


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

    Args:
        dataframe: Full churn dataset.
        test_size: Fraction of the dataset used for testing.
        random_state: Seed for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test.
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

    Args:
        target: Target values.

    Returns:
        Dictionary where keys are class labels and values are counts.
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

    Args:
        test_size: Fraction of the dataset used for testing.
        random_state: Seed for reproducibility.

    Returns:
        Dictionary with train/test sizes and class distributions.
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