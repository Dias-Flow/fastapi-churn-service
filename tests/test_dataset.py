from app.ml.data import (
    create_train_test_split,
    get_dataset_info,
    get_dataset_preview,
    load_churn_dataset,
)
from app.ml.features import FEATURE_COLUMNS, TARGET_COLUMN


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