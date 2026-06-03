from datetime import datetime, timezone
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from app.ml.data import create_train_test_split, get_class_distribution, load_churn_dataset
from app.ml.preprocessing import build_preprocessor


def build_classifier_model(
    model_type: str,
    hyperparameters: dict[str, Any] | None,
    random_state: int,
):
    """
    Build a classifier model by model type.

    Supported model types:
    - logreg
    - random_forest

    Args:
        model_type: Name of the classifier.
        hyperparameters: Custom model hyperparameters.
        random_state: Random seed for reproducibility.

    Returns:
        A tuple with the model object and the final hyperparameters used.

    Raises:
        ValueError: If the model type is not supported.
    """

    hyperparameters = hyperparameters or {}

    if model_type == "logreg":
        default_hyperparameters = {
            "max_iter": 1000,
            "random_state": random_state,
        }

        final_hyperparameters = {
            **default_hyperparameters,
            **hyperparameters,
        }

        return LogisticRegression(**final_hyperparameters), final_hyperparameters

    if model_type == "random_forest":
        default_hyperparameters = {
            "n_estimators": 100,
            "random_state": random_state,
            "class_weight": "balanced",
        }

        final_hyperparameters = {
            **default_hyperparameters,
            **hyperparameters,
        }

        return RandomForestClassifier(**final_hyperparameters), final_hyperparameters

    raise ValueError(
        f"Unsupported model_type: {model_type}. "
        "Supported values are: logreg, random_forest."
    )


def build_churn_pipeline(
    model_type: str = "logreg",
    hyperparameters: dict[str, Any] | None = None,
    random_state: int = 42,
) -> tuple[Pipeline, dict[str, Any]]:
    """
    Build an ML pipeline for churn classification.

    The pipeline contains two steps:
    1. Preprocessing
    2. Classifier model

    Args:
        model_type: Type of classifier to use.
        hyperparameters: Custom classifier hyperparameters.
        random_state: Random seed for reproducibility.

    Returns:
        A tuple with the pipeline and final model hyperparameters.
    """

    preprocessor = build_preprocessor()

    model, final_hyperparameters = build_classifier_model(
        model_type=model_type,
        hyperparameters=hyperparameters,
        random_state=random_state,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline, final_hyperparameters


def train_churn_model(
    test_size: float = 0.2,
    random_state: int = 42,
    model_type: str = "logreg",
    hyperparameters: dict[str, Any] | None = None,
) -> dict:
    """
    Train the churn classification model and calculate basic metrics.

    Args:
        test_size: Fraction of the dataset used for testing.
        random_state: Seed for reproducibility.
        model_type: Type of classifier to train.
        hyperparameters: Custom model hyperparameters.

    Returns:
        Dictionary with model, metrics, and dataset information.
    """

    dataframe = load_churn_dataset()

    X_train, X_test, y_train, y_test = create_train_test_split(
        dataframe=dataframe,
        test_size=test_size,
        random_state=random_state,
    )

    pipeline, final_hyperparameters = build_churn_pipeline(
        model_type=model_type,
        hyperparameters=hyperparameters,
        random_state=random_state,
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    trained_at = datetime.now(timezone.utc).isoformat()

    return {
        "model": pipeline,
        "trained_at": trained_at,
        "model_type": model_type,
        "hyperparameters": final_hyperparameters,
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "test_size": test_size,
        "random_state": random_state,
        "accuracy": float(accuracy),
        "f1": float(f1),
        "train_churn_distribution": get_class_distribution(y_train),
        "test_churn_distribution": get_class_distribution(y_test),
    }


def build_model_metadata(training_result: dict) -> dict:
    """
    Build metadata dictionary from the training result.

    The actual model object is not stored in metadata.
    Metadata is saved as JSON, while the model is saved as joblib.
    """

    return {
        "trained_at": training_result["trained_at"],
        "model_type": training_result["model_type"],
        "hyperparameters": training_result["hyperparameters"],
        "metrics": {
            "accuracy": training_result["accuracy"],
            "f1": training_result["f1"],
        },
        "train_rows": training_result["train_rows"],
        "test_rows": training_result["test_rows"],
        "test_size": training_result["test_size"],
        "random_state": training_result["random_state"],
        "train_churn_distribution": training_result["train_churn_distribution"],
        "test_churn_distribution": training_result["test_churn_distribution"],
    }