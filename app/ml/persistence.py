import json
from pathlib import Path
from typing import Any

import joblib

from app.core.config import MODEL_METADATA_PATH, MODEL_PATH, MODELS_DIR


def ensure_models_dir_exists() -> None:
    """
    Create the models directory if it does not exist.

    This function is needed before saving model files.
    """

    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def save_churn_model(model: Any, path: Path = MODEL_PATH) -> None:
    """
    Save a trained churn model to disk.

    Args:
        model: Trained scikit-learn model or pipeline.
        path: File path where the model should be saved.
    """

    ensure_models_dir_exists()
    joblib.dump(model, path)


def load_churn_model(path: Path = MODEL_PATH) -> Any:
    """
    Load a trained churn model from disk.

    Args:
        path: File path where the model is stored.

    Returns:
        Loaded model object.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """

    if not path.exists():
        raise FileNotFoundError(f"Saved model was not found at path: {path}")

    return joblib.load(path)


def save_model_metadata(
    metadata: dict,
    path: Path = MODEL_METADATA_PATH,
) -> None:
    """
    Save model metadata to a JSON file.

    Metadata contains information such as:
    - training timestamp
    - model type
    - metrics
    - train/test sizes
    """

    ensure_models_dir_exists()

    with path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False)


def load_model_metadata(path: Path = MODEL_METADATA_PATH) -> dict | None:
    """
    Load model metadata from a JSON file.

    Args:
        path: File path where metadata is stored.

    Returns:
        Metadata dictionary if the file exists, otherwise None.
    """

    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def model_file_exists(path: Path = MODEL_PATH) -> bool:
    """
    Check whether the saved model file exists.
    """

    return path.exists()


def metadata_file_exists(path: Path = MODEL_METADATA_PATH) -> bool:
    """
    Check whether the model metadata file exists.
    """

    return path.exists()