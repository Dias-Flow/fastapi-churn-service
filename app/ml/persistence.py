import json
from pathlib import Path
from typing import Any

import joblib

from app.core.config import MODEL_METADATA_PATH, MODEL_PATH, MODELS_DIR


def ensure_models_dir_exists(path: Path | None = None) -> None:
    """
    Create the models directory if it does not exist.

    If a file path is passed, its parent directory is created.
    If no path is passed, the default models directory is created.
    """

    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        return

    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def save_churn_model(
    model: Any,
    path: Path | None = None,
) -> None:
    """
    Save a trained churn model to disk.
    """

    target_path = path or MODEL_PATH
    ensure_models_dir_exists(target_path)
    joblib.dump(model, target_path)


def load_churn_model(
    path: Path | None = None,
) -> Any:
    """
    Load a trained churn model from disk.

    Raises:
        FileNotFoundError: If the model file does not exist.
        Other exceptions may be raised if the file exists but cannot be loaded.
    """

    target_path = path or MODEL_PATH

    if not target_path.exists():
        raise FileNotFoundError(
            f"Saved model was not found at path: {target_path}"
        )

    return joblib.load(target_path)


def save_model_metadata(
    metadata: dict,
    path: Path | None = None,
) -> None:
    """
    Save model metadata to a JSON file.
    """

    target_path = path or MODEL_METADATA_PATH
    ensure_models_dir_exists(target_path)

    with target_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False)


def load_model_metadata(
    path: Path | None = None,
) -> dict | None:
    """
    Load model metadata from a JSON file.

    Returns:
        Metadata dictionary if the file exists, otherwise None.
    """

    target_path = path or MODEL_METADATA_PATH

    if not target_path.exists():
        return None

    with target_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def model_file_exists(
    path: Path | None = None,
) -> bool:
    """
    Check whether the saved model file exists.
    """

    target_path = path or MODEL_PATH
    return target_path.exists()


def metadata_file_exists(
    path: Path | None = None,
) -> bool:
    """
    Check whether the model metadata file exists.
    """

    target_path = path or MODEL_METADATA_PATH
    return target_path.exists()