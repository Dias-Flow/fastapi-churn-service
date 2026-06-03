import json

from app.core.config import MODEL_HISTORY_PATH, MODELS_DIR


def ensure_history_file_exists() -> None:
    """
    Create the history file if it does not exist.

    The history is stored as a JSON list.
    Each item in the list represents one model training run.
    """

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_HISTORY_PATH.exists():
        with MODEL_HISTORY_PATH.open("w", encoding="utf-8") as file:
            json.dump([], file, indent=4)


def load_training_history() -> list[dict]:
    """
    Load model training history from the JSON file.

    Returns:
        A list of training records.
    """

    ensure_history_file_exists()

    with MODEL_HISTORY_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_training_history(history: list[dict]) -> None:
    """
    Save the full training history to the JSON file.
    """

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with MODEL_HISTORY_PATH.open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=4, ensure_ascii=False)


def append_training_record(record: dict) -> None:
    """
    Add one training record to the history file.

    Args:
        record: Model metadata from the latest training run.
    """

    history = load_training_history()
    history.append(record)
    save_training_history(history)


def get_latest_training_record() -> dict | None:
    """
    Return the latest training record.

    Returns:
        The latest record or None if history is empty.
    """

    history = load_training_history()

    if not history:
        return None

    return history[-1]


def get_recent_training_history(limit: int = 10, model_type: str | None = None) -> list[dict]:
    """
    Return recent training records.

    Args:
        limit: Maximum number of records to return.
        model_type: Optional model type filter.

    Returns:
        List of recent training records.
    """

    history = load_training_history()

    if model_type is not None:
        history = [
            record for record in history
            if record.get("model_type") == model_type
        ]

    return history[-limit:]