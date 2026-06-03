from dataclasses import dataclass
from typing import Any


@dataclass
class ModelRegistry:
    """
    Simple in-memory storage for the loaded model and its metadata.

    The model is loaded into memory when the application starts.
    After training, the new model is also stored here.
    """

    model: Any | None = None
    metadata: dict | None = None


model_registry = ModelRegistry()


def set_current_model(model: Any, metadata: dict | None = None) -> None:
    """
    Store the current model and metadata in memory.
    """

    model_registry.model = model
    model_registry.metadata = metadata


def clear_current_model() -> None:
    """
    Remove the current model and metadata from memory.
    """

    model_registry.model = None
    model_registry.metadata = None


def get_current_model() -> Any | None:
    """
    Return the current model from memory.
    """

    return model_registry.model


def get_current_metadata() -> dict | None:
    """
    Return metadata for the current model.
    """

    return model_registry.metadata


def is_model_loaded() -> bool:
    """
    Check whether a model is currently loaded in memory.
    """

    return model_registry.model is not None