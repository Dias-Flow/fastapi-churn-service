from typing import Literal

from fastapi import APIRouter, Body, HTTPException, Query, status

from app.core.config import MODEL_METADATA_PATH, MODEL_PATH
from app.core.errors import ErrorCode
from app.core.logging import get_logger
from app.ml.features import get_model_input_schema
from app.ml.history import append_training_record, get_latest_training_record, get_recent_training_history
from app.ml.persistence import (
    load_model_metadata,
    model_file_exists,
    save_churn_model,
    save_model_metadata,
)
from app.ml.registry import is_model_loaded, set_current_model
from app.ml.training import build_model_metadata, train_churn_model
from app.schemas.churn import (
    ModelMetricsResponse,
    ModelSchemaResponse,
    ModelStatusResponse,
    TrainModelResponse,
    TrainingConfigChurn,
)


router = APIRouter(
    prefix="/model",
    tags=["Model"],
)


logger = get_logger(__name__)


@router.post(
    "/train",
    response_model=TrainModelResponse,
    summary="Train and save churn model",
)
def train_model(
    config: TrainingConfigChurn | None = Body(
        default=None,
        description="Training configuration. If omitted, logreg with default parameters is used.",
    ),
    test_size: float = Query(
        default=0.2,
        gt=0,
        lt=1,
        description="Fraction of the dataset used for testing",
    ),
    random_state: int = Query(
        default=42,
        ge=0,
        description="Random seed used for reproducibility",
    ),
):
    """
    Train the churn classification model and save it to disk.
    """

    try:
        config = config or TrainingConfigChurn()

        logger.info(
            "Model training started. model_type=%s, test_size=%s, random_state=%s",
            config.model_type,
            test_size,
            random_state,
        )

        training_result = train_churn_model(
            test_size=test_size,
            random_state=random_state,
            model_type=config.model_type,
            hyperparameters=config.hyperparameters,
        )

        trained_model = training_result["model"]
        metadata = build_model_metadata(training_result)

        save_churn_model(trained_model)
        save_model_metadata(metadata)
        append_training_record(metadata)

        set_current_model(
            model=trained_model,
            metadata=metadata,
        )

        logger.info(
            "Model training finished. model_type=%s, accuracy=%s, f1=%s",
            training_result["model_type"],
            training_result["accuracy"],
            training_result["f1"],
        )

        response_data = training_result.copy()
        response_data.pop("model")
        response_data["model_saved"] = True
        response_data["model_path"] = str(MODEL_PATH)

        return response_data

    except FileNotFoundError as error:
        logger.exception("Model training failed because dataset file was not found.")

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": ErrorCode.DATASET_NOT_FOUND,
                "message": "Dataset file was not found.",
                "details": {
                    "hint": "Make sure data/churn_dataset.csv exists."
                },
            },
        ) from error

    except ValueError as error:
        logger.exception("Model training failed because of invalid dataset or configuration.")

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": ErrorCode.INVALID_DATASET,
                "message": "Invalid input data or training configuration.",
                "details": {
                    "hint": "Check dataset structure, test_size, model_type and hyperparameters."
                },
            },
        ) from error

    except TypeError as error:
        logger.exception("Model training failed because of invalid hyperparameters.")

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": ErrorCode.MODEL_TRAINING_ERROR,
                "message": "Model training failed because of invalid hyperparameters.",
                "details": {
                    "hint": "Check that hyperparameter names are valid for the selected model_type."
                },
            },
        ) from error


@router.get(
    "/status",
    response_model=ModelStatusResponse,
    summary="Get churn model status",
)
def model_status():
    """
    Return information about the current model.
    """

    metadata = load_model_metadata()

    return {
        "is_trained": model_file_exists(),
        "is_loaded": is_model_loaded(),
        "model_path": str(MODEL_PATH),
        "metadata_path": str(MODEL_METADATA_PATH),
        "trained_at": metadata.get("trained_at") if metadata else None,
        "model_type": metadata.get("model_type") if metadata else None,
        "hyperparameters": metadata.get("hyperparameters") if metadata else None,
        "metrics": metadata.get("metrics") if metadata else None,
        "details": metadata,
    }


@router.get(
    "/schema",
    response_model=ModelSchemaResponse,
    summary="Get model input schema",
)
def model_schema():
    """
    Return the input schema expected by the churn model.
    """

    return get_model_input_schema()


@router.get(
    "/metrics",
    response_model=ModelMetricsResponse,
    summary="Get latest model metrics and training history",
)
def model_metrics(
    limit: int = Query(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of history records to return",
    ),
    model_type: Literal["logreg", "random_forest"] | None = Query(
        default=None,
        description="Optional model type filter",
    ),
):
    """
    Return latest model metrics and recent training history.
    """

    return {
        "latest": get_latest_training_record(),
        "history": get_recent_training_history(
            limit=limit,
            model_type=model_type,
        ),
    }