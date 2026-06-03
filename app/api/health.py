from fastapi import APIRouter

from app.core.config import DATASET_PATH, MODEL_METADATA_PATH, MODEL_PATH
from app.ml.registry import is_model_loaded


router = APIRouter(
    tags=["Health"],
)


@router.get(
    "/health",
    summary="Health check",
)
def health_check():
    """
    Return service health information.

    This endpoint is useful for checking whether the API is alive
    and whether important project files are available.
    """

    dataset_exists = DATASET_PATH.exists()
    model_exists = MODEL_PATH.exists()
    metadata_exists = MODEL_METADATA_PATH.exists()
    model_loaded = is_model_loaded()

    status = "ok" if dataset_exists else "warning"

    return {
        "status": status,
        "service": "ml churn service",
        "dataset": {
            "exists": dataset_exists,
            "path": str(DATASET_PATH),
        },
        "model": {
            "file_exists": model_exists,
            "metadata_exists": metadata_exists,
            "loaded": model_loaded,
            "model_path": str(MODEL_PATH),
            "metadata_path": str(MODEL_METADATA_PATH),
        },
    }