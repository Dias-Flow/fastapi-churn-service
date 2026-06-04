from fastapi import APIRouter, HTTPException, Query, status

from app.core.errors import ErrorCode
from app.core.logging import get_logger
from app.ml.data import get_dataset_info, get_dataset_preview, get_split_info
from app.schemas.churn import DatasetInfoResponse, DatasetRowChurn, SplitInfoResponse


router = APIRouter(
    prefix="/dataset",
    tags=["Dataset"],
)


logger = get_logger(__name__)


@router.get(
    "/preview",
    response_model=list[DatasetRowChurn],
    summary="Preview churn dataset",
)
def preview_dataset(
    limit: int = Query(
        default=5,
        ge=1,
        le=100,
        description="Number of rows to return",
    )
):
    """
    Return the first rows from churn_dataset.csv.
    """

    try:
        return get_dataset_preview(limit=limit)

    except FileNotFoundError as error:
        logger.exception("Dataset preview failed because dataset file was not found.")

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
        logger.exception("Dataset preview failed because dataset is invalid.")

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": ErrorCode.INVALID_DATASET,
                "message": "Dataset is invalid.",
                "details": {
                    "hint": "Check that the dataset is not empty and contains all required columns."
                },
            },
        ) from error


@router.get(
    "/info",
    response_model=DatasetInfoResponse,
    summary="Get churn dataset information",
)
def dataset_info():
    """
    Return basic information about churn_dataset.csv.
    """

    try:
        return get_dataset_info()

    except FileNotFoundError as error:
        logger.exception("Dataset info failed because dataset file was not found.")

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
        logger.exception("Dataset info failed because dataset is invalid.")

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": ErrorCode.INVALID_DATASET,
                "message": "Dataset is invalid.",
                "details": {
                    "hint": "Check that the dataset is not empty and contains all required columns."
                },
            },
        ) from error


@router.get(
    "/split-info",
    response_model=SplitInfoResponse,
    summary="Get train/test split information",
)
def dataset_split_info(
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
    Return information about train/test split.
    """

    try:
        return get_split_info(
            test_size=test_size,
            random_state=random_state,
        )

    except FileNotFoundError as error:
        logger.exception("Dataset split info failed because dataset file was not found.")

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
        logger.exception("Dataset split info failed because dataset is invalid.")

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": ErrorCode.INVALID_DATASET,
                "message": "Dataset is invalid.",
                "details": {
                    "hint": "Check test_size, random_state and dataset structure."
                },
            },
        ) from error