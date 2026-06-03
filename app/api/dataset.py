from fastapi import APIRouter, HTTPException, Query, status

from app.core.errors import ErrorCode
from app.ml.data import get_dataset_info, get_dataset_preview, get_split_info
from app.schemas.churn import DatasetInfoResponse, DatasetRowChurn, SplitInfoResponse


router = APIRouter(
    prefix="/dataset",
    tags=["Dataset"],
)


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

    This endpoint is useful for checking that the dataset was loaded correctly.
    """

    try:
        return get_dataset_preview(limit=limit)

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": ErrorCode.DATASET_NOT_FOUND,
                "message": "Dataset file was not found.",
                "details": {"error": str(error)},
            },
        ) from error

    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": ErrorCode.INVALID_DATASET,
                "message": "Dataset is invalid.",
                "details": {"error": str(error)},
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

    The response includes:
    - number of rows
    - number of columns
    - feature names
    - target name
    - churn class distribution
    """

    try:
        return get_dataset_info()

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": ErrorCode.DATASET_NOT_FOUND,
                "message": "Dataset file was not found.",
                "details": {"error": str(error)},
            },
        ) from error

    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": ErrorCode.INVALID_DATASET,
                "message": "Dataset is invalid.",
                "details": {"error": str(error)},
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

    This endpoint does not train the model.
    It only shows how the dataset will be split before training.
    """

    try:
        return get_split_info(
            test_size=test_size,
            random_state=random_state,
        )

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": ErrorCode.DATASET_NOT_FOUND,
                "message": "Dataset file was not found.",
                "details": {"error": str(error)},
            },
        ) from error

    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": ErrorCode.INVALID_DATASET,
                "message": "Dataset is invalid.",
                "details": {"error": str(error)},
            },
        ) from error