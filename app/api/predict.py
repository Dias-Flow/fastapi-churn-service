from typing import Union

from fastapi import APIRouter, Body, HTTPException, status

from app.core.errors import ErrorCode
from app.ml.prediction import predict_churn
from app.ml.registry import get_current_model, is_model_loaded
from app.schemas.churn import FeatureVectorChurn, PredictionResponseChurn
from app.core.logging import get_logger

router = APIRouter(
    tags=["Prediction"],
)

logger = get_logger(__name__)

@router.post(
    "/predict",
    response_model=PredictionResponseChurn,
    summary="Predict customer churn",
)
def predict(
    features: Union[FeatureVectorChurn, list[FeatureVectorChurn]] = Body(
        ...,
        description="One customer object or a list of customer objects",
    )
):
    """
    Predict churn for one or many customers.

    The endpoint returns:
    - predicted churn class
    - probability of not churn
    - probability of churn

    A trained model must be loaded before this endpoint can be used.
    """

    if not is_model_loaded():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": ErrorCode.MODEL_NOT_LOADED,
                "message": "Model is not trained or not loaded.",
                "details": {
                    "hint": "Train the model using POST /model/train first."
                },
            },
        )

    model = get_current_model()

    try:
        if isinstance(features, list):
            feature_items = features
        else:
            feature_items = [features]

        feature_rows = [
            feature_item.model_dump()
            for feature_item in feature_items
        ]

        logger.info("Prediction requested. count=%s", len(feature_rows))

        predictions = predict_churn(
            model=model,
            feature_rows=feature_rows,
        )

        logger.info("Prediction finished. count=%s", len(predictions))

        return {
            "count": len(predictions),
            "predictions": predictions,
        }

    except Exception as error:
        logger.exception("Prediction failed.")

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": ErrorCode.PREDICTION_ERROR,
                "message": "Prediction failed.",
                "details": {"error": str(error)},
            },
        ) from error