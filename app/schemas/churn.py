from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class FeatureVectorChurn(BaseModel):
    """
    Input features for one customer.

    This schema is used when a user sends customer data to the API.
    """

    monthly_fee: float = Field(
        ...,
        gt=0,
        description="Monthly subscription fee paid by the customer",
        examples=[29.99],
    )
    usage_hours: float = Field(
        ...,
        ge=0,
        description="Number of hours the customer used the service last month",
        examples=[12.5],
    )
    support_requests: int = Field(
        ...,
        ge=0,
        description="Number of customer support requests",
        examples=[2],
    )
    account_age_months: int = Field(
        ...,
        ge=0,
        description="Customer account age in months",
        examples=[8],
    )
    failed_payments: int = Field(
        ...,
        ge=0,
        description="Number of failed payments",
        examples=[1],
    )
    region: Literal["europe", "asia", "america", "africa"] = Field(
        ...,
        description="Customer region",
        examples=["europe"],
    )
    device_type: Literal["mobile", "desktop", "tablet"] = Field(
        ...,
        description="Main device type used by the customer",
        examples=["mobile"],
    )
    payment_method: Literal["card", "paypal", "crypto"] = Field(
        ...,
        description="Customer payment method",
        examples=["card"],
    )
    autopay_enabled: Literal[0, 1] = Field(
        ...,
        description="Whether automatic payments are enabled: 1 means yes, 0 means no",
        examples=[1],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "monthly_fee": 29.99,
                "usage_hours": 12.5,
                "support_requests": 2,
                "account_age_months": 8,
                "failed_payments": 1,
                "region": "europe",
                "device_type": "mobile",
                "payment_method": "card",
                "autopay_enabled": 1,
            }
        }
    )


class DatasetRowChurn(FeatureVectorChurn):
    """
    One row from the training dataset.

    It contains the same input features as FeatureVectorChurn,
    plus the target value: churn.
    """

    churn: Literal[0, 1] = Field(
        ...,
        description="Target value: 1 means customer churned, 0 means customer stayed",
        examples=[0],
    )


class DatasetInfoResponse(BaseModel):
    """
    Basic information about the loaded dataset.
    """

    rows: int = Field(..., description="Number of rows in the dataset")
    columns: int = Field(..., description="Number of columns in the dataset")
    feature_names: list[str] = Field(..., description="List of feature column names")
    target_name: str = Field(..., description="Target column name")
    churn_distribution: dict[int, int] = Field(
        ...,
        description="Number of examples for each churn class",
    )


class SplitInfoResponse(BaseModel):
    """
    Information about train/test split.
    """

    total_rows: int = Field(..., description="Total number of rows in the dataset")
    train_rows: int = Field(..., description="Number of rows in the training set")
    test_rows: int = Field(..., description="Number of rows in the test set")
    test_size: float = Field(..., description="Fraction of the dataset used for testing")
    random_state: int = Field(..., description="Random seed used for reproducibility")
    train_churn_distribution: dict[int, int] = Field(
        ...,
        description="Churn class distribution in the training set",
    )
    test_churn_distribution: dict[int, int] = Field(
        ...,
        description="Churn class distribution in the test set",
    )


class TrainingConfigChurn(BaseModel):
    """
    Configuration for model training.

    model_type controls which classifier will be trained.
    hyperparameters allows the user to pass model-specific parameters.
    """

    model_type: Literal["logreg", "random_forest"] = Field(
        default="logreg",
        description="Classifier type to train",
        examples=["logreg"],
    )
    hyperparameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Custom hyperparameters for the selected model",
        examples=[{"C": 1.0}],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "model_type": "logreg",
                    "hyperparameters": {
                        "C": 1.0,
                        "max_iter": 1000
                    }
                },
                {
                    "model_type": "random_forest",
                    "hyperparameters": {
                        "n_estimators": 150,
                        "max_depth": 5
                    }
                }
            ]
        }
    )


class TrainModelResponse(BaseModel):
    """
    Response returned after training the churn model.
    """

    model_type: str = Field(..., description="Type of the trained model")
    hyperparameters: dict[str, Any] = Field(
        ...,
        description="Final hyperparameters used by the model",
    )
    trained_at: str = Field(..., description="UTC timestamp when the model was trained")
    train_rows: int = Field(..., description="Number of training rows")
    test_rows: int = Field(..., description="Number of test rows")
    test_size: float = Field(..., description="Fraction of the dataset used for testing")
    random_state: int = Field(..., description="Random seed used for reproducibility")
    accuracy: float = Field(..., description="Accuracy score on the test set")
    f1: float = Field(..., description="F1 score on the test set")
    train_churn_distribution: dict[int, int] = Field(
        ...,
        description="Churn class distribution in the training set",
    )
    test_churn_distribution: dict[int, int] = Field(
        ...,
        description="Churn class distribution in the test set",
    )
    model_saved: bool = Field(..., description="Whether the trained model was saved")
    model_path: str = Field(..., description="Path to the saved model file")


class ModelStatusResponse(BaseModel):
    """
    Information about the current model status.
    """

    is_trained: bool = Field(..., description="Whether a saved model exists")
    is_loaded: bool = Field(..., description="Whether a model is loaded in memory")
    model_path: str = Field(..., description="Path to the model file")
    metadata_path: str = Field(..., description="Path to the metadata file")
    trained_at: str | None = Field(
        default=None,
        description="UTC timestamp of the last training",
    )
    model_type: str | None = Field(
        default=None,
        description="Type of the current model",
    )
    hyperparameters: dict | None = Field(
        default=None,
        description="Model hyperparameters",
    )
    metrics: dict | None = Field(
        default=None,
        description="Metrics from the last training",
    )
    details: dict | None = Field(
        default=None,
        description="Full saved model metadata",
    )


class PredictionItemResponseChurn(BaseModel):
    """
    Prediction result for one customer.
    """

    prediction: Literal[0, 1] = Field(
        ...,
        description="Predicted class: 1 means churn, 0 means not churn",
    )
    probability_not_churn: float = Field(
        ...,
        ge=0,
        le=1,
        description="Probability of class 0: customer will stay",
    )
    probability_churn: float = Field(
        ...,
        ge=0,
        le=1,
        description="Probability of class 1: customer will churn",
    )


class PredictionResponseChurn(BaseModel):
    """
    Prediction response for one or many customers.
    """

    count: int = Field(..., description="Number of predictions returned")
    predictions: list[PredictionItemResponseChurn] = Field(
        ...,
        description="List of prediction results",
    )


class ModelSchemaResponse(BaseModel):
    """
    Schema expected by the trained model.
    """

    feature_columns: list[str] = Field(..., description="All input feature columns")
    numeric_features: list[str] = Field(..., description="Numeric feature columns")
    categorical_features: list[str] = Field(..., description="Categorical feature columns")
    target_column: str = Field(..., description="Target column used during training")
    feature_types: dict[str, str] = Field(..., description="Expected feature types")
    allowed_categories: dict[str, list[str]] = Field(
        ...,
        description="Allowed values for categorical features",
    )
    example: dict[str, Any] = Field(..., description="Example input object")


class ErrorResponse(BaseModel):
    """
    Standard API error response.
    """

    code: str = Field(..., description="Application-level error code")
    message: str = Field(..., description="Human-readable error message")
    details: dict | list = Field(
        default_factory=dict,
        description="Additional error details",
    )


class TrainingHistoryRecord(BaseModel):
    """
    One model training history record.
    """

    trained_at: str = Field(..., description="UTC timestamp when the model was trained")
    model_type: str = Field(..., description="Type of the trained model")
    hyperparameters: dict[str, Any] = Field(..., description="Model hyperparameters")
    metrics: dict[str, float] = Field(..., description="Model quality metrics")
    train_rows: int = Field(..., description="Number of training rows")
    test_rows: int = Field(..., description="Number of test rows")
    test_size: float = Field(..., description="Fraction of the dataset used for testing")
    random_state: int = Field(..., description="Random seed used for reproducibility")
    train_churn_distribution: dict[int, int] = Field(
        ...,
        description="Churn distribution in the training set",
    )
    test_churn_distribution: dict[int, int] = Field(
        ...,
        description="Churn distribution in the test set",
    )


class ModelMetricsResponse(BaseModel):
    """
    Latest metrics and recent model training history.
    """

    latest: TrainingHistoryRecord | None = Field(
        default=None,
        description="Latest training record",
    )
    history: list[TrainingHistoryRecord] = Field(
        default_factory=list,
        description="Recent training records",
    )