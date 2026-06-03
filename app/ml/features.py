"""
Feature definitions for the churn ML task.

This module keeps all feature-related constants in one place.
It helps us avoid duplicated column names across the project.
"""


NUMERIC_FEATURES = [
    "monthly_fee",
    "usage_hours",
    "support_requests",
    "account_age_months",
    "failed_payments",
    "autopay_enabled",
]


CATEGORICAL_FEATURES = [
    "region",
    "device_type",
    "payment_method",
]


FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


TARGET_COLUMN = "churn"


REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]


ALLOWED_CATEGORIES = {
    "region": ["europe", "asia", "america", "africa"],
    "device_type": ["mobile", "desktop", "tablet"],
    "payment_method": ["card", "paypal", "crypto"],
}


FEATURE_TYPES = {
    "monthly_fee": "float, greater than 0",
    "usage_hours": "float, greater than or equal to 0",
    "support_requests": "integer, greater than or equal to 0",
    "account_age_months": "integer, greater than or equal to 0",
    "failed_payments": "integer, greater than or equal to 0",
    "region": "string category",
    "device_type": "string category",
    "payment_method": "string category",
    "autopay_enabled": "integer, 0 or 1",
}


EXAMPLE_FEATURE_VECTOR = {
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


def get_model_input_schema() -> dict:
    """
    Return the input schema expected by the churn model.

    This schema is useful for API clients.
    It describes feature names, feature types, allowed categories,
    and an example request.
    """

    return {
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target_column": TARGET_COLUMN,
        "feature_types": FEATURE_TYPES,
        "allowed_categories": ALLOWED_CATEGORIES,
        "example": EXAMPLE_FEATURE_VECTOR,
    }