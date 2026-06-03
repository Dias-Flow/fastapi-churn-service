import pandas as pd

from app.ml.features import FEATURE_COLUMNS


def feature_rows_to_dataframe(feature_rows: list[dict]) -> pd.DataFrame:
    """
    Convert input feature rows to a pandas DataFrame.

    The column order must be exactly the same as during model training.
    This is important because the ML pipeline expects known feature names.
    """

    dataframe = pd.DataFrame(feature_rows)

    return dataframe[FEATURE_COLUMNS]


def predict_churn(model, feature_rows: list[dict]) -> list[dict]:
    """
    Predict churn class and class probabilities.

    Args:
        model: Trained scikit-learn Pipeline.
        feature_rows: List of customer feature dictionaries.

    Returns:
        List of prediction dictionaries.
    """

    dataframe = feature_rows_to_dataframe(feature_rows)

    predicted_classes = model.predict(dataframe)
    predicted_probabilities = model.predict_proba(dataframe)

    model_classes = list(model.classes_)

    not_churn_index = model_classes.index(0)
    churn_index = model_classes.index(1)

    predictions = []

    for predicted_class, probabilities in zip(predicted_classes, predicted_probabilities):
        predictions.append(
            {
                "prediction": int(predicted_class),
                "probability_not_churn": float(probabilities[not_churn_index]),
                "probability_churn": float(probabilities[churn_index]),
            }
        )

    return predictions