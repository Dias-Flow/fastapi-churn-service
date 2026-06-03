from sklearn.pipeline import Pipeline

from app.ml.training import build_churn_pipeline, train_churn_model


def test_build_churn_pipeline_returns_pipeline():
    """
    build_churn_pipeline should return sklearn Pipeline and hyperparameters.
    """

    pipeline, hyperparameters = build_churn_pipeline(
        model_type="logreg",
        hyperparameters={},
        random_state=42,
    )

    assert isinstance(pipeline, Pipeline)
    assert "preprocessor" in pipeline.named_steps
    assert "model" in pipeline.named_steps
    assert hyperparameters["max_iter"] == 1000
    assert hyperparameters["random_state"] == 42


def test_build_churn_pipeline_supports_random_forest():
    """
    build_churn_pipeline should support random_forest model type.
    """

    pipeline, hyperparameters = build_churn_pipeline(
        model_type="random_forest",
        hyperparameters={"n_estimators": 10},
        random_state=42,
    )

    assert isinstance(pipeline, Pipeline)
    assert "preprocessor" in pipeline.named_steps
    assert "model" in pipeline.named_steps
    assert hyperparameters["n_estimators"] == 10
    assert hyperparameters["random_state"] == 42


def test_build_churn_pipeline_rejects_unknown_model_type():
    """
    Unknown model types should raise ValueError.
    """

    try:
        build_churn_pipeline(
            model_type="unknown_model",
            hyperparameters={},
            random_state=42,
        )
    except ValueError as error:
        assert "Unsupported model_type" in str(error)
    else:
        raise AssertionError("Expected ValueError for unsupported model_type")


def test_train_churn_model_returns_metrics():
    """
    train_churn_model should train a model and return metrics.
    """

    result = train_churn_model(
        test_size=0.2,
        random_state=42,
        model_type="logreg",
        hyperparameters={},
    )

    assert "model" in result
    assert "accuracy" in result
    assert "f1" in result
    assert "trained_at" in result
    assert "train_rows" in result
    assert "test_rows" in result

    assert 0 <= result["accuracy"] <= 1
    assert 0 <= result["f1"] <= 1
    assert result["train_rows"] > 0
    assert result["test_rows"] > 0