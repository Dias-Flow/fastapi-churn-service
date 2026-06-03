from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


VALID_FEATURE_VECTOR = {
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


def test_root_endpoint():
    """
    Root endpoint should confirm that the service is running.
    """

    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "ml churn service is running"
    }


def test_health_endpoint():
    """
    Health endpoint should return service, dataset and model information.
    """

    response = client.get("/health")
    data = response.json()

    assert response.status_code == 200
    assert data["service"] == "ml churn service"
    assert "status" in data
    assert "dataset" in data
    assert "model" in data
    assert "exists" in data["dataset"]
    assert "loaded" in data["model"]


def test_dataset_info_endpoint():
    """
    Dataset info endpoint should return dataset metadata.
    """

    response = client.get("/dataset/info")
    data = response.json()

    assert response.status_code == 200
    assert data["rows"] > 0
    assert data["columns"] == 10
    assert data["target_name"] == "churn"
    assert "churn_distribution" in data


def test_dataset_preview_endpoint():
    """
    Dataset preview endpoint should return requested number of rows.
    """

    response = client.get("/dataset/preview?limit=3")
    data = response.json()

    assert response.status_code == 200
    assert isinstance(data, list)
    assert len(data) == 3


def test_dataset_split_info_endpoint():
    """
    Dataset split info endpoint should return train/test split data.
    """

    response = client.get("/dataset/split-info")
    data = response.json()

    assert response.status_code == 200
    assert data["total_rows"] == data["train_rows"] + data["test_rows"]
    assert data["test_size"] == 0.2
    assert "train_churn_distribution" in data
    assert "test_churn_distribution" in data


def test_model_status_endpoint():
    """
    Model status endpoint should return model state information.
    """

    response = client.get("/model/status")
    data = response.json()

    assert response.status_code == 200
    assert "is_trained" in data
    assert "is_loaded" in data
    assert "model_path" in data
    assert "metadata_path" in data


def test_model_schema_endpoint():
    """
    Model schema endpoint should describe expected prediction input.
    """

    response = client.get("/model/schema")
    data = response.json()

    assert response.status_code == 200
    assert "feature_columns" in data
    assert "numeric_features" in data
    assert "categorical_features" in data
    assert "allowed_categories" in data
    assert "example" in data

    assert "monthly_fee" in data["feature_columns"]
    assert "region" in data["categorical_features"]


def test_model_metrics_endpoint():
    """
    Model metrics endpoint should return latest record and history list.
    """

    response = client.get("/model/metrics")
    data = response.json()

    assert response.status_code == 200
    assert "latest" in data
    assert "history" in data
    assert isinstance(data["history"], list)


def test_train_model_endpoint():
    """
    Training endpoint should train and save a churn model.
    """

    response = client.post(
        "/model/train",
        json={
            "model_type": "logreg",
            "hyperparameters": {}
        },
    )
    data = response.json()

    assert response.status_code == 200
    assert data["model_type"] == "logreg"
    assert data["model_saved"] is True
    assert "accuracy" in data
    assert "f1" in data
    assert 0 <= data["accuracy"] <= 1
    assert 0 <= data["f1"] <= 1


def test_predict_endpoint_after_training():
    """
    Prediction endpoint should return churn prediction after training.
    """

    train_response = client.post(
        "/model/train",
        json={
            "model_type": "logreg",
            "hyperparameters": {}
        },
    )

    assert train_response.status_code == 200

    response = client.post(
        "/predict",
        json=VALID_FEATURE_VECTOR,
    )
    data = response.json()

    assert response.status_code == 200
    assert data["count"] == 1
    assert len(data["predictions"]) == 1

    prediction = data["predictions"][0]

    assert prediction["prediction"] in [0, 1]
    assert 0 <= prediction["probability_not_churn"] <= 1
    assert 0 <= prediction["probability_churn"] <= 1


def test_predict_validation_error():
    """
    Invalid prediction input should return standard validation error.
    """

    invalid_payload = {
        "monthly_fee": -10,
        "usage_hours": 12.5,
        "support_requests": 2,
        "account_age_months": 8,
        "failed_payments": 1,
        "region": "wrong_region",
        "device_type": "phone",
        "payment_method": "cash",
        "autopay_enabled": 5,
    }

    response = client.post(
        "/predict",
        json=invalid_payload,
    )
    data = response.json()

    assert response.status_code == 422
    assert data["code"] == "VALIDATION_ERROR"
    assert data["message"] == "Request validation failed."
    assert isinstance(data["details"], list)


def test_train_model_wrong_hyperparameter_error():
    """
    Wrong model hyperparameter should return standard training error.
    """

    response = client.post(
        "/model/train",
        json={
            "model_type": "random_forest",
            "hyperparameters": {
                "wrong_parameter": 123
            },
        },
    )
    data = response.json()

    assert response.status_code == 400
    assert data["code"] == "MODEL_TRAINING_ERROR"
    assert data["message"] == "Model training failed because of invalid hyperparameters."
    assert "error" in data["details"]