from fastapi.testclient import TestClient

import app.ml.persistence as persistence
from app.main import create_app
from app.ml.registry import clear_current_model, is_model_loaded


def test_app_starts_when_saved_model_is_corrupted(tmp_path, monkeypatch):
    """
    Application should start even if the saved model file cannot be loaded.

    In this case, the model registry should stay empty and /model/status
    should still be available.
    """

    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    corrupted_model_path = models_dir / "churn_model.joblib"
    corrupted_metadata_path = models_dir / "churn_metadata.json"

    corrupted_model_path.write_text(
        "this is not a valid joblib model",
        encoding="utf-8",
    )
    corrupted_metadata_path.write_text(
        "{}",
        encoding="utf-8",
    )

    monkeypatch.setattr(persistence, "MODEL_PATH", corrupted_model_path)
    monkeypatch.setattr(persistence, "MODEL_METADATA_PATH", corrupted_metadata_path)

    clear_current_model()

    test_app = create_app()

    with TestClient(test_app) as client:
        response = client.get("/model/status")

    assert response.status_code == 200
    assert is_model_loaded() is False

    data = response.json()
    assert data["is_loaded"] is False