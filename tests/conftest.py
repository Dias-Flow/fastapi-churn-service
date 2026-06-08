import pytest

import app.api.health as health_api
import app.api.model as model_api
import app.core.config as config
import app.ml.history as history_module
import app.ml.persistence as persistence
from app.ml.registry import clear_current_model


@pytest.fixture(autouse=True)
def isolate_model_artifacts(tmp_path, monkeypatch):
    """
    Isolate tests from real model artifacts.

    Tests should not write to real project files:
    - models/churn_model.joblib
    - models/churn_metadata.json
    - models/churn_training_history.json
    """

    models_dir = tmp_path / "models"
    model_path = models_dir / "churn_model.joblib"
    metadata_path = models_dir / "churn_metadata.json"
    history_path = models_dir / "churn_training_history.json"

    monkeypatch.setattr(config, "MODELS_DIR", models_dir)
    monkeypatch.setattr(config, "MODEL_PATH", model_path)
    monkeypatch.setattr(config, "MODEL_METADATA_PATH", metadata_path)
    monkeypatch.setattr(config, "MODEL_HISTORY_PATH", history_path)

    monkeypatch.setattr(persistence, "MODELS_DIR", models_dir)
    monkeypatch.setattr(persistence, "MODEL_PATH", model_path)
    monkeypatch.setattr(persistence, "MODEL_METADATA_PATH", metadata_path)

    monkeypatch.setattr(history_module, "MODELS_DIR", models_dir)
    monkeypatch.setattr(history_module, "MODEL_HISTORY_PATH", history_path)

    monkeypatch.setattr(model_api, "MODEL_PATH", model_path)
    monkeypatch.setattr(model_api, "MODEL_METADATA_PATH", metadata_path)

    monkeypatch.setattr(health_api, "MODEL_PATH", model_path)
    monkeypatch.setattr(health_api, "MODEL_METADATA_PATH", metadata_path)

    clear_current_model()

    yield

    clear_current_model()