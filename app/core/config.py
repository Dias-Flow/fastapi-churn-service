from pathlib import Path


# Project root directory.
# Example:
# fastapi-churn-service/
# ├── app/
# ├── data/
# └── models/
BASE_DIR = Path(__file__).resolve().parents[2]

# Directory where the training dataset is stored.
DATA_DIR = BASE_DIR / "data"

# Path to the churn dataset CSV file.
DATASET_PATH = DATA_DIR / "churn_dataset.csv"

# Directory where trained ML models are stored.
MODELS_DIR = BASE_DIR / "models"

# Path to the saved churn model.
MODEL_PATH = MODELS_DIR / "churn_model.joblib"

# Path to model metadata.
MODEL_METADATA_PATH = MODELS_DIR / "churn_metadata.json"

# Path to model training history.
MODEL_HISTORY_PATH = MODELS_DIR / "churn_training_history.json"

# Basic application metadata.
APP_TITLE = "ML Churn Service"
APP_DESCRIPTION = "Educational FastAPI service for customer churn prediction"
APP_VERSION = "0.4.0"