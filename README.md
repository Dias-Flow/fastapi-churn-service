# ML Churn Service

Educational FastAPI service for customer churn prediction.

The project uses a tabular churn dataset and trains a machine learning model to predict whether a customer will leave the service next month.

## Task

The service solves a binary classification task:

- `churn = 1` — customer churned
- `churn = 0` — customer stayed

## Dataset

The training dataset is stored in:

```text
data/churn_dataset.csv


Dataset columns:

Column	Description
monthly_fee	Monthly subscription fee
usage_hours	Number of usage hours during the last month
support_requests	Number of support requests
account_age_months	Account age in months
failed_payments	Number of failed payments
region	Customer region
device_type	Main device type
payment_method	Payment method
autopay_enabled	Whether autopay is enabled
churn	Target variable
Features

Numeric features:

monthly_fee
usage_hours
support_requests
account_age_months
failed_payments
autopay_enabled

Categorical features:

region
device_type
payment_method
Project structure
app/
  api/
    dataset.py
    health.py
    model.py
    predict.py
  core/
    config.py
    errors.py
    logging.py
  ml/
    data.py
    features.py
    history.py
    persistence.py
    prediction.py
    preprocessing.py
    registry.py
    training.py
  schemas/
    churn.py
  main.py

data/
  churn_dataset.csv

models/
  churn_model.joblib
  churn_metadata.json
  churn_training_history.json

tests/
  test_api.py
  test_dataset.py
  test_training.py
Installation

Create and activate virtual environment:

python -m venv .venv

Activate on Windows:

.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt
Run locally
python -m uvicorn app.main:app --reload

Open API documentation:

http://127.0.0.1:8000/docs
Main endpoints
Method	Endpoint	Description
GET	/	Service status
GET	/health	Health check
GET	/dataset/preview	Dataset preview
GET	/dataset/info	Dataset information
GET	/dataset/split-info	Train/test split information
POST	/model/train	Train and save model
GET	/model/status	Model status
GET	/model/schema	Expected prediction schema
GET	/model/metrics	Latest metrics and training history
POST	/predict	Predict churn
Train model

Request:

{
  "model_type": "logreg",
  "hyperparameters": {}
}

Supported model types:

logreg
random_forest

Example with Random Forest:

{
  "model_type": "random_forest",
  "hyperparameters": {
    "n_estimators": 100,
    "max_depth": 5
  }
}
Predict churn

Request:

{
  "monthly_fee": 29.99,
  "usage_hours": 12.5,
  "support_requests": 2,
  "account_age_months": 8,
  "failed_payments": 1,
  "region": "europe",
  "device_type": "mobile",
  "payment_method": "card",
  "autopay_enabled": 1
}

Response:

{
  "count": 1,
  "predictions": [
    {
      "prediction": 0,
      "probability_not_churn": 0.82,
      "probability_churn": 0.18
    }
  ]
}
Error format

All API errors use the same structure:

{
  "code": "VALIDATION_ERROR",
  "message": "Request validation failed.",
  "details": []
}

Example: prediction without a loaded model:

{
  "code": "MODEL_NOT_LOADED",
  "message": "Model is not trained or not loaded.",
  "details": {
    "hint": "Train the model using POST /model/train first."
  }
}
Tests

Run tests:

python -m pytest

Expected result:

20 passed
Docker

Build Docker image:

docker build -t fastapi-churn .

Run container:

docker run --rm -p 8000:8000 fastapi-churn

Open docs:

http://127.0.0.1:8000/docs

Open health check:

http://127.0.0.1:8000/health
ML pipeline

The model is implemented as a scikit-learn Pipeline:

ColumnTransformer
  - StandardScaler for numeric features
  - OneHotEncoder for categorical features

Classifier
  - LogisticRegression
  - or RandomForestClassifier

The full pipeline is saved with joblib.

Training history

Each training run is saved to:

models/churn_training_history.json

The latest metrics and recent training records are available at:

GET /model/metrics