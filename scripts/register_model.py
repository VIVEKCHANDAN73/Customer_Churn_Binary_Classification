import os
import mlflow
import mlflow.sklearn
import joblib

# Read MLflow URI from environment variable
mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(mlflow_uri)

# Load model
model_path = "models/model.pkl"
model = joblib.load(model_path)

# Register model
with mlflow.start_run():
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="CustomerChurnModel"
    )
