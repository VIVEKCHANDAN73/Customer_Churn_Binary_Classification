import os
import mlflow
import mlflow.sklearn
import joblib
from mlflow.tracking import MlflowClient

# Read MLflow URI from environment variable
mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(mlflow_uri)

# Load model
model_path = "models/model.pkl"
model = joblib.load(model_path)

# Start MLflow run
with mlflow.start_run() as run:
    # Log the sklearn model directly (no wrapper needed)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="CustomerChurnModel"
    )
    
    print("Run ID:", run.info.run_id)

# Transition latest version to Productiond
client = MlflowClient()
latest_versions = client.get_latest_versions("CustomerChurnModel")
if latest_versions:
    version = latest_versions[-1].version  # pick the newest one
    client.transition_model_version_stage(
        name="CustomerChurnModel",
        version=version,
        stage="Production"
    )
    print(f"Model CustomerChurnModel v{version} promoted to Production ✅")
else:
    print("No model version found to promote.")
