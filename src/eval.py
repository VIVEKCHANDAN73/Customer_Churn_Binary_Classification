import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support,confusion_matrix
import mlflow
import mlflow.sklearn
from src.utils import load_params
import joblib

def main(params_path: str, model_path: str, test_path: str):
    params = load_params(params_path)
    df = pd.read_csv(test_path)
    
    X_test = df[params["features"]["numeric"] + params["features"]["categorical"]]
    y_test = df[params["label"]]
    
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    
    metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
    
    # ===== Log metrics to MLflow =====
    mlflow.set_experiment("churn-demo")
    
    with mlflow.start_run(run_name="evaluation") as run:
        # Log metrics
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
    
        
        print(f"Run ID: {run.info.run_id}")
        
    # Save metrics locally for DVC
    Path("models").mkdir(parents=True, exist_ok=True)
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    # Plot heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("models/cm.png")
    # plt.show()
    
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--test", required=True)
    args = ap.parse_args()
    main(args.params, args.model, args.test)