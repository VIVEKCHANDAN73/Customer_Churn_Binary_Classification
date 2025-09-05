import argparse
import joblib
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
from src.utils import load_params
import numpy as np

def build_pipeline(params, categorical, numeric):
    
    # Transformers
    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),  # fill missing cat with mode
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),  # fill missing numeric with mean
        ("scaler", StandardScaler())
    ])

    pre = ColumnTransformer(
        transformers=[
            ("cat", cat_transformer, categorical),
            ("num", num_transformer, numeric)
            ]
        )
    #
    if params["model"]["type"] == "LogisticRegression":
        model = LogisticRegression(
            C=params["model"]["C"],
            max_iter=params["model"]["max_iter"],
            random_state=params["random_state"]
            )
    else:
        raise ValueError("Unsupported model in params.yaml")
    
    pipe = Pipeline(steps=[("pre", pre), ("clf", model)])
    return pipe

def main(params_path: str, train_path: str, model_out: str):
    params = load_params(params_path)
    train_df = pd.read_csv(train_path)
    
    X = train_df[params["features"]["numeric"] + params["features"]["categorical"]]

    y = train_df[params["label"]]
    
    mlflow.set_experiment("churn-demo")
    
    with mlflow.start_run(run_name="training") as run:
        # Log params
        for k,v in params["model"].items():
            mlflow.log_param(f"model.{k}",v)
        mlflow.log_param("random_state", params["random_state"])
        
        pipe = build_pipeline(params, params["features"]["categorical"], params["features"]["numeric"])
        pipe.fit(X,y)
        
        Path(model_out).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, model_out)
        
        # Log model artifact to MLflow and mark signature
        mlflow.sklearn.log_model(pipe, "model")
        existing_run_id = run.info.run_id
        print(f"Run ID: {existing_run_id}") 
            
    
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    main(args.params, args.train, args.model)

    
    