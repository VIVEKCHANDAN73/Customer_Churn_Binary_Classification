import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils import load_params

RAW_PATH = Path("data/raw/raw.csv")

def maybe_download_placeholder():

    if not RAW_PATH.exists():
        raise FileNotFoundError("data/raw/raw.csv not found. Please place Telco Churn CSV at data/raw/raw.csv.")

def main(params_path: str, out_paths: list[str]):
    params = load_params(params_path)
    out_train, out_test = [Path(p) for p in out_paths]
    out_train.parent.mkdir(parents=True, exist_ok=True)
    out_test.parent.mkdir(parents=True, exist_ok=True)


    maybe_download_placeholder()

    df = pd.read_csv(RAW_PATH)
    
    # Basic cleanups aligning with params
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Map label to 0/1
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    
    train_df, test_df = train_test_split(df, test_size=params["split"]["test_size"], random_state=params["random_state"], stratify=df[params["label"]])
    train_df.to_csv(out_train, index=False)
    test_df.to_csv(out_test, index=False)
    
    
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--out", nargs=2, required=True)
    args = ap.parse_args()
    main(args.params, args.out)
