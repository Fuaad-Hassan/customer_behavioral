import polars as pl
import pandas as pd
import xgboost as xgb
import mlflow
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score

def prepare_data(input_path: str) -> pd.DataFrame:
    """
    Loads the clustered feature matrix and prepares it for XGBoost.
    Generates a simulated 'is_purchased' target variable based on engagement.
    """
    print(f"Loading psychographic clusters from {input_path}...")
    df = pl.read_parquet(input_path)
    threshold = df.select(pl.col("session_depth").quantile(0.85)).item()
    df = df.with_columns(
        (pl.col("session_depth") >= threshold).cast(pl.Int32).alias("is_purchased")
    )

    cols_to_drop = ["user_session", "user_id", "time_of_day", "session_depth"]
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]    
    df = df.drop(cols_to_drop)
    
    print("Converting Polars dataframe to Pandas dataframe for Scikit-Learn integration...")
    pdf = df.to_pandas()
    
    return pdf

def train_and_log(pdf: pd.DataFrame):
    """
    Trains the XGBoost propensity model and comprehensively logs parameters, 
    metrics, and the physical artifact to an MLflow SQLite backend.
    """
    print("Splitting data into 80% training and 20% testing sets (stratified)...")
    X = pdf.drop(columns=["is_purchased"])
    y = pdf["is_purchased"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    tracking_uri = "sqlite:///mlflow.db"
    experiment_name = "Behavioral_Propensity_Model"
    
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    params = {
        "learning_rate": 0.05,
        "max_depth": 6,
        "n_estimators": 150,
        "eval_metric": "logloss",
        "random_state": 42
    }
    
    print(f"\nStarting MLflow Run in experiment: '{experiment_name}'")
    with mlflow.start_run() as run:
        run_id = run.info.run_id
    
        print("Training XGBoost Classifier...")
        model = xgb.XGBClassifier(use_label_encoder=False, **params)
        model.fit(X_train, y_train)
        
        print("Generating predictions on the test set...")
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        roc_auc = roc_auc_score(y_test, y_prob)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        accuracy = accuracy_score(y_test, y_pred)
        
        metrics = {
            "roc_auc": roc_auc,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy
        }
        
        print("Logging parameters, metrics, and physical model to MLflow...")
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, "xgb_propensity_model")
        
        print("\n================ MLOps Run Summary ================")
        print(f"MLflow Run ID : {run_id}")
        print(f"Tracking URI  : {tracking_uri}")
        print("---------------------------------------------------")
        print("Evaluation Metrics:")
        print(f"  ROC-AUC   : {roc_auc:.4f}")
        print(f"  Precision : {precision:.4f}")
        print(f"  Recall    : {recall:.4f}")
        print(f"  Accuracy  : {accuracy:.4f}")
        print("===================================================\n")
        print("MLOps pipeline successfully tracked.")

if __name__ == "__main__":
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    input_path = os.path.join(root_dir, "data", "processed", "clustered_users.parquet")

    os.chdir(root_dir)
    
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found.")
        print("verify that clustering.py executed successfully.")
        exit(1)
        
    pdf_features = prepare_data(input_path)
    train_and_log(pdf_features)
