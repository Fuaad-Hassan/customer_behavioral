import polars as pl
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
import os

def run_clustering(input_path: str, output_path: str):
    """
    Executes the Psychographic Segmentation Pipeline.
    Loads behavioral features, scales them, and applies K-Means clustering (k=3)
    to identify behavioral archetypes.
    """
    print(f"Loading behavioral features from {input_path}...")
    
    df = pl.read_parquet(input_path)

    df = df.drop_nulls()
    
    if len(df) == 0:
        print("Error: DataFrame is empty after dropping nulls.")
        return
    
    exclude_cols = ["user_session", "user_id", "time_of_day"]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"Features selected for clustering:\n{feature_cols}")
    
    X = df.select(feature_cols).to_numpy()

    print("\nNormalizing data with StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Training KMeans model (n_clusters=3)...")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X_scaled)
    
    print("Appending cluster assignments to the dataset...")
    df = df.with_columns(
        pl.Series("psychographic_cluster", labels)
    )
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Saving clustered dataframe to {output_path}...")
    df.write_parquet(output_path)
    
    print("\n================ Cluster Diagnostics ================")
    print("Average feature values for each psychographic cluster:")
    
    centroids = df.group_by("psychographic_cluster").agg(
        [pl.col(c).mean().round(4).alias(c) for c in feature_cols]
    ).sort("psychographic_cluster")
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(centroids.to_pandas())
    
    print("\n--- Blueprint Mapping Reference ---")
    print("* The Impulse Buyer : Fast action, short sessions, low hesitation.")
    print("* The Researcher    : High session depth, return visits, long hesitation.")
    print("* The Bargain Hunter: High cart churn, price-sensitive.")
    print("=====================================================")

if __name__ == "__main__":
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    input_path = os.path.join(root_dir, "data", "processed", "behavioral_features.parquet")
    output_path = os.path.join(root_dir, "data", "processed", "clustered_users.parquet")
    
    if not os.path.exists(input_path):
        print(f"Error: Could not find {input_path}")
        print("Please ensure you have run the feature engineering pipeline first.")
        exit(1)
        
    run_clustering(input_path, output_path)