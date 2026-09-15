import polars as pl
import os
import glob

def convert_csv_to_parquet(raw_dir: str, processed_dir: str):
    os.makedirs(processed_dir, exist_ok=True)
    csv_files = glob.glob(os.path.join(raw_dir, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {raw_dir}")
        return

    for csv_file in csv_files:
        base_name = os.path.basename(csv_file)
        name, _ = os.path.splitext(base_name)
        parquet_file = os.path.join(processed_dir, f"{name}.parquet")
        
        print(f"Converting {csv_file} to {parquet_file}...")
        
        try:
            pl.scan_csv(csv_file).sink_parquet(parquet_file)
            print(f"Successfully converted {base_name}")
        except Exception as e:
            print(f"Failed to convert {base_name}: {e}")

if __name__ == "__main__":
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    raw_dir = os.path.join(root_dir, "data", "raw")
    processed_dir = os.path.join(root_dir, "data", "processed")
    convert_csv_to_parquet(raw_dir, processed_dir)
