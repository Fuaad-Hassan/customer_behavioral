import kagglehub
import shutil
import os

def download_data(dest_dir: str = "data/raw"):
    print("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("mkechinov/ecommerce-behavior-data-from-multi-category-store")
    
    print(f"Downloaded to: {path}")
    os.makedirs(dest_dir, exist_ok=True)
    
    print(f"Copying files to {dest_dir}...")
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith('.csv'):
                src_file = os.path.join(root, file)
                dest_file = os.path.join(dest_dir, file)
                print(f"Copying {src_file} -> {dest_file}")
                shutil.copy2(src_file, dest_file)
                
    print("Download and extraction complete.")

if __name__ == "__main__":
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    dest_dir = os.path.join(root_dir, "data", "raw")
    download_data(dest_dir)
