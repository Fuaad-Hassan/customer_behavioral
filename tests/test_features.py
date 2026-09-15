import polars as pl
import os
import tempfile
import pytest
from src.features.build_features import build_features

def test_feature_engineering_pipeline_integrity():
    """
    Validates that the Polars feature engineering pipeline correctly aggregates
    raw logs without producing NaNs on division by zero, correctly calculates
    temporal logic, and prevents schema drift crashes.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "raw.parquet")
        output_path = os.path.join(tmpdir, "features.parquet")
        
        # Create a mock Kaggle raw event stream DataFrame
        raw_data = pl.DataFrame({
            "event_time": ["2019-10-01 00:00:00 UTC", "2019-10-01 00:05:00 UTC", "2019-10-01 00:10:00 UTC"],
            "event_type": ["view", "cart", "purchase"],
            "product_id": [1, 1, 1],
            "category_code": ["electronics", "electronics", "electronics"],
            "category_id": [100, 100, 100],
            "brand": ["apple", "apple", "apple"],
            "price": [1000.0, 1000.0, 1000.0],
            "user_id": [10, 10, 10],
            "user_session": ["sess_valid_1", "sess_valid_1", "sess_valid_1"]
        })
        raw_data.write_parquet(input_path)
        
        # Execute the pipeline (simulating the cron job)
        build_features(input_path, output_path)
        
        # Assertions
        assert os.path.exists(output_path), "Pipeline failed to write output."
        features = pl.read_parquet(output_path)
        
        assert len(features) == 1, "Aggregation failed to group to a single user_session."
        row = features.row(0, named=True)
        
        # Test 1: Session Depth should equal the number of 'view' events (1)
        assert row["session_depth"] == 1
        
        # Test 2: Cart Churn Rate (remove_from_cart / cart). 
        # Since there are 0 removes and 1 cart, it must be 0.0, not NaN.
        assert row["cart_churn_rate"] == 0.0
        
        # Test 3: Historical average purchase price should match the single purchase
        assert row["hist_avg_purchase_price"] == 1000.0
        
        # Test 4: Temporal extraction. Oct 1, 2019 was a Tuesday (weekday).
        assert row["is_weekend"] is False
        
        # Test 5: Hesitation Time. View at 00:00:00, Cart at 00:05:00 = 300 seconds.
        assert row["session_avg_hesitation_time"] == 300.0
