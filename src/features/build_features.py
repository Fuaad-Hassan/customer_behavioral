import polars as pl
import glob
import os
import shutil

def build_features(input_path: str, output_path: str):
    """
    Reads raw event logs and engineers a psychographic feature matrix.
    Uses Polars lazy evaluation and intermediate disk caching to prevent memory exhaustion (OOM).
    """
    print("Starting memory-optimized feature engineering pipeline...")
    
    base_dir = os.path.dirname(output_path)
    temp_dir = os.path.join(base_dir, "temp_features")
    os.makedirs(temp_dir, exist_ok=True)
    
    lf = pl.scan_parquet(input_path)
    lf = lf.with_columns(
        pl.col("event_time").str.replace(" UTC", "", literal=True).str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False).alias("event_time")
    )
    
    print("Step 1: Aggregating User Stats...")
    user_stats = lf.group_by("user_id").agg(
        pl.col("event_time").dt.date().n_unique().alias("return_visit_frequency"),
        (pl.col("event_type") == "remove_from_cart").sum().alias("total_removes"),
        (pl.col("event_type") == "cart").sum().alias("total_carts"),
        pl.col("price").filter(pl.col("event_type") == "purchase").mean().alias("hist_avg_purchase_price")
    ).with_columns(
        (pl.col("total_removes") / pl.col("total_carts")).fill_nan(0.0).alias("cart_churn_rate")
    ).drop(["total_removes", "total_carts"])
    
    user_stats_path = os.path.join(temp_dir, "user_stats.parquet")
    user_stats.sink_parquet(user_stats_path)
    
    print("Step 2: Aggregating Session Stats...")
    session_stats = lf.group_by("user_session").agg(
        pl.col("user_id").first(),
        (pl.col("event_type") == "view").sum().alias("session_depth"),
        (pl.col("event_type") == "view").sum().alias("total_views"),
        pl.col("event_time").min().alias("session_start"),
        pl.col("price").mean().alias("session_avg_price")
    )
    
    session_stats_path = os.path.join(temp_dir, "session_stats.parquet")
    session_stats.sink_parquet(session_stats_path)
    
    print("Step 3: Calculating Hesitation Time...")
    hesitation_step1 = lf.filter(
        pl.col("event_type").is_in(["view", "cart", "purchase"])
    ).group_by("user_session", "product_id").agg(
        pl.col("event_time").filter(pl.col("event_type") == "view").min().alias("first_view_time"),
        pl.col("event_time").filter(pl.col("event_type").is_in(["cart", "purchase"])).min().alias("first_action_time")
    )
    hesitation_step1_path = os.path.join(temp_dir, "hesitation_1.parquet")
    hesitation_step1.sink_parquet(hesitation_step1_path)
    
    hesitation_step2 = pl.scan_parquet(hesitation_step1_path).with_columns(
        (pl.col("first_action_time") - pl.col("first_view_time")).dt.total_seconds().alias("hesitation_time")
    ).group_by("user_session").agg(
        pl.col("hesitation_time").mean().alias("session_avg_hesitation_time")
    )
    hesitation_path = os.path.join(temp_dir, "hesitation_final.parquet")
    hesitation_step2.sink_parquet(hesitation_path)
    
    print("Step 4: Calculating Category Focus...")
    cat_views_step1 = lf.filter(pl.col("event_type") == "view").group_by("user_session", "category_code").agg(
        pl.len().alias("cat_view_count")
    )
    cat_views_step1_path = os.path.join(temp_dir, "cat_views_1.parquet")
    cat_views_step1.sink_parquet(cat_views_step1_path)
    
    max_cat_views = pl.scan_parquet(cat_views_step1_path).group_by("user_session").agg(
        pl.col("cat_view_count").max().alias("max_cat_view_count")
    )
    max_cat_views_path = os.path.join(temp_dir, "max_cat_views.parquet")
    max_cat_views.sink_parquet(max_cat_views_path)
    
    print("Step 5: Joining Features...")
    user_lf = pl.scan_parquet(user_stats_path)
    session_lf = pl.scan_parquet(session_stats_path)
    hesitation_lf = pl.scan_parquet(hesitation_path)
    max_cat_lf = pl.scan_parquet(max_cat_views_path)
    
    final_features = (
        session_lf
        .join(max_cat_lf, on="user_session", how="left")
        .join(hesitation_lf, on="user_session", how="left")
        .join(user_lf, on="user_id", how="left")
    )
    
    final_features = final_features.with_columns(
        (pl.col("max_cat_view_count") / pl.col("total_views")).fill_nan(0.0).alias("category_focus_ratio"),
        (pl.col("session_avg_price") / pl.col("hist_avg_purchase_price")).fill_nan(0.0).alias("relative_price_anchor"),
        (pl.col("session_start").dt.weekday() >= 6).alias("is_weekend"),
        pl.when(pl.col("session_start").dt.hour() < 7).then(pl.lit("late-night"))
          .when(pl.col("session_start").dt.hour() < 18).then(pl.lit("working-hours"))
          .otherwise(pl.lit("evening")).alias("time_of_day")
    ).drop(["total_views", "max_cat_view_count", "session_start", "session_avg_price"])
    
    final_features = final_features.with_columns(
        pl.col("session_avg_hesitation_time").fill_null(0.0),
        pl.col("hist_avg_purchase_price").fill_null(0.0),
        pl.col("relative_price_anchor").fill_null(0.0),
        pl.col("category_focus_ratio").fill_null(0.0),
        pl.col("return_visit_frequency").fill_null(1),
        pl.col("cart_churn_rate").fill_null(0.0)
    )
    
    print("Step 6: Executing and writing final output...")
    try:
        final_features.sink_parquet(output_path)
        print("Written successfully using sink_parquet().")
    except Exception as e:
        print(f"sink_parquet() not fully supported for this join graph. Falling back to collect(streaming=True)...")
        final_features.collect(streaming=True).write_parquet(output_path)
        print("Written successfully using collect(streaming=True).")
        
    print("Cleaning up intermediate files...")
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("Behavioral feature engineering complete.")

if __name__ == "__main__":
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    
    input_files = [f for f in glob.glob(os.path.join(root_dir, "data", "processed", "*.parquet")) if "behavioral_features" not in f]
    
    if not input_files:
        print("No raw event Parquet files found in data/processed/")
        exit(1)
        
    output_path = os.path.join(root_dir, "data", "processed", "behavioral_features.parquet")
    build_features(input_files, output_path)
