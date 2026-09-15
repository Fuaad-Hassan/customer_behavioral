# End-to-End Behavioral Data Science & Inference Pipeline

## Overview
This repository implements an end-to-end MLOps pipeline designed to ingest e-commerce clickstream data, engineer psychographic features, train an XGBoost model tracked by MLflow, and serve real-time behavioral nudges via a FastAPI/Docker microservice. It serves as a structural architectural Proof of Concept (PoC) demonstrating how to construct a continuous data science pipeline from raw logs to containerized inference.

## Tech Stack
* **Data Processing:** Polars
* **Machine Learning:** Scikit-Learn, XGBoost
* **MLOps Tracking:** MLflow (SQLite backend)
* **Serving Layer:** FastAPI, Uvicorn, Docker
* **Testing & Evaluation:** Pytest, SciPy

## Pipeline Architecture
* **Feature Engineering:** Utilizes Polars streaming capabilities for out-of-core processing. The pipeline aggregates raw session logs into numerical matrices representing behavioral signals, such as hesitation time between events and category focus ratios.
* **Segmentation:** Applies K-Means clustering to map users into three distinct behavioral archetypes: Impulse Buyers, Researchers, and Bargain Hunters.
* **Propensity Modeling:** Trains an XGBoost binary classifier to predict conversion propensity. The training lifecycle, parameters, and physical model artifacts are logged to a local SQLite MLflow registry.
* **Serving & Nudge Engine:** A containerized FastAPI endpoint loads the latest registered model artifact. It intercepts payloads with a predicted probability `< 0.50` and prescribes a hardcoded psychological nudge based on the user's psychographic cluster.

## Architectural Reality Check (Limitations & Trade-offs)
This project is an architectural PoC, and several intentional limitations were introduced to prioritize pipeline construction over model optimization:
* **Synthetic Target Variable:** The model predicts engagement intensity rather than guaranteed sales. Because explicit purchase labels were aggregated away during early processing, the target variable `is_purchased` was synthetically derived from the top 15% of `session_depth`.
* **Static Clustering:** The K-Means segmentation was forced at $k=3$ based on a predefined business hypothesis, without empirical validation (e.g., Elbow Method). Furthermore, the centroids are static and do not continuously adapt to real-time behavioral drift.
* **Zero Variance Feature:** The engineered feature `cart_churn_rate` returns `0.0` globally. This is not a logical bug in the pipeline, but a constraint of the underlying Kaggle dataset, which lacked `remove_from_cart` events.
* **Model Conservatism:** The XGBoost model exhibits high precision (~88%) but low recall (~37%). It is mathematically conservative, missing a majority of potential buyers in order to aggressively protect margins and avoid deploying unnecessary discount nudges.

## Evaluation & Testing
The system relies on a two-pronged evaluation strategy:
* **CI/CD Readiness:** The pipeline is fortified with `pytest` integration tests that assert data aggregation logic, boundary conditions, and Pydantic API schema validation to prevent garbage-in, garbage-out (GIGO) errors.
* **A/B Testing Framework:** A standalone SciPy script (`src/models/evaluate.py`) utilizes the Chi-Square Test of Independence to mathematically evaluate the statistical significance and offline ROI of future deployed nudges.

## How to Run

**1. Environment Setup**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Execute the Pipeline**
```bash
# Run feature engineering and clustering (assuming raw data exists in data/raw)
python src/features/build_features.py
python src/models/clustering.py

# Train the model and log to MLflow
python src/models/train_propensity.py
```

**3. Serve the Inference Engine**
```bash
# Build and run the FastAPI container
cd docker
docker compose up --build -d
```

**4. Test the Endpoint**
```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
  "session_avg_hesitation_time": 10.5,
  "return_visit_frequency": 1,
  "hist_avg_purchase_price": 50.0,
  "cart_churn_rate": 0.0,
  "category_focus_ratio": 0.95,
  "relative_price_anchor": 1.0,
  "is_weekend": 1,
  "psychographic_cluster": 1
}'
```
