import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from src.serving.app import app

# Initialize the testing client mapping to our FastAPI application
client = TestClient(app)

@patch("src.serving.app.xgb_model")
def test_predict_abandonment_scarcity_nudge(mock_xgb):
    """
    Validates that the Nudge Engine correctly routes an Impulse Buyer (Cluster 1)
    to a Scarcity Timer when the model predicts abandonment (prob < 0.5).
    """
    # Mocking XGBoost's predict_proba to deterministically return an abandonment probability
    # predict_proba returns [[prob_class_0, prob_class_1]]. We set prob_class_1 to 0.2 (< 0.5).
    mock_xgb.predict_proba.return_value = [[0.8, 0.2]]
    
    payload = {
        "session_avg_hesitation_time": 12.5,
        "return_visit_frequency": 1.0,
        "hist_avg_purchase_price": 50.0,
        "cart_churn_rate": 0.0,
        "category_focus_ratio": 1.0,
        "relative_price_anchor": 1.0,
        "is_weekend": 0.0,
        "psychographic_cluster": 1  # 1 = Impulse Buyer
    }
    
    response = client.post("/predict", json=payload)
    
    # Assert successful API communication
    assert response.status_code == 200
    
    data = response.json()
    # Assert model evaluation constraint
    assert data["predicted_abandonment"] is True
    # Assert exact Nudge Engine routing logic
    assert data["prescriptive_nudge"] == "Deploy Scarcity Timer"


def test_api_schema_validation_rejection():
    """
    Verifies the API boundary defenses by asserting that malformed requests
    missing required features are explicitly rejected before hitting the model.
    """
    # Missing 'psychographic_cluster' and 'is_weekend'
    payload = {
        "session_avg_hesitation_time": 120.5,
        "return_visit_frequency": 2.0
    }
    
    response = client.post("/predict", json=payload)
    
    # Assert HTTP 422 Unprocessable Entity
    assert response.status_code == 422
    data = response.json()
    
    # Ensure Pydantic details exactly which fields are missing
    error_msg = str(data["detail"])
    assert "psychographic_cluster" in error_msg
    assert "is_weekend" in error_msg
