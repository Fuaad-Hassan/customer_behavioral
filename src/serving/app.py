import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

app = FastAPI(
    title="Behavioral Inference Engine",
    description="Real-time Psychographic Nudge & Propensity API",
    version="1.0.0"
)

RUN_ID = "fc6330681d504c62b4d003a1868aedf1"
TRACKING_URI = "sqlite:///mlflow.db"

mlflow.set_tracking_uri(TRACKING_URI)

try:
    print(f"Loading Production XGBoost model from Run ID: {RUN_ID}...")
    model_uri = f"runs:/{RUN_ID}/xgb_propensity_model"
    xgb_model = mlflow.xgboost.load_model(model_uri)
    print("Model loaded successfully and ready for inference.")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load MLflow model. Details: {e}")
    xgb_model = None

class BehavioralFeatures(BaseModel):
    session_avg_hesitation_time: float
    return_visit_frequency: float
    hist_avg_purchase_price: float
    cart_churn_rate: float
    category_focus_ratio: float
    relative_price_anchor: float
    is_weekend: float
    psychographic_cluster: int

@app.post("/predict")
def predict_propensity(features: BehavioralFeatures):
    """
    Receives real-time behavioral features, calculates purchase probability,
    and deploys a psychological nudge if abandonment is predicted.
    """
    if xgb_model is None:
        raise HTTPException(status_code=500, detail="Inference engine offline: Model failed to load.")
        
    try:
        data_dict = features.model_dump() if hasattr(features, "model_dump") else features.dict()
        input_data = pd.DataFrame([data_dict])
        
        prob = float(xgb_model.predict_proba(input_data)[0][1])
        nudge = "None (No Intervention Required)"
        
        if prob < 0.5:
            cluster = features.psychographic_cluster
            if cluster == 1:
                nudge = "Deploy Scarcity Timer"
            elif cluster == 0:
                nudge = "Display Technical Review Comparison"
            elif cluster == 2:
                nudge = "Trigger Price Drop Pop-up"
            else:
                nudge = "Deploy Generic Intervention"
                
        return {
            "purchase_probability": prob,
            "predicted_abandonment": bool(prob < 0.5),
            "prescriptive_nudge": nudge
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
