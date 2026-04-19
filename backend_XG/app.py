from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os

# ==========================================
# 1. INITIALIZE FASTAPI & CORS
# ==========================================
app = FastAPI(
    title="Urban Health Score DSS API",
    description="Backend engine for predicting Urban Ecological Health Index (UEHI) using Machine Learning.",
    version="1.0.0"
)

# Allow the frontend dashboard to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For production, change to the frontend's actual URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 2. LOAD AI MODEL & SCALER
# ==========================================
MODEL_PATH = 'urban_health_rf_model.pkl'
SCALER_PATH = 'urban_health_scaler.pkl'

# Global variables to hold the loaded files
rf_model = None
scaler = None

# We use the startup event to load models when the server boots
@app.on_event("startup")
def load_models():
    global rf_model, scaler
    try:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
            raise FileNotFoundError("Model or Scaler .pkl files are missing from the directory!")
            
        print("⏳ Loading Machine Learning model and Scaler...")
        rf_model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("✅ Models loaded successfully. The AI Engine is ready!")
    except Exception as e:
        print(f"❌ CRITICAL ERROR LOADING AI: {e}")

# ==========================================
# 3. DEFINE FRONTEND DATA SCHEMA
# ==========================================
class GridData(BaseModel):
    Greenness: float = Field(..., description="Percentage of green cover", example=45.5)
    Urban_Density: float = Field(..., description="Built-up density metric", example=10.2)
    UHI: float = Field(..., description="Urban Heat Island effect index", example=-0.5)
    LSTVI: float = Field(..., description="Land Surface Temperature Vegetation Index", example=0.05)
    PM25: float = Field(..., description="Particulate matter 2.5 concentration", example=20.1)

# ==========================================
# 4. API ENDPOINTS
# ==========================================

# Health Check Endpoint (For frontend to verify connection)
@app.get("/")
def health_check():
    return {"status": "online", "message": "Urban Health AI Engine is running perfectly."}

# The main Prediction Endpoint
@app.post("/predict")
def predict_urban_health(data: GridData):
    # Failsafe in case the model didn't load properly on startup
    if rf_model is None or scaler is None:
        raise HTTPException(status_code=500, detail="AI Model is not loaded on the server.")

    try:
        # 1. Extract features in the EXACT order they were trained
        features = [[
            data.Greenness, 
            data.Urban_Density, 
            data.UHI, 
            data.LSTVI, 
            data.PM25
        ]]
        
        # 2. Scale the data
        features_scaled = scaler.transform(features)
        
        # 3. Predict the raw decimal score
        raw_score = rf_model.predict(features_scaled)[0]
        
        # 4. Round to nearest integer and bind to 1-5 scale
        final_score = int(np.round(np.clip(raw_score, 1, 5)))
        
        # 5. Map to Category
        categories = {
            1: "Severely Unhealthy", 
            2: "Unhealthy", 
            3: "Moderate", 
            4: "Healthy", 
            5: "Very Healthy"
        }
        
        # 6. Send the JSON payload back to the dashboard
        return {
            "success": True,
            "inputs_received": data.dict(),
            "prediction": {
                "raw_score": round(raw_score, 2),
                "final_score": final_score,
                "category": categories[final_score]
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")
