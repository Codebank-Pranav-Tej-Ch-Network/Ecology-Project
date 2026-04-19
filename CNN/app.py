import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================================================
# 1. LOAD THE MODEL
# =====================================================================
try:
    logger.info("Loading Spatial CNN Model...")
    model = tf.keras.models.load_model('spatial_model.keras')
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model. Ensure 'spatial_model.keras' is in the directory. Error: {e}")
    raise RuntimeError("Model loading failed.")

# =====================================================================
# 2. API CONFIGURATION
# =====================================================================
app = FastAPI(title="EcoCity Spatial Engine", version="2.0")

# Allow React frontend to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this to your specific frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================================
# 3. DATA SCHEMAS & MAPPINGS
# =====================================================================
class CityGridRequest(BaseModel):
    grid: list[list[str]]  # Expects a 2D array of characters e.g., [['H', 'G'], ['I', '.']]

LABEL_MAP = {'G': 0, 'H': 1, 'I': 2, 'B': 3, 'P': 4, '.': 5}
NUM_CHANNELS = 6

# =====================================================================
# 4. ENDPOINTS
# =====================================================================
@app.get("/")
def health_check():
    return {"status": "online", "model": "spatial_cnn_v1"}

@app.post("/simulate")
def simulate_city(request: CityGridRequest):
    try:
        grid_data = request.grid
        
        # 1. Validate Input
        if not grid_data or not grid_data[0]:
            raise HTTPException(status_code=400, detail="Grid cannot be empty")
            
        height = len(grid_data)
        width = len(grid_data[0])
        
        # 2. Dynamic One-Hot Encoding (Size Agnostic)
        # We create a tensor of shape (1, height, width, 6)
        tensor = np.zeros((1, height, width, NUM_CHANNELS), dtype=np.float32)
        
        for r in range(height):
            if len(grid_data[r]) != width:
                raise HTTPException(status_code=400, detail="Grid must be perfectly rectangular")
                
            for c in range(width):
                char = grid_data[r][c].upper()
                if char not in LABEL_MAP:
                    # Default unknown tiles to empty '.'
                    char = '.'
                channel_idx = LABEL_MAP[char]
                tensor[0, r, c, channel_idx] = 1.0

        # 3. Run Inference
        # Because we used Global Average Pooling, the model doesn't care about 'height' or 'width'
        prediction = model.predict(tensor, verbose=0)
        final_score = float(prediction[0][0])
        
        # Cap score between 0 and 100 for UX safety
        final_score = max(0.0, min(100.0, final_score))

        return {
            "success": True,
            "grid_dimensions": {"width": width, "height": height},
            "sustainability_score": round(final_score, 2),
            "message": "Spatial analysis complete."
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================================
# 5. EXECUTION
# =====================================================================
if __name__ == "__main__":
    print("Starting Uvicorn server on port 8000...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
