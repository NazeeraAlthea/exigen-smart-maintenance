from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from src.ml_models.ticketing.inference import predict_ticket_from_text
from src.ml_models.maintenance_predictor.inference import predict_rul

app = FastAPI(
    title="Exigen Smart Maintenance ML API",
    description="Microservice untuk NLP Ticketing dan Prediksi RUL",
    version="1.0.0"
)

# ---- SCHEMAS ----
class TicketRequest(BaseModel):
    id_laporan: str
    teks_keluhan: str
    sumber_input: str = "Web"
    waktu_lapor: str = ""

class RULRequest(BaseModel):
    asset_id: str
    features: Dict[str, Any]

class RULResponse(BaseModel):
    asset_id: str
    predicted_rul_days: float
    status: str

# ---- ENDPOINTS ----

@app.get("/")
def read_root():
    return {"message": "Welcome to Exigen ML API. Endpoints: /predict/ticket, /predict/rul"}

@app.post("/predict/ticket")
def predict_ticket(req: TicketRequest):
    try:
        result = predict_ticket_from_text(req.id_laporan, req.teks_keluhan)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/rul", response_model=RULResponse)
def predict_asset_rul(req: RULRequest):
    try:
        rul = predict_rul(req.features)
        return RULResponse(
            asset_id=req.asset_id,
            predicted_rul_days=rul,
            status="Sukses"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
