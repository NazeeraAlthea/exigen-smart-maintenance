import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import sys

# Daftarkan folder ml_models ke path agar utils/inference bisa di-import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../ml_models/ticketing")))
from ml_models.ticketing.inference import ExigenInferenceEngine

app = FastAPI(
    title="Exigen Smart Helpdesk NLP Engine",
    description="API Gateway Fase 1 Laporan Tiket untuk Komunikasi dengan Next.js Backend",
    version="1.0.0"
)

# Inisialisasi engine saat API dinyalakan
engine = ExigenInferenceEngine()

# Model validasi data untuk Jose (Next.js TypeScript Request Body)
class TextTicketRequest(BaseModel):
    text_complaint: str

# Model validasi data RUL
class RULRequest(BaseModel):
    asset_id: str
    features: dict

class RULResponse(BaseModel):
    asset_id: str
    predicted_rul_days: float
    status: str

@app.post("/api/predict/text", summary="Menerima keluhan langsung berupa ketikan teks (WhatsApp/Web Form)")
async def predict_from_text(payload: TextTicketRequest):
    try:
        result = engine.predict_ticket(input_data=payload.text_complaint, mode="text")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict/voice", summary="Menerima unggahan file rekaman audio (.wav/.mp3)")
async def predict_from_voice(file: UploadFile = File(...)):
    # Validasi ekstensi berkas
    if not file.filename.endswith(('.wav', '.mp3', '.ogg', '.m4a')):
        raise HTTPException(status_code=400, detail="Format berkas audio tidak didukung.")
        
    temp_dir = "temp_audio"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, file.filename)
    
    # Simpan file sementara yang dikirim Jose dari Next.js
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        result = engine.predict_ticket(input_data=temp_file_path, mode="voice")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Bersihkan file sampah agar storage server tidak penuh
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/api/transcribe/voice", summary="Mentranskripsi file rekaman audio menjadi teks")
async def transcribe_voice(file: UploadFile = File(...)):
    if not file.filename.endswith(('.wav', '.mp3', '.ogg', '.m4a', '.webm')):
        raise HTTPException(status_code=400, detail="Format berkas audio tidak didukung.")
        
    temp_dir = "temp_audio"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, file.filename)
    
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        text = engine.transkripsi_audio(temp_file_path)
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/api/predict/rul", summary="Prediksi Remaining Useful Life (RUL) Aset")
@app.post("/predict/rul", summary="Prediksi Remaining Useful Life (RUL) Aset (Legacy)")
async def predict_asset_rul(payload: RULRequest):
    try:
        from ml_models.maintenance_predictor.inference import predict_rul
        rul = predict_rul(payload.features)
        return RULResponse(
            asset_id=payload.asset_id,
            predicted_rul_days=rul,
            status="Sukses"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)