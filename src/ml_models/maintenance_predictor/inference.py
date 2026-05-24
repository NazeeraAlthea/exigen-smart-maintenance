import os
import joblib
import pandas as pd

# Load model secara global agar tidak reload per request
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../../models/rf_model.pkl")
rf_model = None

try:
    if os.path.exists(MODEL_PATH):
        rf_model = joblib.load(MODEL_PATH)
        print("✅ RF Model loaded successfully for inference.")
    else:
        print("⚠️ Warning: rf_model.pkl not found. Using mock RUL prediction.")
except Exception as e:
    print(f"⚠️ Error loading RF Model: {e}. Using mock RUL prediction.")

def predict_rul(features: dict) -> float:
    """
    Melakukan prediksi RUL (Remaining Useful Life) berdasarkan fitur aset.
    Features yang diharapkan sesuai dengan training model (contoh: biaya_perbaikan_kumulatif, jumlah_kerusakan, dll)
    """
    if rf_model is not None:
        # Konversi dict ke DataFrame karena scikit-learn membutuhkan format matriks 2D
        df = pd.DataFrame([features])
        
        # Di sini Anda mungkin butuh preprocessing tambahan seperti scaling atau one-hot encoding
        # Tergantung dari pipeline training Anda (apakah pipeline sudah include preprocessor)
        
        try:
            pred = rf_model.predict(df)
            return round(float(pred[0]), 2)
        except Exception as e:
            print(f"Error during real prediction: {e}")
            # Fallback ke mock jika fitur tidak cocok
    
    # Mock fallback
    # Jika tidak ada model, kita buat kalkulasi pura-pura
    base_rul = 1000
    if features.get("jumlah_kerusakan", 0) > 5:
        base_rul -= 300
    if features.get("biaya_perbaikan_kumulatif", 0) > 10000000:
        base_rul -= 200
        
    return float(max(10, base_rul))
