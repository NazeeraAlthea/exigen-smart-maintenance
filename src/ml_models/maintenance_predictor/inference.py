import os
import joblib
import pandas as pd
import numpy as np

# Load model secara global agar tidak reload per request
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../models/maintenance_predictor_rul_mlp.pkl"))
mlp_pipeline = None

try:
    if os.path.exists(MODEL_PATH):
        mlp_pipeline = joblib.load(MODEL_PATH)
        print("✅ MLP RUL Model loaded successfully for inference.")
    else:
        print(f"⚠️ Warning: {MODEL_PATH} not found. Using mock RUL prediction.")
except Exception as e:
    print(f"⚠️ Error loading MLP Model: {e}. Using mock RUL prediction.")

def predict_rul(features: dict) -> float:
    """
    Melakukan prediksi RUL (Remaining Useful Life) berdasarkan fitur aset menggunakan model MLP.
    Features yang diharapkan sesuai dengan training model (contoh: Biaya_Total_Log, Total_Komplain, dll)
    """
    if mlp_pipeline is not None:
        try:
            encoder = mlp_pipeline['encoder']
            scaler = mlp_pipeline['scaler']
            model = mlp_pipeline['model']
            fitur_x = mlp_pipeline['features']
            
            # Buat DataFrame dari features input
            df = pd.DataFrame([features])
            
            # Preprocessing: tambahkan fitur log jika hanya fitur dasar yang dikirim
            if 'Biaya_Total' in df.columns and 'Biaya_Total_Log' not in df.columns:
                df['Biaya_Total_Log'] = np.log1p(df['Biaya_Total'])
            if 'Biaya_Mean' in df.columns and 'Biaya_Mean_Log' not in df.columns:
                df['Biaya_Mean_Log'] = np.log1p(df['Biaya_Mean'])
                
            # Pastikan semua fitur di fitur_x ada di df (jika tidak ada, beri default/0)
            for col in fitur_x:
                if col not in df.columns:
                    if col in ['Kategori', 'Sub Kategori', 'Tipe', 'Merek', 'Tingkat Kekritisan']:
                        df[col] = '-'
                    else:
                        df[col] = 0.0
            
            # Pilih hanya fitur yang dibutuhkan sesuai urutan training
            df_input = df[fitur_x].copy()
            
            # Transformasikan kolom kategori menggunakan TargetEncoder
            kat_cols = ['Kategori', 'Sub Kategori', 'Tipe', 'Merek', 'Tingkat Kekritisan']
            df_input[kat_cols] = encoder.transform(df_input[kat_cols])
            
            # Standarisasi fitur dengan StandardScaler
            df_scaled = scaler.transform(df_input)
            
            # Lakukan prediksi
            pred = model.predict(df_scaled)
            return round(float(pred[0]), 2)
        except Exception as e:
            print(f"Error during real MLP prediction: {e}")
            # Fallback jika terjadi error
            
    # Mock fallback
    base_rul = 1000.0
    # Coba baca fitur dengan variasi penulisan case-insensitive
    total_komplain = features.get("Total_Komplain", features.get("jumlah_kerusakan", 0))
    biaya_total = features.get("Biaya_Total", features.get("biaya_perbaikan_kumulatif", 0))
    
    if total_komplain > 5:
        base_rul -= 300
    if biaya_total > 10000000:
        base_rul -= 200
        
    return float(max(10, base_rul))
