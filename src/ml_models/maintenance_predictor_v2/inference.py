import os
import joblib
import pandas as pd
import numpy as np

# Load model secara global agar tidak reload per request
# Menggunakan model MLP v2 bebas leak secara default
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../models/maintenance_predictor_v2_mlp.pkl"))
mlp_pipeline = None

try:
    if os.path.exists(MODEL_PATH):
        mlp_pipeline = joblib.load(MODEL_PATH)
        print("[INFO] MLP v2 RUL Model loaded successfully for inference.")
    else:
        print(f"[WARN] Warning: {MODEL_PATH} not found. Using mock RUL prediction.")
except Exception as e:
    print(f"[ERROR] Error loading MLP Model: {e}. Using mock RUL prediction.")

def predict_rul(features: dict) -> float:
    """
    Melakukan prediksi RUL (Remaining Useful Life) berdasarkan fitur aset menggunakan model MLP v2.
    Features yang diharapkan sesuai dengan training model (contoh: biaya_total, total_komplain, dll)
    """
    if mlp_pipeline is not None:
        try:
            pipeline = mlp_pipeline['pipeline']
            fitur_x = mlp_pipeline['features']
            mean_cost_tipe = mlp_pipeline.get('mean_cost_tipe', {})
            
            # Normalisasi keys dari input features ke format training (Capitalized & Space)
            normalized_features = {}
            key_mapping = {
                'kategori': 'Kategori',
                'subkategori': 'Sub Kategori',
                'sub_kategori': 'Sub Kategori',
                'subKategori': 'Sub Kategori',
                'tipe': 'Tipe',
                'merek': 'Merek',
                'tingkatkekritisan': 'Tingkat Kekritisan',
                'tingkat_kekritisan': 'Tingkat Kekritisan',
                'tingkatKekritisan': 'Tingkat Kekritisan',
                'jumlah_kerusakan': 'Total_Komplain',
                'total_komplain': 'Total_Komplain',
                'totalKomplain': 'Total_Komplain',
                'biaya_perbaikan_kumulatif': 'Biaya_Total',
                'biaya_total': 'Biaya_Total',
                'biayaTotal': 'Biaya_Total',
                'biaya_mean': 'Biaya_Mean',
                'biayaMean': 'Biaya_Mean',
                'hari_antar_komplain_mean': 'Hari_Antar_Komplain_Mean',
                'hariAntarKomplainMean': 'Hari_Antar_Komplain_Mean',
                'umur_saat_komplain_terakhir': 'Age_At_Complaint',
                'umurSaatKomplainTerakhir': 'Age_At_Complaint',
                'age_at_complaint': 'Age_At_Complaint',
                'umur': 'Age_At_Complaint',
                'umur_saat_ini': 'Age_At_Complaint',
                'frekuensi_hari': 'Frekuensi_Hari',
                'frekuensiHari': 'Frekuensi_Hari',
                'complaint_velocity': 'Complaint_Velocity',
                'complaintVelocity': 'Complaint_Velocity',
                'cost_deviation_ratio': 'Cost_Deviation_Ratio',
                'costDeviationRatio': 'Cost_Deviation_Ratio'
            }
            
            # Pindahkan data ke dict dengan key standar
            for k, v in features.items():
                std_key = key_mapping.get(k, k)
                normalized_features[std_key] = v
            
            # Buat DataFrame dari features input yang sudah dinormalisasi
            df = pd.DataFrame([normalized_features])
            
            # Preprocessing: tambahkan fitur log jika hanya fitur dasar yang dikirim
            if 'Biaya_Total' in df.columns and 'Biaya_Total_Log' not in df.columns:
                df['Biaya_Total_Log'] = np.log1p(df['Biaya_Total'])
            if 'Biaya_Mean' in df.columns and 'Biaya_Mean_Log' not in df.columns:
                df['Biaya_Mean_Log'] = np.log1p(df['Biaya_Mean'])
                
            global_mean_cost = mlp_pipeline.get('global_mean_cost', 1.0)
            
            # Hitung Cost_Deviation_Ratio secara dinamis jika tidak dikirim
            if 'Cost_Deviation_Ratio' not in df.columns:
                biaya_total_raw = df.get('Biaya_Total', pd.Series([0.0]))[0]
                tipe_val = df.get('Tipe', pd.Series(['-']))[0]
                tipe_mean = mean_cost_tipe.get(tipe_val, global_mean_cost)
                if tipe_mean == 0 or pd.isna(tipe_mean):
                    tipe_mean = global_mean_cost
                df['Cost_Deviation_Ratio'] = biaya_total_raw / tipe_mean
                
            # Pastikan semua fitur di fitur_x ada di df (jika tidak ada, beri default/0)
            for col in fitur_x:
                if col not in df.columns:
                    if col in ['Kategori', 'Sub Kategori', 'Tipe', 'Merek', 'Tingkat Kekritisan']:
                        df[col] = '-'
                    else:
                        df[col] = 0.0
            
            # Pilih hanya fitur yang dibutuhkan sesuai urutan training
            df_input = df[fitur_x].copy()
            
            # Lakukan prediksi (prediksi lifespan) menggunakan pipeline (encoder, scaler, model otomatis ditangani)
            pred_lifespan = pipeline.predict(df_input)
            pred_lifespan_val = float(pred_lifespan[0])
            
            # Hitung RUL = Lifespan - Age_At_Complaint
            age_val = float(df_input['Age_At_Complaint'].values[0])
            pred_rul = pred_lifespan_val - age_val
            
            return round(max(0.0, pred_rul), 2)
        except Exception as e:
            print(f"Error during real MLP prediction: {e}")
            # Fallback jika terjadi error
            
    # Mock fallback
    base_rul = 1000.0
    total_komplain = features.get("Total_Komplain", features.get("jumlah_kerusakan", 0))
    biaya_total = features.get("Biaya_Total", features.get("biaya_perbaikan_kumulatif", 0))
    
    if total_komplain > 5:
        base_rul -= 300
    if biaya_total > 10000000:
        base_rul -= 200
        
    return float(max(10, base_rul))
