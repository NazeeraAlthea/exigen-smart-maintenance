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
    Mendukung dua skenario umur:
    1. RUL Setelah Komplain: umur_saat_ini (Age_Today) sama dengan umur_saat_komplain_terakhir (Age_At_Complaint).
    2. RUL Cron: umur_saat_ini (Age_Today) bertambah, sedangkan umur_saat_komplain_terakhir (Age_At_Complaint) tetap.
    """
    # Resolve age values
    age_today_keys = ['umur_saat_ini', 'umurSaatIni', 'current_age', 'age_today', 'umur']
    age_today_val = None
    for k in age_today_keys:
        if k in features:
            try:
                age_today_val = float(features[k])
                break
            except (ValueError, TypeError):
                continue
                
    age_at_complaint_keys = ['umur_saat_komplain_terakhir', 'umurSaatKomplainTerakhir', 'age_at_complaint', 'umur_saat_komplain', 'age_at_last_complaint']
    age_at_complaint_val = None
    for k in age_at_complaint_keys:
        if k in features:
            try:
                age_at_complaint_val = float(features[k])
                break
            except (ValueError, TypeError):
                continue
                
    # Fallbacks if one or both are missing
    if age_today_val is None and age_at_complaint_val is not None:
        age_today_val = age_at_complaint_val
    elif age_at_complaint_val is None and age_today_val is not None:
        age_at_complaint_val = age_today_val
    elif age_today_val is None and age_at_complaint_val is None:
        age_today_val = 0.0
        age_at_complaint_val = 0.0

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
            
            # Paksa atur Age_At_Complaint berdasarkan resolusi umur di atas
            normalized_features['Age_At_Complaint'] = age_at_complaint_val
            
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
            
            # Hitung RUL = Lifespan - Age_Today
            pred_rul = pred_lifespan_val - age_today_val
            
            return round(max(0.0, pred_rul), 2)
        except Exception as e:
            print(f"Error during real MLP prediction: {e}")
            # Fallback jika terjadi error
            
    # Mock fallback
    base_rul = 1000.0 - age_today_val
    total_komplain = features.get("Total_Komplain", features.get("jumlah_kerusakan", features.get("total_komplain", 0)))
    biaya_total = features.get("Biaya_Total", features.get("biaya_perbaikan_kumulatif", features.get("biaya_total", 0)))
    
    if total_komplain > 5:
        base_rul -= 300
    if biaya_total > 10000000:
        base_rul -= 200
        
    return float(max(10.0, base_rul))


def predict_rul_batch(features_list: list[dict]) -> list[float]:
    """
    Melakukan prediksi RUL (Remaining Useful Life) secara batch untuk efisiensi tinggi.
    Menerima list of dict berisi features untuk masing-masing aset.
    Mengmengembalikan list of float berisi prediksi RUL masing-masing aset.
    """
    if not features_list:
        return []
        
    def resolve_ages(feat):
        age_today_keys = ['umur_saat_ini', 'umurSaatIni', 'current_age', 'age_today', 'umur']
        age_today_val = None
        for k in age_today_keys:
            if k in feat:
                try:
                    age_today_val = float(feat[k])
                    break
                except (ValueError, TypeError):
                    continue
                    
        age_at_complaint_keys = ['umur_saat_komplain_terakhir', 'umurSaatKomplainTerakhir', 'age_at_complaint', 'umur_saat_komplain', 'age_at_last_complaint']
        age_at_complaint_val = None
        for k in age_at_complaint_keys:
            if k in feat:
                try:
                    age_at_complaint_val = float(feat[k])
                    break
                except (ValueError, TypeError):
                    continue
                    
        if age_today_val is None and age_at_complaint_val is not None:
            age_today_val = age_at_complaint_val
        elif age_at_complaint_val is None and age_today_val is not None:
            age_at_complaint_val = age_today_val
        elif age_today_val is None and age_at_complaint_val is None:
            age_today_val = 0.0
            age_at_complaint_val = 0.0
            
        return age_today_val, age_at_complaint_val

    if mlp_pipeline is None:
        results = []
        for feat in features_list:
            age_today_val, _ = resolve_ages(feat)
            base_rul = 1000.0 - age_today_val
            total_komplain = feat.get("Total_Komplain", feat.get("jumlah_kerusakan", feat.get("total_komplain", 0)))
            biaya_total = feat.get("Biaya_Total", feat.get("biaya_perbaikan_kumulatif", feat.get("biaya_total", 0)))
            if total_komplain > 5:
                base_rul -= 300
            if biaya_total > 10000000:
                base_rul -= 200
            results.append(float(max(10.0, base_rul)))
        return results

    try:
        pipeline = mlp_pipeline['pipeline']
        fitur_x = mlp_pipeline['features']
        mean_cost_tipe = mlp_pipeline.get('mean_cost_tipe', {})
        global_mean_cost = mlp_pipeline.get('global_mean_cost', 1.0)
        
        normalized_list = []
        age_today_vals = []
        
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
            'frekuensi_hari': 'Frekuensi_Hari',
            'frekuensiHari': 'Frekuensi_Hari',
            'complaint_velocity': 'Complaint_Velocity',
            'complaintVelocity': 'Complaint_Velocity',
            'cost_deviation_ratio': 'Cost_Deviation_Ratio',
            'costDeviationRatio': 'Cost_Deviation_Ratio'
        }
        
        for feat in features_list:
            age_today_val, age_at_complaint_val = resolve_ages(feat)
            age_today_vals.append(age_today_val)
            
            norm_feat = {}
            for k, v in feat.items():
                std_key = key_mapping.get(k, k)
                norm_feat[std_key] = v
            norm_feat['Age_At_Complaint'] = age_at_complaint_val
            normalized_list.append(norm_feat)
            
        df = pd.DataFrame(normalized_list)
        
        # Log-transform variables
        if 'Biaya_Total' in df.columns:
            df['Biaya_Total_Log'] = np.log1p(df['Biaya_Total'].fillna(0.0))
        else:
            df['Biaya_Total'] = 0.0
            df['Biaya_Total_Log'] = 0.0
            
        if 'Biaya_Mean' in df.columns:
            df['Biaya_Mean_Log'] = np.log1p(df['Biaya_Mean'].fillna(0.0))
        else:
            df['Biaya_Mean'] = 0.0
            df['Biaya_Mean_Log'] = 0.0
            
        # Cost Deviation Ratio calculation
        if 'Cost_Deviation_Ratio' not in df.columns:
            if 'Tipe' in df.columns:
                tipe_means = df['Tipe'].map(mean_cost_tipe).fillna(global_mean_cost)
                tipe_means = tipe_means.replace(0, global_mean_cost).fillna(global_mean_cost)
                df['Cost_Deviation_Ratio'] = df['Biaya_Total'] / tipe_means
            else:
                df['Cost_Deviation_Ratio'] = 0.0
                
        # Fill missing features from fitur_x with default values
        for col in fitur_x:
            if col not in df.columns:
                if col in ['Kategori', 'Sub Kategori', 'Tipe', 'Merek', 'Tingkat Kekritisan']:
                    df[col] = '-'
                else:
                    df[col] = 0.0
                    
        # Select and copy inputs in correct order
        df_input = df[fitur_x].copy()
        
        # Predict lifespan
        pred_lifespan = pipeline.predict(df_input)
        
        # Subtract age_today (element-wise) to get RUL
        results = []
        for pred_ls, age_td in zip(pred_lifespan, age_today_vals):
            rul = pred_ls - age_td
            results.append(round(max(0.0, float(rul)), 2))
            
        return results
        
    except Exception as e:
        print(f"Error during batch MLP prediction: {e}")
        # Fallback to single predictions
        results = []
        for feat in features_list:
            results.append(predict_rul(feat))
        return results
