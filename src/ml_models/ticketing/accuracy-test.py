import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

# 1. Fixing Python Path Routing agar aman membaca modul utils
CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
SRC_DIR = os.path.abspath(os.path.join(CURRENT_FILE_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from utils import super_clean_text

def check_pkl_accuracy():
    # 2. Tentukan lokasi data uji murni
    path_master = os.path.abspath(os.path.join(CURRENT_FILE_DIR, "../../../data/dataset_tiket_master_bersih.csv"))
    if not os.path.exists(path_master):
        print(f"❌ File dataset tidak ditemukan di: {path_master}")
        return
        
    print("⏳ Membaca dataset untuk pengujian...")
    df_master = pd.read_csv(path_master, sep='|', on_bad_lines='skip')
    kolom_target = ['tipe_aset', 'lokasi_gedung', 'lokasi_lantai', 'lokasi_zona', 'kategori_aset', 'severity']
    
    # Bersihkan data murni untuk testing
    df_master = df_master.dropna(subset=kolom_target + ['teks_keluhan_awam'])
    df_master = df_master[df_master['severity'].isin(['Ringan', 'Sedang', 'Berat', 'Fatal'])]
    
    # Ambil sampel data untuk uji performa (menggunakan porsi evaluasi teks bersih)
    X_test_clean = df_master['teks_keluhan_awam'].astype(str).apply(super_clean_text)
    y_test = df_master[kolom_target].values

    # 3. Daftar 3 file .pkl yang ingin kamu cek (Sesuaikan dengan nama file asli di foldermu)
    model_dir = os.path.abspath(os.path.join(CURRENT_FILE_DIR, "../../../models/ticketing"))
    list_models = [
        "ticket_v1.0.0_tfidf.pkl",
        "ticket_v1.1.1_tfidf.pkl",
        "ticket_v1.1.1.0_tfidf.pkl"  # <-- Ganti/sesuaikan nama filenya di sini jika berbeda
    ]
    
    print("\n" + "="*60)
    print("📊 HASIL KOMPARASI AKURASI MODEL LOKAL .PKL")
    print("="*60)

    for pkl_name in list_models:
        path_pkl = os.path.join(model_dir, pkl_name)
        
        if not os.path.exists(path_pkl):
            print(f"⚠️ File {pkl_name} tidak ditemukan di folder models, dilewati.")
            continue
            
        try:
            # Load model pkl secara lokal
            model_pipeline = joblib.load(path_pkl)
            
            # Lakukan prediksi pada data uji
            y_pred = model_pipeline.predict(X_test_clean)
            
            # Hitung metrik utama kelompok (Exact Match Ratio)
            exact_match = np.all(y_pred == y_test, axis=1).mean()
            
            print(f"🎯 Model: {pkl_name}")
            print(f"   ➡️ Exact Match Ratio (Akurasi Penuh 6 Entitas): {exact_match*100:.2f}%")
            
            # Opsional: Tampilkan akurasi parsial per kolom target
            print("   ➡️ Akurasi per Entitas:")
            for idx, col in enumerate(kolom_target):
                acc_col = accuracy_score(y_test[:, idx], y_pred[:, idx])
                print(f"      • {col:<15} : {acc_col*100:.2f}%")
            print("-"*60)
            
        except Exception as e:
            print(f"❌ Gagal memuat atau menguji {pkl_name}. Error: {e}")
            print("-"*60)

if __name__ == "__main__":
    check_pkl_accuracy()