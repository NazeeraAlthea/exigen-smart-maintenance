import os
import sys

# =====================================================================
# FIXING PYTHON PATH ROUTING (Tambahkan ini di baris paling atas!)
# =====================================================================
# Dapatkan lokasi absolut dari folder 'src' (mundur 2 tingkat dari file train.py)
CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_FILE_DIR, "../.."))

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)
# =====================================================================

# Sekarang impor absolut dari sudut pandang folder 'src' dijamin berjalan 100% aman!
from ml_models.ticketing.utils import super_clean_text 

import concurrent.futures
import pandas as pd
import numpy as np
import dagshub
import mlflow
import mlflow.sklearn
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

def proses_satu_baris(args):
    teks, label_asli = args
    severity_kelas = label_asli[5] # Index 5 adalah severity
    
    n_copies = 0
    if severity_kelas == 'Fatal': n_copies = 6
    elif severity_kelas == 'Berat': n_copies = 3
    elif severity_kelas in ['Ringan', 'Sedang']: n_copies = 3
        
    hasil_sementara = [(teks, label_asli)]
    for _ in range(n_copies):
        hasil_sementara.append((teks, label_asli))
    return hasil_sementara

def run_training():
    # 1. Dapatkan lokasi absolut dari folder tempat file train.py ini berada
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Tarik rute secara akurat ke file dataset (mundur 3 tingkat dari train.py)
    path_master = os.path.abspath(os.path.join(BASE_DIR, "../../../data/dataset_tiket_master_bersih.csv"))
    
    if not os.path.exists(path_master):
        print(f"❌ File dataset tidak ditemukan di: {path_master}")
        return

    print("⏳ Membaca dataset...")
    df_master = pd.read_csv(path_master, sep='|', on_bad_lines='skip')
    kolom_target = ['tipe_aset', 'lokasi_gedung', 'lokasi_lantai', 'lokasi_zona', 'kategori_aset', 'severity']
    df_master = df_master.dropna(subset=kolom_target + ['teks_keluhan_awam'])
    df_master = df_master[df_master['severity'].isin(['Ringan', 'Sedang', 'Berat', 'Fatal'])]

    X_raw = df_master['teks_keluhan_awam'].astype(str)
    Y = df_master[kolom_target]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, Y, test_size=0.2, random_state=42)

    # Paralel Augmentasi
    print("⏳ Memulai augmentasi data multi-threading...")
    antrean_tugas = [(teks, y_train.iloc[i].values) for i, teks in enumerate(X_train_raw)]
    X_train_aug_list, Y_train_aug_list = [], []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for hasil_baris in executor.map(proses_satu_baris, antrean_tugas):
            for res_teks, res_label in hasil_baris:
                X_train_aug_list.append(res_teks)
                Y_train_aug_list.append(res_label)

    X_train_aug = pd.Series(X_train_aug_list)
    y_train_aug = pd.DataFrame(Y_train_aug_list, columns=kolom_target)

    # Preprocessing memakai fungsi dari utils.py
    print("⏳ Membersihkan teks dengan RegEx Location Binding...")
    X_train_clean = X_train_aug.apply(super_clean_text)
    X_test_clean = X_test_raw.apply(super_clean_text)

    # MLflow tracking
    print("⏳ Menghubungkan ke DagsHub/MLflow...")
    dagshub.init(repo_owner='NazeeraAlthea', repo_name='exigen-smart-maintenance', mlflow=True)
    
    with mlflow.start_run(run_name="Eksperimen_Ultimate_Regex_Bigram"):
        print("🤖 Melatih Model Multi-Output Random Forest...")
        tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2, max_df=0.9)
        X_train_tfidf = tfidf.fit_transform(X_train_clean)
        
        clf = MultiOutputClassifier(RandomForestClassifier(n_estimators=300, class_weight='balanced', n_jobs=-1, random_state=42))
        
        # Fit model
        estimators_ = [RandomForestClassifier(n_estimators=300, class_weight='balanced', n_jobs=-1, random_state=42).fit(X_train_tfidf, y_train_aug.iloc[:, i]) for i in range(y_train_aug.shape[1])]
        clf.estimators_ = estimators_
        clf.classes_ = [est.classes_ for est in estimators_]
        
        pipeline = Pipeline([('tfidf', tfidf), ('clf', clf)])
        
        # Evaluasi
        print("\n" + "="*50)
        print("🎯 LAPORAN EVALUASI PER LABEL (ENTITAS)")
        print("="*50)
        
        y_pred = pipeline.predict(X_test_clean)
        
        # 1. Menghitung Exact Match Ratio
        exact_match = np.all(y_pred == y_test.values, axis=1).mean()
        mlflow.log_metric("exact_match_ratio", exact_match)
        print(f"✅ Exact Match Ratio (Benar 6 Entitas Sekaligus): {exact_match:.4f}\n")
        
        # 2. Menghitung Evaluasi Rinci Tiap Label
        for i, col in enumerate(kolom_target):
            acc_col = accuracy_score(y_test[col], y_pred[:, i])
            print(f"🔹 Evaluasi Target: {col.upper()} (Akurasi: {acc_col:.4f})")
            print(classification_report(y_test[col], y_pred[:, i], zero_division=0))
            print("-" * 50)
            
            # Log akurasi individual per kolom ke MLflow
            mlflow.log_metric(f"accuracy_{col}", acc_col)
        
        # 3. Tarik rute secara akurat ke folder models untuk penyimpanan hasil joblib
        model_save_dir = os.path.abspath(os.path.join(BASE_DIR, "../../../models/ticketing"))
        os.makedirs(model_save_dir, exist_ok=True)
        
        path_save_model = os.path.join(model_save_dir, "ticket_v1.1.1_tfidf.pkl")
        
        import joblib
        joblib.dump(pipeline, path_save_model)
        # =====================================================================
        # UPLOAD & REGISTRASI MODEL KE DAGSHUB MLFLOW
        # =====================================================================
        print("☁️ Mengunggah model ke DagsHub MLflow Model Registry...")
        mlflow.sklearn.log_model(
            sk_model=pipeline, 
            artifact_path="model_tfidf_rf",
            registered_model_name="Exigen_Smart_Ticketing_Model" # <--- Ini yang membuatnya masuk ke tab 'Models' DagsHub
        )
        # =====================================================================
        
        print(f"\n✅ Training selesai! Model disimpan di lokal ({path_save_model}) DAN di DagsHub Registry!")

if __name__ == "__main__":
    run_training()