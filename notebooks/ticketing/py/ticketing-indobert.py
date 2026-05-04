import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score

import mlflow
import mlflow.sklearn
import dagshub

# ==========================================
# 1. IMPORT DATASET
# ==========================================
file_path = "../../data/synthetic_report_dataset.csv"

with open(file_path, 'r', encoding='utf-8') as f:
    baris_pertama = f.readline()
    print("Isi baris pertama file mentah:\n👉", baris_pertama)

pemisah = '|' if '|' in baris_pertama else ','
print(f"Sistem mendeteksi pemisah kolom: '{pemisah}'\n")

nama_kolom = ['teks_keluhan_awam', 'teks_laporan_teknisi', 'kategori_aset', 'severity', 'root_cause', 'tindakan']
df = pd.read_csv(file_path, sep=pemisah, names=nama_kolom, on_bad_lines='skip')

print(f"Baris sebelum dibersihkan: {df.shape[0]}")

df = df.dropna(subset=['teks_keluhan_awam', 'teks_laporan_teknisi'], how='all')
df = df[df['teks_keluhan_awam'].astype(str).str.contains('teks_keluhan', case=False) == False]
df = df.reset_index(drop=True)

df['teks_keluhan_awam'] = df['teks_keluhan_awam'].astype(str)
df['teks_laporan_teknisi'] = df['teks_laporan_teknisi'].astype(str)

df['input_teks'] = df['teks_keluhan_awam'] + " " + df['teks_laporan_teknisi']

kolom_target = ['kategori_aset', 'severity', 'root_cause', 'tindakan']
Y = df[kolom_target]

print(f"Total data SIAP DITRAINING: {df.shape[0]}")
print(f"Bentuk Input (X): 1 Kolom Teks Gabungan")
print(f"Bentuk Target (Y): {Y.shape[1]} Kolom Kunci Jawaban")
print("\n--- Intip Isi Target (Y) ---")
print(Y.head(3))

# ==========================================
# 2. LOAD MODEL INDOBERT
# ==========================================
print("\nMemuat otak Deep Learning IndoBERT... (Tunggu sebentar)")
model_name = "indobenchmark/indobert-base-p1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = AutoModel.from_pretrained(model_name)

# ==========================================
# 3. FEATURE EXTRACTION
# ==========================================
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

print("Sedang mengekstrak makna kalimat ke dalam 768 dimensi... (Ini mungkin memakan waktu 1-2 menit)")
X_embeddings = np.vstack(df['input_teks'].apply(get_bert_embedding).values)
y = df[kolom_target]

# ==========================================
# 4. TRAIN-TEST SPLIT
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X_embeddings, y, test_size=0.2, random_state=42)

# ==========================================
# 5. MODELING & TUNING
# ==========================================
dagshub.init(repo_owner='NazeeraAlthea', repo_name='exigen-smart-maintenance', mlflow=True)
mlflow.set_experiment("Smart_Ticketing_Baseline_IndoBERT")

with mlflow.start_run(run_name="IndoBERT_RF_Hyperparameter_Tuning"):
    print("\nMulai melatih model dan mencari parameter terbaik...")
    
    rf_base = RandomForestClassifier(random_state=42)
    multi_target_rf = MultiOutputClassifier(rf_base, n_jobs=1)

    param_grid = {
        'estimator__n_estimators': [100, 200],
        'estimator__max_depth': [None, 10]
    }

    grid_search = GridSearchCV(multi_target_rf, param_grid, cv=3, n_jobs=1, verbose=1)
    grid_search.fit(X_train, y_train.values)

    print(f"\nBest Parameters: {grid_search.best_params_}")
    
    mlflow.log_params(grid_search.best_params_)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    print("\n=== HASIL EVALUASI FINAL (INDOBERT + RF) ===")
    
    exact_match_manual = (y_test.values == y_pred).all(axis=1).mean()
    mlflow.log_metric("exact_match_ratio", exact_match_manual)
    print(f"Exact Match Ratio (Benar Semua 4 Kolom): {exact_match_manual * 100:.2f}%")

    print("\n--- Akurasi Individu per Target ---")
    for i, col in enumerate(kolom_target):
        acc = accuracy_score(y_test.iloc[:, i], y_pred[:, i])
        mlflow.log_metric(f"accuracy_{col}", acc)
        print(f"Akurasi {col:15}: {acc * 100:.2f}%")
        
    mlflow.sklearn.log_model(best_model, "best_indobert_rf_model")
    print("\n✅ Run MLflow Selesai! Model dan metrik berhasil dikirim ke DagsHub.")