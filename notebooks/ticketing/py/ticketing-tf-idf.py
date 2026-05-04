import pandas as pd
import numpy as np
import re
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Library Khusus Bahasa Indonesia
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# MLflow
import mlflow
import mlflow.sklearn
import dagshub

# ==========================================
# 1. IMPORT DATA
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
# 2. CLEANING & PREPROCESSING
# ==========================================
stemmer = StemmerFactory().create_stemmer()
stopword_remover = StopWordRemoverFactory().create_stop_word_remover()

def clean_preprocessing(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = stopword_remover.remove(text)
    text = stemmer.stem(text)
    return text

print("\nSedang melakukan Preprocessing teks...")
df['clean_teks'] = df['input_teks'].apply(clean_preprocessing)

# ==========================================
# 3. TRAIN-TEST SPLIT
# ==========================================
X = df['clean_teks']
y = df[kolom_target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 4. DATA AUGMENTATION
# ==========================================
def synonym_augmentation(text):
    synonyms = {"panas": "overheat", "rusak": "kendala", "bocor": "netes", "mati": "padam"}
    words = text.split()
    augmented_words = [synonyms.get(w, w) for w in words]
    return " ".join(augmented_words)

X_train_aug = X_train.apply(synonym_augmentation)
X_train_final = pd.concat([X_train, X_train_aug])
y_train_final = pd.concat([y_train, y_train])

print(f"\nData setelah Augmentasi: {X_train_final.shape[0]} baris.")

# ==========================================
# 5. FEATURE ENGINEERING & MODELING
# ==========================================
mlflow.set_tracking_uri("file:../../mlruns") 
mlflow.set_experiment("Smart_Ticketing_Baseline_TFIDF")

with mlflow.start_run(run_name="TFIDF_RF_Hyperparameter_Tuning"):
    print("Mulai Hyperparameter Tuning dengan GridSearchCV dan MLflow Tracking...")
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=42)))
    ])

    param_grid = {
        'tfidf__max_features': [1000, 2000],
        'clf__estimator__n_estimators': [100, 200],
        'clf__estimator__max_depth': [None, 10, 20]
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=1, verbose=1)
    grid_search.fit(X_train_final, y_train_final)

    print(f"\nBest Parameters: {grid_search.best_params_}")
    
    mlflow.log_params(grid_search.best_params_)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    print("\n=== HASIL EVALUASI FINAL ===")
    
    exact_match_manual = (y_test.values == y_pred).all(axis=1).mean()
    mlflow.log_metric("exact_match_ratio", exact_match_manual)
    print(f"Exact Match Ratio (Benar Semua 4 Kolom): {exact_match_manual * 100:.2f}%")

    print("\n--- Akurasi Individu per Target ---")
    for i, col in enumerate(kolom_target):
        acc = accuracy_score(y_test.iloc[:, i], y_pred[:, i])
        mlflow.log_metric(f"accuracy_{col}", acc)
        print(f"Akurasi {col:15}: {acc * 100:.2f}%")
        
    mlflow.sklearn.log_model(best_model, "best_rf_tfidf_model")
    print("\n✅ Run MLflow Selesai! Model dan metrik berhasil disimpan ke 'mlruns/'.")