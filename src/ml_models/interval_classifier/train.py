import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Urutan frekuensi dari yang PALING KETAT ke PALING LONGGAR
FREKUENSI_ORDER = {
    'Harian': 1,
    'Mingguan': 2,
    'Bulanan': 3,
    'Tiga Bulanan': 4,
    'Semester': 5,
    'Tahunan': 6,
    'Reaktif': 7
}

def assign_frekuensi(row, frekuensi_lookup):
    """
    Menentukan frekuensi maintenance berdasarkan kondisi aset.
    
    Logika:
    - Lihat daftar frekuensi yang tersedia untuk tipe aset ini
    - Jika aset kondisinya buruk (banyak komplain, severity tinggi, kekritisan tinggi, umur tua)
      -> pilih frekuensi yang LEBIH KETAT (interval pendek)
    - Jika kondisinya baik -> pilih frekuensi LEBIH LONGGAR (interval panjang)
    """
    key = (row['Kategori'], row['Sub Kategori'], row['Tipe'])
    available = frekuensi_lookup.get(key, [])
    
    if not available:
        return None
    
    if len(available) == 1:
        return available[0]
    
    # Urutkan dari ketat ke longgar
    available_sorted = sorted(available, key=lambda x: FREKUENSI_ORDER.get(x, 99))
    
    # Hitung skor urgensi (0.0 = baik, 1.0 = sangat buruk)
    urgency = 0.0
    
    # Faktor 1: Jumlah komplain (normalisasi 0-1, cap di 10 komplain)
    urgency += min(row['Jumlah_Komplain'] / 10.0, 1.0) * 0.30
    
    # Faktor 2: Rata-rata severity (normalisasi 0-1, skala 1-4)
    urgency += (row['Rata_Severity'] / 4.0) * 0.30
    
    # Faktor 3: Tingkat kekritisan aset
    kritis_map = {'Critical': 1.0, 'Major': 0.6, 'Minor': 0.2}
    urgency += kritis_map.get(row['Tingkat Kekritisan'], 0.3) * 0.20
    
    # Faktor 4: Umur aset (normalisasi 0-1, cap di 10 tahun)
    urgency += min(row['Umur_Aset_Tahun'] / 10.0, 1.0) * 0.20
    
    # Konversi skor urgensi ke indeks frekuensi
    # urgency tinggi -> indeks rendah (frekuensi ketat)
    # urgency rendah -> indeks tinggi (frekuensi longgar)
    idx = int((1.0 - urgency) * (len(available_sorted) - 1))
    idx = max(0, min(idx, len(available_sorted) - 1))
    
    return available_sorted[idx]

def train_model():
    print("=" * 60)
    print("  INTERVAL CLASSIFIER - XGBoost Training Pipeline")
    print("=" * 60)
    
    print("\n📂 Memuat dataset...")
    # 1. Load Data
    master_df = pd.read_excel('data/master_aset_enriched.xlsx')
    komplain_df = pd.read_excel('data/aset_komplain_enriched.xlsx')
    frekuensi_df = pd.read_excel('data/rencana_kegiatan_frekuensi_enriched.xlsx')
    print(f"  Master Aset: {len(master_df)} baris")
    print(f"  Komplain: {len(komplain_df)} baris")
    print(f"  Frekuensi: {len(frekuensi_df)} baris")

    # 2. Agregasi data komplain per ID Aset
    print("\n🔧 Memproses fitur komplain...")
    severity_map = {'Rendah': 1, 'Sedang': 2, 'Tinggi': 3, 'Kritis': 4, 'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
    komplain_df['Severity_Num'] = komplain_df['Severity'].map(severity_map).fillna(2)
    
    komplain_agregat = komplain_df.groupby('ID Aset').agg(
        Jumlah_Komplain=pd.NamedAgg(column='ID Aset', aggfunc='count'),
        Rata_Severity=pd.NamedAgg(column='Severity_Num', aggfunc='mean')
    ).reset_index()
    print(f"  Aset dengan riwayat komplain: {len(komplain_agregat)}")
    
    # 3. Gabungkan Master Aset dengan Komplain
    print("\n🔗 Menggabungkan data...")
    aset_lengkap = pd.merge(master_df, komplain_agregat, left_on='ID', right_on='ID Aset', how='left')
    aset_lengkap['Jumlah_Komplain'] = aset_lengkap['Jumlah_Komplain'].fillna(0)
    aset_lengkap['Rata_Severity'] = aset_lengkap['Rata_Severity'].fillna(0)
    
    # 4. Hitung umur aset (fitur baru!)
    print("\n📅 Menghitung umur aset...")
    aset_lengkap['Tanggal_Instalasi_Parsed'] = pd.to_datetime(aset_lengkap['Tanggal Instalasi'], format='%d-%m-%Y', errors='coerce')
    today = pd.Timestamp.now()
    aset_lengkap['Umur_Aset_Tahun'] = ((today - aset_lengkap['Tanggal_Instalasi_Parsed']).dt.days / 365.25).fillna(0).clip(lower=0)
    
    # 5. Buat lookup frekuensi yang tersedia per tipe aset
    print("\n📋 Membangun lookup frekuensi...")
    frekuensi_dedup = frekuensi_df.drop_duplicates(subset=['Kategori', 'Sub Kategori', 'Tipe', 'Frekuensi'])
    frekuensi_lookup = frekuensi_dedup.groupby(['Kategori', 'Sub Kategori', 'Tipe'])['Frekuensi'].apply(list).to_dict()
    print(f"  {len(frekuensi_lookup)} tipe aset dengan jadwal maintenance")
    
    # 6. Assign frekuensi cerdas: 1 label per aset berdasarkan kondisi
    print("\n🧠 Menentukan label frekuensi berdasarkan kondisi aset...")
    aset_lengkap['Frekuensi'] = aset_lengkap.apply(lambda row: assign_frekuensi(row, frekuensi_lookup), axis=1)
    
    # Buang aset yang tidak punya mapping frekuensi
    final_df = aset_lengkap.dropna(subset=['Frekuensi'])
    print(f"  Aset dengan label valid: {len(final_df)} dari {len(aset_lengkap)}")
    print(f"\n  Distribusi label:")
    print(final_df['Frekuensi'].value_counts().to_string())
    
    if len(final_df) == 0:
        raise ValueError("Data gabungan kosong! Pastikan Kategori, Sub Kategori, dan Tipe memiliki kecocokan pada file frekuensi.")

    # 7. Siapkan fitur model (drop kolom yang bukan fitur)
    unused_cols = ['ID', 'ID Aset', 'Nama', 'Merek', 'Model', 'Tanggal Instalasi', 
                   'Tanggal_Instalasi_Parsed', 'Lokasi Gedung', 'Lokasi Lantai', 'Lokasi Zona', 'Status']
    df_model = final_df.drop(columns=[col for col in unused_cols if col in final_df.columns])
    
    # 8. Encoding Fitur Kategorikal
    print("\n🔢 Melakukan feature encoding...")
    cat_cols = ['Kategori', 'Sub Kategori', 'Tipe', 'Tingkat Kekritisan']
    encoders = {}
    
    for col in cat_cols:
        le = LabelEncoder()
        clean_col = df_model[col].fillna('Unknown').astype(str)
        df_model[col] = le.fit_transform(clean_col)
        encoders[col] = le
        
    # Label encoding target 'Frekuensi'
    target_le = LabelEncoder()
    df_model['Frekuensi'] = target_le.fit_transform(df_model['Frekuensi'].astype(str))
    encoders['Frekuensi'] = target_le
    
    # 9. Pemisahan Fitur dan Target
    X = df_model.drop(columns=['Frekuensi'])
    y = df_model['Frekuensi']
    
    print(f"\n  Fitur yang digunakan: {list(X.columns)}")
    print(f"  Jumlah data training: {len(X)}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 10. MLFlow Tracking & Model Training
    print("\n🚀 Melatih model menggunakan XGBoost Classifier...")
    mlflow.set_experiment("Interval_Classifier_XGBoost")
    with mlflow.start_run():
        xgb_params = {
            'n_estimators': 200,
            'max_depth': 7,
            'learning_rate': 0.1,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        mlflow.log_params(xgb_params)
        
        clf = XGBClassifier(**xgb_params)
        clf.fit(X_train, y_train)
        
        # Evaluasi
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(clf, "xgb_model")
        
        print(f"\n✅ Training selesai dengan Accuracy: {acc*100:.2f}%")
        print("\nLaporan Performa:")
        print(classification_report(y_test, y_pred, target_names=target_le.classes_))
        
        # Feature importance
        print("📊 Feature Importance:")
        for fname, fimp in sorted(zip(X.columns, clf.feature_importances_), key=lambda x: -x[1]):
            print(f"  {fname:25s} {fimp:.4f}")
        
        # 11. Simpan model
        os.makedirs('models', exist_ok=True)
        model_artifact = {
            'model': clf,
            'encoders': encoders,
            'features': list(X.columns),
            'frekuensi_order': FREKUENSI_ORDER
        }
        joblib.dump(model_artifact, 'models/interval_classifier.pkl')
        print("\n💾 Artefak model berhasil disimpan di: models/interval_classifier.pkl")

if __name__ == "__main__":
    train_model()
