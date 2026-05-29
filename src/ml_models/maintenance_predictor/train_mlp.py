import os
import re
import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
import joblib
import mlflow
import warnings
warnings.filterwarnings('ignore')

def parse_frekuensi(val):
    if pd.isna(val): return 30
    val_str = str(val).lower()
    num = re.findall(r'\d+', val_str)
    n = int(num[0]) if num else 1
    if 'minggu' in val_str: return n * 7
    if 'bulan' in val_str: return n * 30
    if 'tahun' in val_str: return n * 365
    if 'hari' in val_str: return n
    return 30

def main():
    print("=== DATA LOADING ===")
    path_komplain = 'data/aset_komplain_enriched.xlsx'
    path_master = 'data/master_aset_enriched.xlsx'
    path_ganti = 'data/riwayat_penggantian_aset.xlsx'
    path_frek = 'data/rencana_kegiatan_frekuensi_enriched.xlsx'
    
    df_komplain = pd.read_excel(path_komplain)
    df_master = pd.read_excel(path_master)
    df_ganti = pd.read_excel(path_ganti)
    df_frek = pd.read_excel(path_frek)
    
    df_komplain = df_komplain.rename(columns={'ID Aset': 'ID', 'Nama Aset': 'Nama'})
    df_ganti = df_ganti.rename(columns={'ID Aset Lama': 'ID', 'Nama Aset Lama': 'Nama'})
    
    for col in ['Tanggal Perencanaan', 'Tanggal Pengerjaan', 'Tanggal Selesai']:
        df_komplain[col] = pd.to_datetime(df_komplain[col], format='%d-%m-%Y', errors='coerce')
    df_master['Tanggal Instalasi'] = pd.to_datetime(df_master['Tanggal Instalasi'], format='%d-%m-%Y', errors='coerce')
    df_ganti['Tanggal Penggantian'] = pd.to_datetime(df_ganti['Tanggal Penggantian'], format='%d-%m-%Y', errors='coerce')
    df_komplain = df_komplain.dropna(subset=['Tanggal Pengerjaan'])
    
    print("=== FILTER PENGGANTIAN VALID ===")
    administrative_keywords = ['upgrade', 'standar', 'kontrak', 'spare']
    pattern = '|'.join(administrative_keywords)
    mask_valid = ~df_ganti['Alasan Penggantian'].str.contains(pattern, case=False, na=False)
    df_ganti_valid = df_ganti[mask_valid].copy()
    
    df_ganti_valid = df_ganti_valid.dropna(subset=['Tanggal Penggantian']).sort_values('Tanggal Penggantian').drop_duplicates(subset=['ID'])
    
    print("=== FILTER KOMPLAIN HANTU ===")
    df_komplain_valid = pd.merge(df_komplain, df_ganti_valid[['ID', 'Tanggal Penggantian']], on='ID', how='inner')
    df_komplain_valid = df_komplain_valid[df_komplain_valid['Tanggal Pengerjaan'] <= df_komplain_valid['Tanggal Penggantian']].copy()
    
    sev_map = {'Ringan': 1, 'Sedang': 2, 'Berat': 3, 'Fatal': 4, 'Rendah': 1, 'Tinggi': 3, 'Kritis': 4, 'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
    df_komplain_valid['Severity_Num'] = df_komplain_valid['Severity'].map(sev_map).fillna(1)
    
    df_komplain_valid = df_komplain_valid.sort_values(by=['ID', 'Tanggal Pengerjaan'])
    df_komplain_valid['Hari_Antar_Komplain'] = df_komplain_valid.groupby('ID')['Tanggal Pengerjaan'].diff().dt.days
    
    first_complaint = df_komplain_valid.groupby('ID')['Tanggal Pengerjaan'].min().reset_index()
    first_complaint.columns = ['ID', 'Tanggal_Komplain_Pertama']
    last_complaint = df_komplain_valid.groupby('ID')['Tanggal Pengerjaan'].max().reset_index()
    last_complaint.columns = ['ID', 'Tanggal_Komplain_Terakhir']
    
    agg_funcs = {
        'Tanggal Pengerjaan': 'count',
        'Severity_Num': ['mean', 'max'],
        'Biaya Perbaikan': ['sum', 'mean'],
        'Hari_Antar_Komplain': 'mean'
    }
    df_komplain_agg = df_komplain_valid.groupby('ID').agg(agg_funcs)
    df_komplain_agg.columns = ['Total_Komplain', 'Severity_Mean', 'Severity_Max', 'Biaya_Total', 'Biaya_Mean', 'Hari_Antar_Komplain_Mean']
    df_komplain_agg = df_komplain_agg.reset_index()
    
    df_komplain_agg = pd.merge(df_komplain_agg, first_complaint, on='ID', how='left')
    df_komplain_agg = pd.merge(df_komplain_agg, last_complaint, on='ID', how='left')
    
    print("=== LABELING DATA ===")
    df_labeled = df_ganti_valid[['ID', 'Tanggal Penggantian']].copy()
    df_labeled = pd.merge(df_labeled, df_master[['ID', 'Tanggal Instalasi', 'Kategori', 'Sub Kategori', 'Tipe', 'Merek', 'Tingkat Kekritisan']], on='ID', how='inner')
    
    df_labeled['Umur_Aset_Total_Hari'] = (df_labeled['Tanggal Penggantian'] - df_labeled['Tanggal Instalasi']).dt.days
    df_labeled = df_labeled.dropna(subset=['Umur_Aset_Total_Hari'])
    df_labeled = df_labeled[df_labeled['Umur_Aset_Total_Hari'] > 0]
    
    # GABUNG HISTORI KOMPLAIN VALID
    df_labeled = pd.merge(df_labeled, df_komplain_agg, on='ID', how='left')
    
    # BUANG ASET YANG MATI MENDADAK TANPA KOMPLAIN
    df_labeled = df_labeled.dropna(subset=['Total_Komplain'])
    print(f"Total Aset Emas (Punya Komplain & Mati Natural) siap latih: {len(df_labeled)}")
    
    # Filter Outlier pada target RUL
    Q1 = df_labeled['Umur_Aset_Total_Hari'].quantile(0.25)
    Q3 = df_labeled['Umur_Aset_Total_Hari'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_labeled = df_labeled[(df_labeled['Umur_Aset_Total_Hari'] >= lower_bound) & (df_labeled['Umur_Aset_Total_Hari'] <= upper_bound)]
    print(f"Total Aset setelah filter outlier: {len(df_labeled)}")
    
    df_frek_unik = df_frek.drop_duplicates(subset=['Kategori', 'Sub Kategori', 'Tipe'])
    df_labeled = pd.merge(df_labeled, df_frek_unik[['Kategori', 'Sub Kategori', 'Tipe', 'Frekuensi']], on=['Kategori', 'Sub Kategori', 'Tipe'], how='left')
    
    # --- SMART FEATURE ENGINEERING ---
    df_labeled['Frekuensi_Hari'] = df_labeled['Frekuensi'].apply(parse_frekuensi)
    
    df_labeled['Durasi_Aktif_Komplain'] = (df_labeled['Tanggal_Komplain_Terakhir'] - df_labeled['Tanggal_Komplain_Pertama']).dt.days
    df_labeled['Durasi_Aktif_Komplain'] = df_labeled['Durasi_Aktif_Komplain'].replace(0, 1) # Mencegah pembagian dengan 0
    df_labeled['Complaint_Velocity'] = df_labeled['Total_Komplain'].fillna(0) / df_labeled['Durasi_Aktif_Komplain']
    df_labeled['Complaint_Velocity'] = df_labeled['Complaint_Velocity'].fillna(0)
    
    mean_cost_tipe = df_labeled.groupby('Tipe')['Biaya_Total'].transform('mean')
    mean_cost_tipe = mean_cost_tipe.replace(0, 1) # Mencegah pembagian dengan 0
    df_labeled['Cost_Deviation_Ratio'] = df_labeled['Biaya_Total'].fillna(0) / mean_cost_tipe
    
    df_labeled['Umur_Saat_Komplain_Terakhir'] = (df_labeled['Tanggal_Komplain_Terakhir'] - df_labeled['Tanggal Instalasi']).dt.days
    df_labeled['Umur_Saat_Komplain_Terakhir'] = df_labeled['Umur_Saat_Komplain_Terakhir'].fillna(0)
    
    impute_zero_cols = ['Total_Komplain', 'Severity_Mean', 'Severity_Max', 'Biaya_Total', 'Biaya_Mean']
    df_labeled[impute_zero_cols] = df_labeled[impute_zero_cols].fillna(0)
    
    median_hari = df_labeled['Hari_Antar_Komplain_Mean'].median()
    df_labeled['Hari_Antar_Komplain_Mean'] = df_labeled['Hari_Antar_Komplain_Mean'].fillna(median_hari if not pd.isna(median_hari) else 0)
    
    df_labeled['Biaya_Total_Log'] = np.log1p(df_labeled['Biaya_Total'])
    df_labeled['Biaya_Mean_Log'] = np.log1p(df_labeled['Biaya_Mean'])
    
    fitur_x = ['Total_Komplain', 'Severity_Mean', 'Severity_Max', 'Biaya_Total_Log', 'Biaya_Mean_Log', 'Hari_Antar_Komplain_Mean',
               'Kategori', 'Sub Kategori', 'Tipe', 'Merek', 'Tingkat Kekritisan', 'Umur_Saat_Komplain_Terakhir',
               'Frekuensi_Hari', 'Complaint_Velocity', 'Cost_Deviation_Ratio']
    
    print("\n=== TRAINING MLP REGRESSOR RUL PURE (DENGAN HYPERPARAMETER TUNING) ===")
    df_labeled = df_labeled.sort_values(by='Tanggal Penggantian').reset_index(drop=True)
    split_idx = int(len(df_labeled) * 0.8)
    df_train = df_labeled.iloc[:split_idx].copy()
    df_test = df_labeled.iloc[split_idx:].copy()
    
    kat_cols = ['Kategori', 'Sub Kategori', 'Tipe', 'Merek', 'Tingkat Kekritisan']
    
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Maintenance_Predictor_RUL_MLP")
    
    with mlflow.start_run():
        encoder = ce.TargetEncoder(cols=kat_cols, smoothing=10)
        df_train[kat_cols] = encoder.fit_transform(df_train[kat_cols], df_train['Umur_Aset_Total_Hari'])
        df_test[kat_cols] = encoder.transform(df_test[kat_cols])
        
        X_train = df_train[fitur_x]
        y_train = df_train['Umur_Aset_Total_Hari']
        X_test = df_test[fitur_x]
        y_test = df_test['Umur_Aset_Total_Hari']
        
        # Neural Networks are sensitive to scale, so we use StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        base_mlp = MLPRegressor(random_state=42, early_stopping=True, max_iter=500)
        
        param_dist = {
            'hidden_layer_sizes': [(64, 32), (128, 64), (128, 64, 32), (64, 64)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam'],
            'alpha': [0.0001, 0.001, 0.01, 0.1],
            'learning_rate_init': [0.001, 0.005, 0.01]
        }
        
        print("Memulai proses Auto-Tuning RUL dengan MLP Regressor... Mohon tunggu.")
        random_search = RandomizedSearchCV(
            estimator=base_mlp,
            param_distributions=param_dist,
            n_iter=20,           
            scoring='neg_mean_absolute_error',
            cv=5,                
            verbose=1,
            random_state=42,
            n_jobs=1
        )
        
        random_search.fit(X_train_scaled, y_train)
        
        best_mlp_model = random_search.best_estimator_
        print("\n[Parameter Terbaik Ditemukan]:")
        print(random_search.best_params_)
        
        y_pred_hari = best_mlp_model.predict(X_test_scaled)
        y_test_hari = y_test.values
        
        mae = mean_absolute_error(y_test_hari, y_pred_hari)
        rmse = np.sqrt(mean_squared_error(y_test_hari, y_pred_hari))
        mape = mean_absolute_percentage_error(y_test_hari, y_pred_hari)
        r2 = r2_score(y_test_hari, y_pred_hari)
        
        print("\n=== HASIL EVALUASI REAL DATA (MLP RUL) ===")
        print(f"R-Squared (R2) : {r2:.4f}")
        print(f"MAE            : {mae:.2f} Hari")
        print(f"RMSE           : {rmse:.2f} Hari")
        print(f"MAPE           : {mape * 100:.2f}%")
        
        mlflow.log_params(random_search.best_params_)
        mlflow.log_params({"model": "MLPRegressor_Pure_RUL", "target_filtering": True, "outlier_removal": "IQR"})
        mlflow.log_metrics({"r2": r2, "mae": mae, "rmse": rmse, "mape": mape})
        
        os.makedirs('models', exist_ok=True)
        # Save pipeline with scaler included
        model_pipeline = {'encoder': encoder, 'scaler': scaler, 'model': best_mlp_model, 'features': fitur_x}
        joblib.dump(model_pipeline, 'models/maintenance_predictor_rul_mlp.pkl')
        print("\nModel pipeline MLP RUL berhasil di-save ke: models/maintenance_predictor_rul_mlp.pkl")

if __name__ == '__main__':
    main()
