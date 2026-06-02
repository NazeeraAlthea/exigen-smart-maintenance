import os
import re
import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
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
    
    print("=== LABELING DATA ===")
    df_labeled = df_ganti_valid[['ID', 'Tanggal Penggantian']].copy()
    df_labeled = pd.merge(df_labeled, df_master[['ID', 'Tanggal Instalasi', 'Kategori', 'Sub Kategori', 'Tipe', 'Merek', 'Tingkat Kekritisan']], on='ID', how='inner')
    
    df_labeled['Umur_Aset_Total_Hari'] = (df_labeled['Tanggal Penggantian'] - df_labeled['Tanggal Instalasi']).dt.days
    df_labeled = df_labeled.dropna(subset=['Umur_Aset_Total_Hari'])
    df_labeled = df_labeled[df_labeled['Umur_Aset_Total_Hari'] > 0]
    
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
    df_labeled['Frekuensi_Hari'] = df_labeled['Frekuensi'].apply(parse_frekuensi)
    
    rows = []
    print(f"Mengompilasi dataset event-sliced bebas leak untuk {len(df_labeled)} aset...")
    for idx, asset in df_labeled.iterrows():
        asset_id = asset['ID']
        t_install = asset['Tanggal Instalasi']
        t_replace = asset['Tanggal Penggantian']
        
        df_comp_asset = df_komplain_valid[df_komplain_valid['ID'] == asset_id]
        df_comp_asset = df_comp_asset.sort_values(by='Tanggal Pengerjaan')
        
        if len(df_comp_asset) == 0:
            continue
            
        df_comp_asset['Days_Since_Prev'] = df_comp_asset['Tanggal Pengerjaan'].diff().dt.days
        
        for i in range(1, len(df_comp_asset) + 1):
            sub_history = df_comp_asset.iloc[:i]
            current_complaint = df_comp_asset.iloc[i-1]
            t_current = current_complaint['Tanggal Pengerjaan']
            
            age = (t_current - t_install).days
            if age <= 0:
                age = 1
                
            total_lifespan = (t_replace - t_install).days
            
            total_komplain = i
            sev_mean = sub_history['Severity_Num'].mean()
            sev_max = sub_history['Severity_Num'].max()
            biaya_total = sub_history['Biaya Perbaikan'].sum()
            biaya_mean = sub_history['Biaya Perbaikan'].mean()
            
            if i == 1:
                hari_antar_komplain_mean = 0.0
            else:
                hari_antar_komplain_mean = sub_history['Days_Since_Prev'].dropna().mean()
                if pd.isna(hari_antar_komplain_mean):
                    hari_antar_komplain_mean = 0.0
                
            biaya_total_log = np.log1p(biaya_total)
            biaya_mean_log = np.log1p(biaya_mean)
            comp_velocity = total_komplain / age
            
            rows.append({
                'ID': asset_id,
                'Kategori': asset['Kategori'],
                'Sub Kategori': asset['Sub Kategori'],
                'Tipe': asset['Tipe'],
                'Merek': asset['Merek'],
                'Tingkat Kekritisan': asset['Tingkat Kekritisan'],
                'Frekuensi_Hari': asset['Frekuensi_Hari'],
                'Age_At_Complaint': age,
                'Total_Komplain': total_komplain,
                'Severity_Mean': sev_mean,
                'Severity_Max': sev_max,
                'Biaya_Total_Raw': biaya_total,
                'Biaya_Total_Log': biaya_total_log,
                'Biaya_Mean_Log': biaya_mean_log,
                'Hari_Antar_Komplain_Mean': hari_antar_komplain_mean,
                'Complaint_Velocity': comp_velocity,
                'Total_Lifespan': total_lifespan,
                'Tanggal Penggantian': t_replace
            })
            
    df_event_sliced = pd.DataFrame(rows)
    print(f"Total baris dataset siap latih: {len(df_event_sliced)}")
    
    print("\n=== TEMPORAL SPLIT BERBASIS ID (ANTI OVERLAP) ===")
    df_event_sliced = df_event_sliced.sort_values(by='Tanggal Penggantian').reset_index(drop=True)
    unique_ids = df_event_sliced['ID'].unique()
    split_idx = int(len(unique_ids) * 0.8)
    train_ids = set(unique_ids[:split_idx])
    test_ids = set(unique_ids[split_idx:])
    
    df_train = df_event_sliced[df_event_sliced['ID'].isin(train_ids)].copy()
    df_test = df_event_sliced[df_event_sliced['ID'].isin(test_ids)].copy()
    
    print(f"Train set: {len(df_train)} baris ({len(train_ids)} aset)")
    print(f"Test set: {len(df_test)} baris ({len(test_ids)} aset)")
    
    print("\n=== HITUNG MEAN BIAYA PER TIPE (ANTI LOOK-AHEAD BEBAS LEAK) ===")
    df_train_unique_assets = df_train.drop_duplicates(subset=['ID'])
    mean_cost_tipe = df_train_unique_assets.groupby('Tipe')['Biaya_Total_Raw'].mean().to_dict()
    
    global_mean_cost = df_train_unique_assets['Biaya_Total_Raw'].mean()
    if pd.isna(global_mean_cost) or global_mean_cost == 0:
        global_mean_cost = 1.0
        
    def apply_cost_dev_ratio(df, means, global_mean):
        tipe_means = df['Tipe'].map(means).fillna(global_mean)
        tipe_means = tipe_means.replace(0, global_mean)
        return df['Biaya_Total_Raw'] / tipe_means
        
    df_train['Cost_Deviation_Ratio'] = apply_cost_dev_ratio(df_train, mean_cost_tipe, global_mean_cost)
    df_test['Cost_Deviation_Ratio'] = apply_cost_dev_ratio(df_test, mean_cost_tipe, global_mean_cost)
    
    fitur_x = ['Total_Komplain', 'Severity_Mean', 'Severity_Max', 'Biaya_Total_Log', 'Biaya_Mean_Log', 'Hari_Antar_Komplain_Mean',
               'Kategori', 'Sub Kategori', 'Tipe', 'Merek', 'Tingkat Kekritisan',
               'Frekuensi_Hari', 'Complaint_Velocity', 'Cost_Deviation_Ratio', 'Age_At_Complaint']
    
    print("\n=== TRAINING MLP REGRESSOR RUL PURE (DENGAN HYPERPARAMETER TUNING) ===")
    kat_cols = ['Kategori', 'Sub Kategori', 'Tipe', 'Merek', 'Tingkat Kekritisan']
    
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Maintenance_Predictor_RUL_MLP_v2")
    
    with mlflow.start_run():
        X_train = df_train[fitur_x]
        y_train = df_train['Total_Lifespan']
        X_test = df_test[fitur_x]
        y_test = df_test['Total_Lifespan']
        
        # Neural Networks are sensitive to scale, so we include StandardScaler in the pipeline
        pipeline = Pipeline([
            ('encoder', ce.TargetEncoder(cols=kat_cols, smoothing=10)),
            ('scaler', StandardScaler()),
            ('model', MLPRegressor(random_state=42, early_stopping=True, max_iter=500))
        ])
        
        param_dist = {
            'model__hidden_layer_sizes': [(64, 32), (128, 64), (128, 64, 32), (64, 64)],
            'model__activation': ['relu', 'tanh'],
            'model__solver': ['adam'],
            'model__alpha': [0.0001, 0.001, 0.01, 0.1],
            'model__learning_rate_init': [0.001, 0.005, 0.01]
        }
        
        print("Memulai proses Auto-Tuning RUL dengan MLP Regressor... Mohon tunggu.")
        random_search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_dist,
            n_iter=20,           
            scoring='neg_mean_absolute_error',
            cv=5,                
            verbose=1,
            random_state=42,
            n_jobs=1
        )
        
        random_search.fit(X_train, y_train)
        
        best_pipeline = random_search.best_estimator_
        print("\n[Parameter Terbaik Ditemukan]:")
        print(random_search.best_params_)
        
        y_pred_lifespan = best_pipeline.predict(X_test)
        y_test_lifespan = y_test.values
        
        mae = mean_absolute_error(y_test_lifespan, y_pred_lifespan)
        rmse = np.sqrt(mean_squared_error(y_test_lifespan, y_pred_lifespan))
        mape = mean_absolute_percentage_error(y_test_lifespan, y_pred_lifespan)
        r2 = r2_score(y_test_lifespan, y_pred_lifespan)
        
        print("\n=== HASIL EVALUASI REAL DATA (MLP RUL) ===")
        print(f"R-Squared (R2) : {r2:.4f}")
        print(f"MAE            : {mae:.2f} Hari")
        print(f"RMSE           : {rmse:.2f} Hari")
        print(f"MAPE           : {mape * 100:.2f}%")
        
        mlflow.log_params(random_search.best_params_)
        mlflow.log_params({"model": "MLPRegressor_v2", "target_filtering": True, "outlier_removal": "IQR", "leak_free": True, "pipeline_cv": True})
        mlflow.log_metrics({"r2": r2, "mae": mae, "rmse": rmse, "mape": mape})
        
        os.makedirs('models', exist_ok=True)
        # Save pipeline directly
        model_pipeline = {
            'pipeline': best_pipeline, 
            'features': fitur_x,
            'mean_cost_tipe': mean_cost_tipe,
            'global_mean_cost': global_mean_cost
        }
        joblib.dump(model_pipeline, 'models/maintenance_predictor_v2_mlp.pkl')
        print("\nModel pipeline MLP RUL berhasil di-save ke: models/maintenance_predictor_v2_mlp.pkl")

if __name__ == '__main__':
    main()
