import pandas as pd
import numpy as np

# Load Data
df_komplain = pd.read_excel('data/aset_komplain_enriched.xlsx')
df_master = pd.read_excel('data/master_aset_enriched.xlsx')
df_ganti = pd.read_excel('data/riwayat_penggantian_aset.xlsx')

# Parsing dates
df_komplain['Tanggal Pengerjaan'] = pd.to_datetime(df_komplain['Tanggal Pengerjaan'], format='%d-%m-%Y', errors='coerce')
df_master['Tanggal Instalasi'] = pd.to_datetime(df_master['Tanggal Instalasi'], format='%d-%m-%Y', errors='coerce')
df_ganti['Tanggal Penggantian'] = pd.to_datetime(df_ganti['Tanggal Penggantian'], format='%d-%m-%Y', errors='coerce')

df_komplain = df_komplain.rename(columns={'ID Aset': 'ID'})
df_master = df_master.rename(columns={'ID': 'ID'})

def process_and_export(df_g_source, output_path, filter_umur_nol=True):
    df_g = df_g_source.copy()
    df_g = df_g.rename(columns={'ID Aset Lama': 'ID'})
    
    # Merge master
    df_exp = pd.merge(df_g[['ID', 'Nama Aset Lama', 'Tanggal Penggantian', 'Alasan Penggantian']], 
                         df_master[['ID', 'Kategori', 'Tipe', 'Tanggal Instalasi']], 
                         on='ID', how='inner')
    
    # Hitung Umur
    df_exp['Umur_Hari'] = (df_exp['Tanggal Penggantian'] - df_exp['Tanggal Instalasi']).dt.days
    
    if filter_umur_nol:
        df_exp = df_exp[df_exp['Umur_Hari'] > 0]
        
    # Pisahkan komplain valid (sebelum ganti) dan hantu (setelah ganti)
    df_k_merged = pd.merge(df_komplain[['ID', 'Tanggal Pengerjaan']], df_exp[['ID', 'Tanggal Penggantian']], on='ID', how='inner')
    
    # Komplain Valid
    df_k_valid = df_k_merged[df_k_merged['Tanggal Pengerjaan'] <= df_k_merged['Tanggal Penggantian']]
    last_valid = df_k_valid.groupby('ID')['Tanggal Pengerjaan'].max().reset_index()
    last_valid = last_valid.rename(columns={'Tanggal Pengerjaan': 'Tanggal_Komplain_Terakhir'})
    
    total_komplain_valid = df_k_valid.groupby('ID')['Tanggal Pengerjaan'].count().reset_index()
    total_komplain_valid = total_komplain_valid.rename(columns={'Tanggal Pengerjaan': 'Total_Komplain'})
    
    # Komplain Hantu
    df_k_hantu = df_k_merged[df_k_merged['Tanggal Pengerjaan'] > df_k_merged['Tanggal Penggantian']]
    last_hantu = df_k_hantu.groupby('ID')['Tanggal Pengerjaan'].max().reset_index()
    last_hantu = last_hantu.rename(columns={'Tanggal Pengerjaan': 'Tanggal_Komplain_Hantu'})
    
    # Merge back to export
    df_exp = pd.merge(df_exp, last_valid, on='ID', how='left')
    df_exp = pd.merge(df_exp, total_komplain_valid, on='ID', how='left')
    df_exp = pd.merge(df_exp, last_hantu, on='ID', how='left')
    
    df_exp['Total_Komplain'] = df_exp['Total_Komplain'].fillna(0)
    
    # Calculate Jeda
    df_exp['Jeda_Komplain_Ke_Ganti (Hari)'] = (df_exp['Tanggal Penggantian'] - df_exp['Tanggal_Komplain_Terakhir']).dt.days
    
    # Status
    df_exp['Status_Kematian'] = np.where(df_exp['Tanggal_Komplain_Terakhir'].isna(), 'Mati Mendadak (Tanpa Komplain)', 'Mati Setelah Komplain')
    df_exp['Ada_Komplain_Hantu'] = np.where(df_exp['Tanggal_Komplain_Hantu'].notna(), 'Ya', 'Tidak')
    
    # Format Dates to string for Excel readability
    date_cols = ['Tanggal Penggantian', 'Tanggal Instalasi', 'Tanggal_Komplain_Terakhir', 'Tanggal_Komplain_Hantu']
    for col in date_cols:
        df_exp[col] = df_exp[col].dt.strftime('%d-%m-%Y')
        
    # Rearrange columns
    cols = ['ID', 'Nama Aset Lama', 'Kategori', 'Tipe', 'Tanggal Instalasi', 'Tanggal Penggantian', 'Umur_Hari', 
            'Alasan Penggantian', 'Status_Kematian', 'Total_Komplain', 'Tanggal_Komplain_Terakhir', 'Jeda_Komplain_Ke_Ganti (Hari)', 
            'Ada_Komplain_Hantu', 'Tanggal_Komplain_Hantu']
    df_exp = df_exp[cols]
    
    # Save to Excel
    df_exp.to_excel(output_path, index=False)
    print(f"Data berhasil diekspor ke: {output_path}")


# 1. EXPORT FILTERED (Hanya riwayat kerusakan alami)
pattern = '|'.join(['upgrade', 'standar', 'kontrak', 'spare'])
mask_valid = ~df_ganti['Alasan Penggantian'].str.contains(pattern, case=False, na=False)
df_ganti_valid = df_ganti[mask_valid].copy()
process_and_export(df_ganti_valid, 'data/investigasi_penggantian_aset.xlsx', filter_umur_nol=True)

# 2. EXPORT UNFILTERED (Semua data riwayat penggantian tanpa filter)
process_and_export(df_ganti, 'data/investigasi_penggantian_aset_unfiltered.xlsx', filter_umur_nol=False)
