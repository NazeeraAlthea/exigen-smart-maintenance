import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda():
    print("Memuat dataset...")
    # 1. Load Data
    master_df = pd.read_excel('data/master_aset_enriched.xlsx')
    komplain_df = pd.read_excel('data/aset_komplain_enriched.xlsx')
    frekuensi_df = pd.read_excel('data/rencana_kegiatan_frekuensi_enriched.xlsx')

    # 2. Agregasi data komplain
    severity_map = {'Rendah': 1, 'Sedang': 2, 'Tinggi': 3, 'Kritis': 4, 'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
    komplain_df['Severity_Num'] = komplain_df['Severity'].map(severity_map).fillna(2)
    
    komplain_agregat = komplain_df.groupby('ID Aset').agg(
        Jumlah_Komplain=pd.NamedAgg(column='ID Aset', aggfunc='count'),
        Rata_Severity=pd.NamedAgg(column='Severity_Num', aggfunc='mean')
    ).reset_index()
    
    # 3. Gabungkan Data
    aset_lengkap = pd.merge(master_df, komplain_agregat, left_on='ID', right_on='ID Aset', how='left')
    aset_lengkap['Jumlah_Komplain'] = aset_lengkap['Jumlah_Komplain'].fillna(0)
    aset_lengkap['Rata_Severity'] = aset_lengkap['Rata_Severity'].fillna(0)
    
    # 4. Gabung dengan Frekuensi
    final_df = pd.merge(aset_lengkap, frekuensi_df, on=['Kategori', 'Sub Kategori', 'Tipe'], how='inner')
    final_df = final_df.dropna(subset=['Frekuensi'])

    # Setup output directory
    out_dir = os.path.join(os.getcwd(), 'eda_outputs')
    os.makedirs(out_dir, exist_ok=True)

    # Basic Info
    print("=== INFO DATASET ===")
    print(final_df.info())
    print("\n=== STATISTIK DESKRIPTIF ===")
    print(final_df[['Jumlah_Komplain', 'Rata_Severity']].describe())

    # Set style
    sns.set(style="whitegrid")

    # 1. Plot Distribusi Target (Frekuensi)
    plt.figure(figsize=(10, 6))
    sns.countplot(data=final_df, x='Frekuensi', order=final_df['Frekuensi'].value_counts().index)
    plt.title('Distribusi Kelas Target (Frekuensi)')
    plt.xlabel('Frekuensi Pemeliharaan')
    plt.ylabel('Jumlah Aset')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'distribusi_frekuensi.png'))
    plt.close()

    # 2. Distribusi Jumlah Komplain vs Frekuensi
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=final_df, x='Frekuensi', y='Jumlah_Komplain', order=final_df['Frekuensi'].value_counts().index)
    plt.title('Distribusi Jumlah Komplain berdasarkan Frekuensi')
    plt.xlabel('Frekuensi Pemeliharaan')
    plt.ylabel('Jumlah Komplain')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'komplain_vs_frekuensi.png'))
    plt.close()

    # 3. Distribusi Rata-Rata Severity vs Frekuensi
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=final_df, x='Frekuensi', y='Rata_Severity', order=final_df['Frekuensi'].value_counts().index)
    plt.title('Distribusi Rata-Rata Severity berdasarkan Frekuensi')
    plt.xlabel('Frekuensi Pemeliharaan')
    plt.ylabel('Rata-Rata Severity')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'severity_vs_frekuensi.png'))
    plt.close()

    # 4. Heatmap Tingkat Kekritisan vs Frekuensi (Crosstab)
    plt.figure(figsize=(10, 6))
    crosstab_krit = pd.crosstab(final_df['Tingkat Kekritisan'], final_df['Frekuensi'])
    sns.heatmap(crosstab_krit, annot=True, fmt='d', cmap='Blues')
    plt.title('Hubungan Tingkat Kekritisan dengan Frekuensi')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'kekritisan_vs_frekuensi.png'))
    plt.close()

    print(f"\nEDA Selesai. Plot disimpan di: {out_dir}")

if __name__ == '__main__':
    run_eda()
