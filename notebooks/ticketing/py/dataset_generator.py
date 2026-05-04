import os
import pandas as pd
import io
from google import genai

# ==========================================
# 1. SETUP GEMINI API
# ==========================================
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
client = genai.Client()

# ==========================================
# 2. DEFINE PROMPT
# ==========================================
print("Meminta Gemini untuk membuat dataset sintetik... (Mohon tunggu sekitar 10-20 detik)")

prompt = """
Kamu adalah asisten pembuat dataset Machine Learning.
Buatkan saya dataset sintetik sebanyak 500 baris. 

Skenario: Laporan kerusakan aset pabrik yang berisi cerita sebab-akibat antara keluhan orang awam dan laporan perbaikan oleh teknisi.

ATURAN KETAT:
- Pisahkan antar kolom HANYA menggunakan simbol pipe (|). JANGAN gunakan koma sebagai pemisah kolom.
- Jangan berikan teks pembuka atau penutup apapun. LANGSUNG berikan data mentahnya.

KOLOM YANG WAJIB ADA (Tepat 6 kolom):
teks_keluhan_awam|teks_laporan_teknisi|kategori_aset|severity|root_cause|tindakan

Pilihan untuk 4 kolom terakhir:
- kategori_aset: (HVAC, Pompa Air, Kelistrikan, Mesin Produksi)
- severity: (Rendah, Sedang, Tinggi)
- root_cause: (Tersumbat, Keausan, Overheat, Usia Pakai, Konsleting)
- tindakan: (Pembersihan, Penggantian Part, Pelumasan, Kalibrasi, Perbaikan Jaringan)

Contoh Baris yang Benar:
AC di ruang meeting lantai 2 netes air terus nih|udah disemprot selang pembuangannya karena mampet lumut, sekalian tambah freon|HVAC|Sedang|Tersumbat|Pembersihan
"""

# ==========================================
# 3. GENERATE CONTENT
# ==========================================
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=prompt,
)

# ==========================================
# 4. PROCESS & SAVE DATA
# ==========================================
csv_raw_data = response.text

# Membersihkan format markdown (seperti ```csv atau ```text) jika ada
if csv_raw_data.startswith("```"):
    # Membuang baris pertama dan baris terakhir
    lines = csv_raw_data.strip().split('\n')
    csv_raw_data = '\n'.join(lines[1:-1])

try:
    # Mengubah teks dari Gemini menjadi DataFrame Pandas
    df_sintetik = pd.read_csv(io.StringIO(csv_raw_data.strip()), sep='|')
    
    # Menyimpan dataset ke file CSV lokal
    file_name = "../../data/synthetic_report_dataset.csv"
    
    # CEK: Apakah file sudah ada sebelumnya?
    if os.path.exists(file_name):
        # Jika sudah ada, mode='a' (append) dan header=False agar nama kolom tidak ikut masuk ke tengah data
        df_sintetik.to_csv(file_name, mode='a', header=False, index=False)
        print(f"\n✅ BERHASIL! Data baru ditambahkan (Append) ke '{file_name}'")
        
        # Mengecek total baris keseluruhan sekarang
        df_total = pd.read_csv(file_name)
        print(f"Total baris keseluruhan saat ini: {df_total.shape[0]}")
    else:
        # Jika file belum ada, buat file baru (overwrite biasa)
        df_sintetik.to_csv(file_name, index=False)
        print(f"\n✅ BERHASIL! Dataset awal berhasil dibuat dan disimpan sebagai '{file_name}'")
        print(f"Jumlah baris: {df_sintetik.shape[0]}")
    
    print("\n--- 5 Baris Data Terbaru ---")
    print(df_sintetik.head()) # Menggunakan print, bukan display
    
except Exception as e:
    print(f"❌ Gagal memproses format data. Error: {e}")
    print("Teks mentah dari Gemini:\n", csv_raw_data)