import re
import os
import pandas as pd

# =====================================================================
# 1. INISIALISASI KAMUS SLANG (HYBRID: INTERNET + KUSTOM ASET GENERATED)
# =====================================================================
SLANG_DICT = {}

# Jalur A: Ambil dari Kamus Alay Internet Terbuka
URL_SLANG = "https://raw.githubusercontent.com/nasalsabila/kamus-alay/master/colloquial-indonesian-lexicon.csv"
try:
    df_slang = pd.read_csv(URL_SLANG)
    SLANG_DICT = dict(zip(df_slang['slang'].str.lower(), df_slang['formal'].str.lower()))
except Exception:
    pass

# Jalur B: Ambil dari Kamus Slang Hasil Automated Generation (*.csv)
# Pastikan path ini mengarah ke lokasi file hasil export .ipynb kamu tadi
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
PATH_KAMUS_ASET = os.path.abspath(os.path.join(CURRENT_DIR, "../../../data/kamus_slang_aset.csv")) # Sesuaikan struktur folder proyek

if os.path.exists(PATH_KAMUS_ASET):
    try:
        df_kamus_aset = pd.read_csv(PATH_KAMUS_ASET)
        kamus_aset_kustom = dict(zip(df_kamus_aset['slang'].str.lower(), df_kamus_aset['formal'].str.lower()))
        # Timpa/gabungkan ke kamus utama. Jika ada crash, kamus aset kustom yang menang
        SLANG_DICT.update(kamus_aset_kustom)
        print(f"[OK] Sukses memuat {len(kamus_aset_kustom)} aturan slang aset dari {PATH_KAMUS_ASET}")
    except Exception as e:
        print(f"[WARN] Gagal membaca file kamus_slang_aset.csv lokal: {e}")
else:
    print(f"[WARN] File {PATH_KAMUS_ASET} tidak ditemukan. Deteksi imbuhan otomatis menggunakan fallback internet saja.")


# =====================================================================
# 2. ENGINE PREPROCESSING PRODUKSI (FAST & LIGHTWEIGHT)
# =====================================================================
def super_clean_text(text: str) -> str:
    """Melakukan standardisasi teks keluhan, pembersihan stuttering,
    pengikatan lokasi (diutamakan), dan normalisasi seluruh imbuhan aset instan via kamus.
    """
    if not text or not isinstance(text, str):
        return ""
        
    text = text.lower()
    
    # Langkah 1: Hapus kata berulang berurutan (Typo Stuttering)
    text = re.sub(r'\b(\w+)(?:\s+\1\b)+', r'\1', text)
    
    # Langkah 2: Bersihkan tanda strip nempel (Contoh: ac-nya -> acnya)
    text = text.replace("-", "")
    
    # LANGKAH 3: LEXICON LOCATION BINDING (DIPINDAH KE ATAS!)
    # Amankan lokasi sebelum singkatan nama gedung (A, B, C, D) diubah oleh kamus slang alay
    text = re.sub(r'\b(gedung|gdng|gd\.?|tower|twr|blok|blk|lobby|lobi|area)\s*([a-z0-9]+)\b', r'gedung_\2', text)
    text = re.sub(r'\b(lantai|lt\.?|level)\s*([a-z0-9]+)\b', r'lantai_\2', text)
    text = re.sub(r'\b(ruang|rg\.?|kamar|kmr)\s*([a-z0-9]+)\b', r'ruang_\2', text)
    
    # Langkah 4: Normalisasi Bahasa Gaul & Seluruh Imbuhan Aset Instan via O(1) Dictionary
    # Kata 'gedung_d' akan dihitung 1 kata utuh (\w+), sehingga huruf 'd' selamat dari terjemahan kamus.
    kata_kata = re.findall(r'\b\w+\b', text)
    kata_normal = [SLANG_DICT.get(kata, kata) for kata in kata_kata]
    text = " ".join(kata_normal)
    
    # Langkah 5: Buang karakter spesial tersisa, sisakan alphanumeric dan underscore (_)
    text = re.sub(r'[^a-z0-9_]', ' ', text)
    
    # Langkah 6: Normalisasi spasi ganda hasil pembersihan regex
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text