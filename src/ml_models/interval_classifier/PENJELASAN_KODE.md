# 📖 Penjelasan Kode: Interval Classifier (`train.py`)

## Alur Keseluruhan (Big Picture)

```
Data Mentah (3 file Excel)
    ↓
Gabungkan & Bersihkan
    ↓
Tentukan Label Target (Frekuensi) per Aset
    ↓
Encode fitur → angka
    ↓
Train XGBoost
    ↓
Simpan model (.pkl)
```

---

## Step-by-step

### Step 1: Load 3 File Data

```python
master_df    → Data semua aset (ID, Kategori, Tipe, Kekritisan, Tanggal Instalasi...)
komplain_df  → Riwayat kerusakan/komplain tiap aset (Severity, Jenis Kerusakan...)
frekuensi_df → Tabel referensi: tipe aset X bisa dijadwalkan Harian/Mingguan/Bulanan/dll
```

Kenapa 3 file? Karena masing-masing punya informasi berbeda yang kita butuhkan.

---

### Step 2: Agregasi Komplain per Aset

Dari ratusan baris komplain → 1 baris ringkasan per aset:

```
ID Aset | Jumlah_Komplain | Rata_Severity
A001    | 8               | 3.5  (sering rusak, parah)
A002    | 1               | 1.0  (jarang rusak, ringan)
```

**Kenapa?** Model butuh angka ringkas, bukan ratusan baris mentah. "Aset ini sudah rusak 8 kali dengan rata-rata keparahan 3.5" lebih informatif.

---

### Step 3: Gabung Master + Komplain

```python
aset_lengkap = merge(master_df, komplain_agregat)
```

Sekarang tiap aset punya info lengkap: karakteristiknya (Kategori, Tipe) + riwayat kerusakannya (Jumlah_Komplain, Rata_Severity).

---

### Step 4: Hitung Umur Aset

```python
Umur_Aset_Tahun = (hari_ini - Tanggal_Instalasi) / 365
```

**Kenapa?** Aset yang sudah 8 tahun tentu butuh perawatan lebih sering dari yang baru 1 tahun.

---

### Step 5: Bangun Lookup Frekuensi

```python
# Dari tabel frekuensi, buat dictionary:
('Mechanical', 'Tata Udara', 'AC Split') → ['Harian', 'Bulanan', 'Semester']
('Arsitektur', 'Interior', 'Plafon')     → ['Reaktif', 'Harian', 'Bulanan', 'Tahunan']
```

**Ini BUKAN pivot**. Ini adalah operasi `groupby → list aggregation`, yaitu mengelompokkan data 
dan mengumpulkan semua nilai frekuensi yang tersedia ke dalam satu list per kelompok.

Perbedaan dengan pivot:

| Operasi | Apa yang terjadi | Bentuk hasil |
|---|---|---|
| **Pivot** | Baris jadi kolom. Misalnya kolom "Harian", "Bulanan", "Semester" masing-masing jadi kolom baru dengan nilai di dalamnya | Tabel lebar (wide) |
| **GroupBy + List** (yang kita pakai) | Baris dikumpulkan jadi satu list per kelompok. Hasilnya dictionary, bukan tabel | Dictionary / lookup |

Visualisasi:

```
DATA AWAL (frekuensi_df):
| Kategori   | Sub Kategori | Tipe     | Frekuensi |
|------------|-------------|----------|-----------|
| Mechanical | Tata Udara  | AC Split | Harian    |
| Mechanical | Tata Udara  | AC Split | Bulanan   |
| Mechanical | Tata Udara  | AC Split | Semester  |

KALAU PIVOT → jadi tabel lebar:
| Kategori   | Sub Kategori | Tipe     | Harian | Bulanan | Semester |
|------------|-------------|----------|--------|---------|----------|
| Mechanical | Tata Udara  | AC Split | ✓      | ✓       | ✓        |

YANG KITA PAKAI (GroupBy + List) → jadi dictionary:
{('Mechanical','Tata Udara','AC Split'): ['Harian','Bulanan','Semester']}
```

Kita pakai dictionary karena lebih praktis untuk di-lookup saat fungsi `assign_frekuensi()` 
perlu cek: "Aset tipe ini bisa dijadwalkan apa saja?"

---

### Step 6: Assign Label Cerdas — `assign_frekuensi()` ← Inti Kecerdasan

Fungsi ini menyelesaikan masalah accuracy 17%. Logikanya:

```
Hitung skor urgensi (0.0 = baik, 1.0 = buruk):
├── 30% dari Jumlah Komplain     (sering rusak → urgen)
├── 30% dari Rata Severity        (rusak parah → urgen)
├── 20% dari Tingkat Kekritisan   (Critical > Major > Minor)
└── 20% dari Umur Aset            (tua → urgen)

Lalu pilih frekuensi dari daftar yang tersedia:
├── Skor urgensi TINGGI → pilih frekuensi KETAT    (Harian/Mingguan)
└── Skor urgensi RENDAH → pilih frekuensi LONGGAR  (Semester/Tahunan)
```

**Contoh konkrit:**

| Aset   | Komplain | Severity | Kekritisan | Umur | Skor   | Pilihan dari [Harian, Bulanan, Semester] |
|--------|----------|----------|------------|------|--------|------------------------------------------|
| AC-001 | 8x       | 3.5      | Critical   | 7th  | **0.81** | → **Harian** (kondisi buruk)           |
| AC-002 | 0x       | 0.0      | Minor      | 1th  | **0.06** | → **Semester** (kondisi baik)          |

**Kenapa begini?** Sebelumnya 1 aset punya banyak label identik → model bingung. Sekarang 1 aset = 1 label, dan labelnya masuk akal berdasarkan kondisi nyata.

---

### Step 7: Feature Encoding

```python
'Mechanical' → 2, 'Electrical' → 1, 'Arsitektur' → 0  (dst)
```

XGBoost hanya bisa baca angka, bukan teks. `LabelEncoder` mengubah setiap kategori teks jadi angka unik.

---

### Step 8: Train & Simpan

- Split data 80% training, 20% testing (dengan `stratify` agar proporsi kelas seimbang)
- Train XGBoost dengan hyperparameter:
  - `n_estimators=200` → jumlah pohon keputusan
  - `max_depth=7` → kedalaman pohon (lebih dalam = lebih kompleks)
  - `subsample=0.8` → tiap pohon pakai 80% data (mencegah overfitting)
  - `colsample_bytree=0.8` → tiap pohon pakai 80% fitur
- Simpan model + semua encoder ke `models/interval_classifier.pkl`

---

## Perbandingan: Sebelum vs Sesudah

| Aspek | Versi Lama (17%) | Versi Baru |
|---|---|---|
| Label per aset | 1 aset muncul 3-4x dengan label beda | 1 aset = 1 label |
| Logika label | Random / semua di-keep | Ditentukan oleh kondisi aset |
| Fitur umur aset | Tidak dipakai | Dipakai |
| Hyperparameter | `max_depth=5`, `n_estimators=100` | `max_depth=7`, `n_estimators=200`, + regularisasi |

---

## Fitur yang Digunakan Model

| Fitur | Tipe | Penjelasan |
|---|---|---|
| `Kategori` | Kategorikal | Grup besar aset (Mechanical, Electrical, dll) |
| `Sub Kategori` | Kategorikal | Sub-grup (Tata Udara, Plumbing, dll) |
| `Tipe` | Kategorikal | Jenis spesifik (AC Split, Pompa Air, dll) |
| `Tingkat Kekritisan` | Kategorikal | Critical / Major / Minor |
| `Jumlah_Komplain` | Numerik | Berapa kali aset ini pernah dikomplain |
| `Rata_Severity` | Numerik | Rata-rata keparahan komplain (1-4) |
| `Umur_Aset_Tahun` | Numerik | Usia aset sejak instalasi dalam tahun |
