# ULTIMATE MASTER CONTEXT: EXIGEN SMART MAINTENANCE SYSTEM

Pahami seluruh dokumen ini sebagai landasan dasar (*Single Source of Truth*) untuk memahami arsitektur data, rekayasa fitur, model pemelajaran mesin, strategi DevOps, alur integrasi tim, serta penyelesaian masalah teknis dalam proyek Exigen Smart Maintenance.

---

## 1. STRUKTUR PROYEK & STRATEGI AGILE MVP
* **Nama Produk:** Exigen Smart Maintenance (Intelligent Asset Management System).
* **Mitra Industri:** NTG (Project-Based Learning, Kelompok 2).
* **Tujuan Strategis:** Mengubah data catatan operasional (event-based logs) dan keluhan manual menjadi ekosistem pemeliharaan prediktif digital guna meminimalkan biaya perawatan serta mencegah downtime tidak terencana.
* **Strategi Kontrak Minimum Viable Product (MVP):** Mengikuti prinsip Agile Scrum Lite, ruang lingkup proyek secara ketat dipangkas dari 4 model fiktif/terpisah menjadi **2 PILAR MODEL INTI** yang terintegrasi secara end-to-end. Langkah ini diambil untuk memastikan model yang dikembangkan robust, komputasinya efisien, dan siap dilepas ke tahap produksi web.
* **Daftar Pengembang & Pembagian Tugas Terbaru (Kelompok 2):**
  - **Muhammad Arya Maulana (Anda):** Scrum Master & Lead NLP Engineer (Penanggung Jawab Pemodelan Teks/Suara Tiket Pintar).
  - **Jose Febryan Limbor:** Predictive Maintenance & Backend Engineer (Lead Pengembangan Model Tabular Estimasi Umur Sisa Mesin, Arsitektur Database relasional, serta Infrastruktur API Server).
  - **Najma Gusti Ayu Mahesa:** Frontend Developer (Perancang Antarmuka Web, Dashboard Analytics, & UI Formulir Laporan).
  - **Melvin Okniel Sinaga:** Frontend Developer (Perancang Antarmuka Web, Mobile-Responsive Form Teknisi, & Integrasi Alur UI).
  
---

## 2. PILAR UTAMA 1: MODEL PREDICTIVE MAINTENANCE (FOKUS MELVIN)
Pilar ini berfungsi sebagai jantung analitik sistem Exigen untuk memprediksi secara kuantitatif sisa waktu operasional aman suatu mesin sebelum mengalami kegagalan fungsi.

### A. Strategi Transformasi Data (Event-to-Event Prediction)
* **Keterbatasan Data Asli:** Dataset historis dari perusahaan mitra tidak memiliki log penggunaan harian berkelanjutan (*continuous time-series data* seperti sensor getaran atau temperatur per jam).
* **Solusi Rekayasa:** Mengubah paradigma pemrosesan data menjadi *Event-to-Event Prediction* berbasis kejadian (*event-based data*) dengan memanfaatkan tabel riwayat `df_perbaikan` dan `df_aset`. 
* **Target Variabel ($Y$):** `Remaining Useful Life` (RUL), yang didefinisikan sebagai jumlah jarak (dalam satuan hari) dari satu kejadian perbaikan/perawatan ke kejadian pemeliharaan kritis berikutnya.

### B. Feature Engineering & Representasi Data Tabular
Untuk memberikan kemampuan pemahaman runtun waktu pada model regresi tradisional, data ditransformasikan menjadi bentuk Tabular 2D menggunakan dua teknik:
1. **Sliding Window:** Pengelompokan kejadian perawatan berdasarkan rentang jendela waktu bergerak tertentu untuk menangkap tren degradasi mesin.
2. **Lag Features:** Pembuatan fitur jeda waktu historis, seperti `jeda_servis_sebelumnya`, `total_komplain_aset`, `biaya_perbaikan_rata_rata`, dan `severity_maksimal_tercatat`. Perhitungan matematika agregasi ini dieksekusi murni via skrip Python (`groupby()`, `mean()`, `max()`) untuk menjamin akurasi 100% dan menghindari halusinasi matematis dari model LLM generator.

### C. Arsitektur Model & Metrik Evaluasi
* **Kandidat Algoritma Pembanding:** Pemelajaran Mesin Tradisional Ensemble (`Random Forest Regressor` dan `XGBoost Regressor`) dikomparasikan dengan arsitektur Jaringan Saraf Tiruan Deep Learning (`MLP Regressor`).
* **Justifikasi Arsitektur:** Model ensemble tradisional (Random Forest/XGBoost) dipilih sebagai *core engine* utama karena performanya jauh lebih stabil, efisien secara komputasi, dan superior dalam mengenali pola set fitur lag tabular 2D dibandingkan model MLP.
* **Metrik Evaluasi Sukses:** Performa model dilacak secara ketat menggunakan kombinasi nilai galat minimum dari *Mean Absolute Error* (MAE) dan *Root Mean Squared Error* (RMSE), serta persentase kebaikan suai melalui skor *R-Squared* ($R^2$).

---

## 3. PILAR UTAMA 2: NLP SMART TICKETING + VOICE INPUT (FOKUS ARYA)
Pilar ini berfungsi sebagai gerbang masuk utama laporan insiden kerusakan (*helpdesk*), menyulap bahasa alami bebas (baik ketikan teks manual WhatsApp atau rekaman suara teknisi di lapangan) menjadi entitas terstruktur.

### A. Alur Pemrosesan Input Suara (Voice Input Pipeline)
* **Gerbang Depan:** Teknisi merekam suara keluhan kerusakan melalui antarmuka web.
* **Modul Speech-to-Text (STT):** File audio (`.wav` / `.mp3`) diproses di depan *pipeline* utama menggunakan **Whisper API (Groq)** atau **Web Speech API HTML5**.
* **Konfigurasi Anti-Lag:** Menggunakan varian model `base` dengan jenis komputasi `compute_type="int8"`. Konfigurasi ini hanya mengonsumsi memori RAM/VRAM sebesar ~140 MB, menjamin proses inferensi suara berjalan di bawah hitungan detik tanpa membebani server backend web saat demo aplikasi. Ditambahkan pula filter `vad_filter=True` untuk memangkas hening otomatis.
* **Metrik Validasi Ilmu:** Tingkat akurasi transkripsi diuji menggunakan penghitungan *Word Error Rate* (WER) dan *Character Error Rate* (CER) untuk membuktikan validitas ilmiah sistem di depan dosen penguji.

### B. Rekayasa Fitur Teks & Preprocessing Super
Teks hasil transkripsi suara atau ketikan teks mentah WhatsApp akan melalui rangkaian pembersihan data (*Data Preparation*) yang terstandarisasi di dalam memori eksekusi:
1. **Case Folding & RegEx Cleansing:** Mengubah teks menjadi huruf kecil serta menghapus seluruh angka dan tanda baca yang tidak relevan.
2. **Lexicon Location Binding (Trik RegEx Utama):** Untuk mempertahankan konteks spasial yang sering kali terpisah, skrip RegEx mendeteksi pola teks bangunan dan mengikatnya menjadi satu token leksikon khusus (contoh kalimat: "lantai 3" secara otomatis dilekatkan menjadi token tunggal `lantai_3`). Langkah ini krusial untuk mencegah algoritma pembobotan menghancurkan hubungan makna kata penunjuk lokasi.
3. **Kamus Slang & Sastrawi Pipeline:** Mengonversi kosakata non-baku teknisi ke bentuk baku melalui kamus *slang mapping*, dilanjutkan dengan proses pembuangan kata hubung (*Stopword Removal*) serta pemotongan kata imbuhan (*Stemming*) menggunakan library `Sastrawi`.
4. **Ekstraksi Fitur Matematika:** Teks bersih ditransformasikan menggunakan `TfidfVectorizer` dengan batasan ketat `max_features=3000` dan rentang n-gram `ngram_range=(1, 2)` (kombinasi kata tunggal dan Bigram) untuk menekan *sparse matrix noise*.
5. **Data Augmentation (Back-Translation):** Untuk mengatasi keterbatasan baris data latih asli, skrip generator memanggil teknik *Back-Translation* (Indonesia ➡️ Inggris ➡️ Indonesia) dengan bantuan modul `deep-translator` dan *Multithreading* (`ThreadPoolExecutor` dengan 10 *parallel workers*) untuk menduplikasi variasi struktur gramatikal secara *in-memory* hingga mengembang menjadi total **7.725 baris data siap latih**.

### C. Arsitektur Pemodelan Multi-Output Classification
* **Konfigurasi Model:** Menggunakan wrapper `MultiOutputClassifier` dari Scikit-Learn yang membungkus algoritma `RandomForestClassifier`.
* **Hyperparameter Setup:** Jumlah pohon diatur pada parameter `n_estimators=300` (atau diturunkan ke `100` untuk optimasi kecepatan iterasi lokal) dengan penyeimbang bobot otomatis `class_weight='balanced'`. Penyeimbangan bobot ini merupakan kunci utama untuk mendongkrak akurasi kelas minoritas pada entitas tingkat keparahan (*Severity*) hingga menembus angka lebih dari 86%.
* **Target Output Berjumlah 6 Entitas Kunci:** Model memprediksi 6 label target sekaligus dalam satu kali proses inferensi teks: `Tipe Aset`, `Lokasi Gedung`, `Lokasi Lantai`, `Lokasi Zona`, `Kategori Departemen`, dan `Tingkat Severity`.
* **Metrik Evaluasi Utama:** Tolok ukur keberhasilan model diukur menggunakan metrik ketat **Exact Match Ratio** (seluruh 6 target harus terprediksi benar secara simultan dalam satu tiket). Model saat ini sukses meraih skor *Exact Match Ratio* sebesar **65,53% - 70,65%** murni menggunakan algoritma pemelajaran mesin tradisional tanpa bantuan komputasi LLM generatif.

---

## 4. INTEGRASI TIM & MANAJEMEN FASE SIKLUS TIKET (END-TO-END)
Ekosistem Exigen Smart Maintenance membagi siklus penanganan aset ke dalam dua fase operasional terpisah demi menjaga integritas data tanpa mengorbankan kecepatan administrasi:

### A. FASE 1: PINTU MASUK LAPORAN (OTOMASI ADMINISTRASI - NLP ENGINE)
1. **Pemicu Peristiwa:** Staf gedung atau user awam mengirimkan aduan via aplikasi web atau WhatsApp Bot (Contoh: *"AC split di ruang HRD lantai 2 bocor parah netes air"*).
2. **Ekstraksi Cerdas:** Modul NLP Arya membedah teks/suara tersebut dalam hitungan milidetik, secara otomatis melahirkan tiket dengan status **"Open"** dan mengisi prediksi parameter awal secara cerdas:
   - `Tipe Aset` = AC Split, `Severity` (Awal) = Tinggi, `Lokasi Gedung` = Gedung Utama, `Lokasi Lantai` = Lantai 2, `Lokasi Zona` = Ruang HRD, `Kategori Departemen` = HVAC.
3. **Logika Kondisional Sistem:** Server backend secara otomatis memetakan draf tiket ini ke database teknisi yang sesuai bidangnya.
4. **Sinyal Tindakan Sistem:** Menggunakan Rule-Based Logic, sistem memicu notifikasi teks darurat instan (*WhatsApp API Alert*) ke HP teknisi pelaksana karena parameter `Severity` awal terdeteksi "Tinggi".

### B. FASE 2: PENYELESAIAN TIKET (VALIDASI FAKTA LAPANGAN & PREDICTIVE INPUT)
Ketika perbaikan selesai fisik, teknisi wajib menutup tiket melalui dashboard mobile mereka. Data masukan Fase 2 ini bersifat **Fakta Objektif Riil Lapangan** yang berfungsi membuang noise laporan awam agar tidak mengotori "bahan bakar" Machine Learning.

Berikut rincian kolom data Fase 2 dan mekanisme perolehannya di website:
1. **`ID Aset` (Angka):** Kunci utama relasional (*Primary Key*). Didapatkan melalui **Input Teknisi secara manual** via *dropdown select* aset yang terdaftar di area tersebut, atau didapat instan via **Scan Kode QR / Barcode** yang tertempel fisik pada unit mesin menggunakan kamera HP teknisi.
2. **`Nama Aset` (Teks):** **Ditarik Otomatis oleh Sistem** dari master tabel database `df_aset` (49k baris) sesaat setelah `ID Aset` terkunci.
3. **`Tanggal Perencanaan` (Tanggal):** **Dihasilkan Otomatis oleh Sistem** (*Auto-generated timestamp*) saat Admin/Sistem memvalidasi penugasan tiket di Fase 1.
4. **`Tanggal Pengerjaan` (Tanggal):** **Di-generate Otomatis oleh Sistem** saat teknisi menekan tombol *"Mulai Kerjakan"* pada antarmuka aplikasi.
5. **`Tanggal Selesai` (Tanggal):** **Dihasilkan Otomatis oleh Sistem** (*Auto-generated timestamp*) tepat ketika teknisi menekan tombol *"Selesai Diperbaiki / Close Ticket"*. Selisih hari antara tanggal ini dengan riwayat pengerjaan masa lalu akan dikurangkan di backend Python untuk memunculkan nilai **RUL (Sisa Hari Menuju Rusak)** untuk model Melvin.
6. **`Jenis Kerusakan` (Teks/Kategorikal):** Fakta kegagalan komponen (Contoh: *Kompresor Macet, Pipa Pecah, Freon Habis*). **Wajib Di-input Teknisi** lewat pilihan *dropdown* menu pasca-pembongkaran unit.
7. **`Severity` Riil (Teks/Kategorikal):** Tingkat keparahan akhir (Contoh: *Kritis, Sedang, Rendah*). **Di-input Teknisi** untuk memvalidasi kondisi lapangan asli (bertindak sebagai supervisor kebenaran tebakan NLP Fase 1).
8. **`Penyebab` / Root Cause (Teks/Kategorikal):** Akar masalah pemicu kerusakan (Contoh: *Usia Pakai/Wear and Tear, Tegangan Listrik Drop, Kelalaian Operasional*). **Wajib Di-input Teknisi** lewat pilihan menu *dropdown*.
9. **`Biaya Perbaikan` (Angka):** **Di-input oleh Admin** berdasarkan nota/faktur pembelian suku cadang riil, atau **di-input Teknisi** berdasarkan standar harga jasa logistik internal perusahaan. Data finansial ini langsung disuapkan ke **Model Cost Estimator Najma** sebagai data training baru.
10. **`Spare Part Digunakan` (Teks):** **Di-input Teknisi** sesuai komponen cadangan yang diambil resmi dari gudang. Menandakan umur subsistem terkait kembali segar ke angka 0 hari.
11. **`Teknisi Pelaksana` (Teks):** **Diambil Otomatis oleh Sistem** berdasarkan informasi sesi login akun teknisi (*Session Authentication Login*) yang sedang membuka aplikasi.

---

## 5. STRATEGI DEVOPS, DEPLOYMENT, & PENYELESAIAN ERROR
Untuk mempermudah kerja sama antar-developer, disepakati standarisasi operasional berfokus pada efisiensi komputasi:

### A. SOP Manajemen Repositori Git & .gitignore (.pkl Prevention)
* **Penyimpanan Kode (GitHub):** GitHub digunakan khusus untuk melacak riwayat baris skrip kode teks (`.py`, `.ipynb`, `.json`). File biner berukuran besar seperti model pemelajaran mesin (`.pkl` atau `.joblib`) **DILARANG KERAS** di-push ke GitHub karena memicu perlambatan repository.
* **File Aturan `.gitignore`:** Pengembang wajib memasang berkas `.gitignore` di root folder dengan konfigurasi:
  ```text
  models/*.pkl
  models/*.joblib
  __pycache__/
  .env
  venv/


* Penyimpanan Model Biner (DagsHub / Cloud): File model disimpan dan dilacak metriknya memanfaatkan repositori MLflow lokal yang tersinkronisasi ke DagsHub (NazeeraAlthea/exigen-smart-maintenance). Anggota tim mengunduh file biner model secara manual dari dashboard DagsHub dan meletakkannya di folder lokal models/ masing-masing.

### B. API Data Contract (JSON Payload Schema)

* Request Payload (POST /api/predict-ticket):
    ```
    {
    "id_laporan": "TKT-202605-001",
    "teks_keluhan": "AC split di ruang HRD lantai 2 bocor parah netes air",
    "sumber_input": "WhatsApp_Bot",
    "waktu_lapor": "2026-05-24T13:45:00Z"
    }
    ```

* Response Payload (Returned JSON Object):
    ```
    {
    "status_proses": "Sukses",
    "tiket_id": "TKT-202605-001",
    "hasil_prediksi_ai": {
        "tipe_aset": "AC Split",
        "lokasi_gedung": "Gedung Utama",
        "lokasi_lantai": "Lantai 2",
        "lokasi_zona": "Ruang HRD",
        "kategori_departemen": "HVAC",
        "tingkat_severity": "Tinggi"
    },
    "saran_tindakan_sistem": "Segera kirimkan push-notification jadwal darurat ke teknisi spesialis HVAC"
    }
    ```

## 6. INSTRUKSI OPERASIONAL UNTUK AGEN AI
Setiap kali Anda menerima perintah instruksi coding, debugging, atau penyusunan draf dokumen akademis (Bab 4/Bab 5) terkait proyek Exigen Smart Maintenance dari user, Anda WAJIB tunduk pada batasan arsitektur tradisional machine learning (TF-IDF + Random Forest untuk NLP, Event-to-Event Lag Features untuk Predictive Maintenance) yang tercantum di dalam berkas ini. Dilarang keras berasumsi atau menyisipkan pustaka deep learning runtun waktu (seperti LSTM/GRU) atau LLM generatif ke dalam solusi kode produksi kecuali diminta secara eksplisit.

Konteks master di atas kini telah sempurna, mencakup pembagian kerja dua fase siklus tiket (Fase 1 Laporan & Fase 2 Penutupan Tiket oleh Teknisi), lengkap beserta cara sistem memperoleh datanya di lapangan untuk menyuplai model regresi milik Melvin!
