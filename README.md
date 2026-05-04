md_content = """# 🚀 Exigen: Intelligent Asset Management & Predictive Maintenance

![Python Version](https://img.shields.io/badge/Python-3.11.9-blue.svg?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active_Development-success.svg)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF.svg?logo=github-actions&logoColor=white)
![MLOps](https://img.shields.io/badge/MLOps-MLflow_%7C_DagsHub-0194E2.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Exigen** adalah platform analitik pemeliharaan cerdas (*Smart Maintenance*) end-to-end yang dirancang untuk mengubah data log historis teknisi dan sistem tiket menjadi wawasan prediktif yang dapat ditindaklanjuti. 

Bergeser dari paradigma *Reactive Maintenance* (menunggu aset rusak) ke *Predictive & Prescriptive Maintenance*, Exigen memanfaatkan kapabilitas algoritma **Machine Learning** dan **Natural Language Processing (NLP)** untuk menekan *downtime*, mengoptimalkan jadwal pemeliharaan, serta mengestimasi anggaran perbaikan sebelum kegagalan fatal terjadi pada mesin industri.

---

## 🎯 Key Features & Capabilities

* **Smart Auto-Ticketing:** Membedah teks keluhan bebas dari *user* menjadi tiket terstruktur (Kategori, Severity, Root Cause, Action) menggunakan NLP (*IndoBERT & TF-IDF*).
* **Failure Prediction:** Memprediksi probabilitas sisa umur pakai (*Remaining Useful Life*) mesin berdasarkan riwayat *event-based*.
* **Dynamic Budgeting:** Mengalkulasi estimasi pengeluaran perbaikan secara otomatis untuk kuartal berikutnya.
* **Automated MLOps Pipeline:** Dilengkapi dengan CI/CD yang memungkinkan *auto-retraining* model setiap kali ada data historis baru yang masuk.

---

## 🧠 Core AI Architectures (The 4 Pillars)

Sistem ini diorkestrasikan oleh 4 model AI independen yang terintegrasi secara *tightly-coupled* untuk menghasilkan ekosistem pemeliharaan yang komprehensif:

1. **Maintenance Predictor** ⏱️ 
   Memprediksi jarak waktu (dalam hitungan hari) menuju kerusakan atau kebutuhan servis berikutnya menggunakan algoritma berbasis *Regression* yang tangguh terhadap *outlier*.
2. **Interval Classifier** 📅
   Menganalisis frekuensi kerusakan untuk mengkategorikan ulang urgensi jadwal pemeliharaan (Harian, Mingguan, Bulanan, Triwulan) menggunakan optimasi penanganan data *imbalanced* (SMOTENC).
3. **Cost Estimator** 💰
   Menghitung estimasi biaya perbaikan aset secara dinamis berdasarkan tingkat keparahan (*severity*) anomali yang terdeteksi.
4. **NLP Smart Ticketing** 🎫 *(Menggantikan konsep lama Report Generator)*
   Model pemrosesan bahasa alami yang mengekstraksi informasi dari teks keluhan awam dan log teknisi (*unstructured text*) lalu mengklasifikasikannya secara otomatis menjadi 4 parameter target perbaikan secara simultan.

---

## 💻 Tech Stack Specification

* **Data Handling:** `Pandas`, `NumPy`
* **Traditional ML:** `Scikit-Learn` (Random Forest, XGBoost, MultiOutputClassifier)
* **Deep Learning & NLP:** `PyTorch`, `Transformers` (HuggingFace IndoBERT), `Sastrawi` (Indonesian Stemming & Stopwords)
* **MLOps & Tracking:** `MLflow`, `DagsHub`

---

## 🏗️ Project Architecture & Repository Structure

Proyek ini disusun dengan prinsip modularitas untuk memisahkan fase riset (EDA), pelacakan eksperimen, dan kode produksi agar siap untuk skala *Enterprise*.

```
EXIGEN-PREDICTIVE-MAINTENANCE/
├── .github/workflows/          # CI/CD Pipelines (Code Testing & Auto-Retrain)
├── data/                       # Dataset mentah, historis teknisi, & data sintetik
├── mlruns/                     # Database lokal MLflow untuk tracking metrik eksperimen
├── models/                     # Artefak model yang siap di-deploy (.pkl, .h5)
├── notebooks/                  # Jupyter Notebooks untuk Riset, EDA & Feature Engineering
├── src/                        # Source code utama untuk production
│   ├── ml_models/              # Modul spesifik untuk tiap pilar AI
│   │   ├── cost_estimator/
│   │   ├── interval_classifier/
│   │   ├── maintenance_predictor/
│   │   └── nlp_ticketing/      # Pipeline preprocessing teks & klasifikasi tiket (Auto-Logger)
│   ├── monitoring/             # Script pemantauan performa model & data drift
│   ├── web/                    # Endpoint API komunikasi dengan Frontend
│   └── predict.py              # Orchestrator & Inference Engine gabungan
├── requirements.txt            # Dependensi library Python
└── README.md                   # Dokumentasi Utama
```

🚀 Getting Started
1. Installation
Clone repositori ini dan masuk ke dalam direktori proyek:

```
git clone [https://github.com/UsernameAnda/exigen-smart-maintenance.git](https://github.com/UsernameAnda/exigen-smart-maintenance.git)
cd exigen-smart-maintenance
```
Buat virtual environment dan instal dependensi (Disarankan menggunakan Python 3.11+):

```
python -m venv venv
source venv/bin/activate  # Untuk Linux/Mac
venv\\Scripts\\activate     # Untuk Windows

pip install -r requirements.txt
```
2. Running MLflow Tracking (Local)
Untuk memantau komparasi metrik dari eksperimen NLP dan model lainnya:

```
mlflow ui --backend-store-uri file:./mlruns
```
Buka http://localhost:5000 di browser Anda.

👥 The Exigen Team (Group 2)
Platform ini dirancang dan dikembangkan sebagai bagian dari Project Based Learning (Mitra Industri: NTG).

Muhammad Arya Maulana - Scrum Master, Web Dev, & NLP Model Engineer

Melvin Okniel Sinaga - Predictive Maintenance Engineer

Jose Febryan Limbor - Interval Classifier Engineer

Najma Gusti Ayu Mahesa - Cost Estimator Engineer

© 2026 Exigen Smart Maintenance. All rights reserved.
"""