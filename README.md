# 🚀 Exigen: Intelligent Asset Management & Predictive Maintenance

![Python Version](https://img.shields.io/badge/Python-3.11.9-blue.svg?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active_Development-success.svg)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF.svg?logo=github-actions&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Exigen** adalah platform analitik pemeliharaan cerdas (*Smart Maintenance*) end-to-end yang dirancang untuk mengubah data log historis teknisi dan sistem tiket menjadi wawasan prediktif yang dapat ditindaklanjuti. 

Bergeser dari paradigma *Reactive Maintenance* (menunggu aset rusak) ke *Predictive & Prescriptive Maintenance*, Exigen memanfaatkan kapabilitas algoritma **Machine Learning** dan **Natural Language Processing (NLP)** untuk menekan *downtime* sebelum kegagalan fatal terjadi pada mesin industri.

---

## 🎯 Key Features & Capabilities

* **Smart Auto-Ticketing:** Membedah teks keluhan bebas dari *user* menjadi tiket terstruktur (Kategori, Severity, Root Cause, Action) menggunakan NLP (*IndoBERT & TF-IDF*).
* **Failure Prediction:** Memprediksi probabilitas sisa umur pakai (*Remaining Useful Life*) mesin berdasarkan riwayat *event-based*.

---

## 🧠 Core AI Architectures (The 2 Pillars)

Sistem ini diorkestrasikan oleh 2 model AI independen yang terintegrasi secara *tightly-coupled* untuk menghasilkan ekosistem pemeliharaan yang komprehensif:

1. **Maintenance Predictor** ⏱️ 
   Memprediksi jarak waktu (dalam hitungan hari) menuju kerusakan atau kebutuhan servis berikutnya menggunakan algoritma berbasis *Regression* yang tangguh terhadap *outlier*.
2. **NLP Smart Ticketing** 🎫 *(Menggantikan konsep lama Report Generator)*
   Model pemrosesan bahasa alami yang mengekstraksi informasi dari teks keluhan awam dan log teknisi (*unstructured text*) lalu mengklasifikasikannya secara otomatis menjadi 4 parameter target perbaikan secara simultan.

---

## 💻 Tech Stack Specification

* **Data Handling:** `Pandas`, `NumPy`
* **Traditional ML:** `Scikit-Learn` (Random Forest)
* **Deep Learning & NLP:** `PyTorch`, `Transformers` (HuggingFace IndoBERT), `Sastrawi` (Indonesian Stemming & Stopwords)

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
git clone [https://github.com/NazeeraAlthea/exigen-smart-maintenance.git](https://github.com/NazeeraAlthea/exigen-smart-maintenance.git)
cd exigen-smart-maintenance
```
Buat virtual environment dan instal dependensi (Disarankan menggunakan Python 3.11+):

```
python -m venv venv
source venv/bin/activate  # Untuk Linux/Mac
venv\\Scripts\\activate     # Untuk Windows

pip install -r requirements.txt
```

👥 The Exigen Team (Group 2)
Platform ini dirancang dan dikembangkan sebagai bagian dari Project Based Learning (Mitra Industri: NTG).

Muhammad Arya Maulana
Melvin Okniel Sinaga
Jose Febryan Limbor
Najma Gusti Ayu Mahesa

© 2026 Exigen Smart Maintenance. All rights reserved.