# 🚀 Exigen: AI-Powered Predictive Maintenance System

![Python Version](https://img.shields.io/badge/python-3.11.9-blue.svg)
![Status](https://img.shields.io/badge/status-development-orange.svg)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF.svg)
![MLflow](https://img.shields.io/badge/Tracking-MLflow-0194E2.svg)

**Exigen Predictive Maintenance** adalah platform analitik cerdas yang dirancang untuk memantau kesehatan aset mesin secara *real-time*. Dengan memanfaatkan algoritma *Machine Learning* dan *Deep Learning*, sistem ini mengotomatiskan prediksi kerusakan, klasifikasi urgensi pemeliharaan, dan estimasi biaya perbaikan sebelum kegagalan fatal terjadi.



---

## 🧠 Core AI Models
Sistem ini diorkestrasikan oleh 4 model AI independen yang bekerja secara simultan untuk menghasilkan laporan analitik yang komprehensif:

1. **Maintenance Predictor:** Memprediksi probabilitas dan sisa umur pakai (*Remaining Useful Life*).
2. **Cost Estimator:** Menghitung estimasi biaya perbaikan secara dinamis berdasarkan tingkat keparahan anomali yang terdeteksi.
3. **Interval Classifier:** Mengkategorikan urgensi tindakan pemeliharaan (harian, mingguan, bulanan, triwulan, semester, tahunan).
4. **Report Generator:** Menyusun narasi laporan kesehatan aset secara otomatis berdasarkan hasil prediksi ketiga model di atas.

---

## 🏗️ Arsitektur & Struktur Direktori

Proyek ini memisahkan fase riset, pelacakan eksperimen, dan kode produksi dengan sangat ketat agar siap untuk skala *Enterprise*.

```text
EXIGEN-PREDICTIVE-MAINTENANCE/
├── .github/workflows/          # CI/CD Pipelines (Code Testing & Auto-Retrain)
├── data/                       # Dataset historis & data sensor (.xlsx)
├── mlruns/                     # Database lokal MLflow untuk tracking eksperimen
├── models/                     # Artefak model yang siap dideploy (.h5, .json, .pkl)
├── notebooks/                  # Jupyter Notebooks untuk EDA & Feature Engineering
├── src/
│   ├── ml_models/              # Modul spesifik untuk tiap arsitektur AI
│   │   ├── cost_estimator/
│   │   ├── interval_classifier/
│   │   ├── maintenance_predictor/
│   │   └── report_generator/
│   ├── monitoring/             # Script untuk memantau performa model dan data drift
│   ├── web/                    # Endpoint API untuk komunikasi dengan Frontend
│   └── predict.py              # Orchestrator pemanggil model (Inference Engine)
├── README.md                   # Dokumentasi Utama
└── requirements.txt            # Dependensi Python