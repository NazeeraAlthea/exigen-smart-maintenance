import joblib
import pandas as pd
import numpy as np

class IntervalClassifierPredictor:
    def __init__(self, model_path='models/interval_classifier.pkl'):
        """
        Inisialisasi predictor dengan memuat artifact model yang telah ditraining.
        """
        try:
            artifact = joblib.load(model_path)
            self.model = artifact['model']
            self.encoders = artifact['encoders']
            self.features = artifact['features']
        except FileNotFoundError:
            raise FileNotFoundError(f"Model artifact tidak ditemukan di {model_path}. Harap jalankan train.py terlebih dahulu.")
            
    def predict(self, asset_data: dict) -> str:
        """
        Memprediksi frekuensi/interval maintenance untuk sebuah aset.
        
        Parameter:
        asset_data (dict): Dictionary yang mengandung keys fitur.
                           Contoh: {'Kategori': 'Mechanical', 'Sub Kategori': 'Tata Udara', 
                                    'Tipe': 'AC Split', 'Tingkat Kekritisan': 'High',
                                    'Jumlah_Komplain': 2, 'Rata_Severity': 3.0}
        
        Return:
        str: Hasil observasi model, contoh "Bulanan"
        """
        # Konversi input 1 observasi menjadi DataFrame pandas
        df = pd.DataFrame([asset_data])
        
        # Validasi kolom/fitur minimal terpenuhi
        missing_cols = [col for col in self.features if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Input data kekurangan fitur-fitur wajib berikut: {missing_cols}")
        
        # Pastikan kolom terurut sesuai saat model ditraining
        df = df[self.features].copy()
        
        # Pengubahan fitur dari label teks menuju angka numerik yang dapat dibaca XGBoost
        cat_cols = ['Kategori', 'Sub Kategori', 'Tipe', 'Tingkat Kekritisan']
        for col in cat_cols:
            if col in self.encoders and col in df.columns:
                le = self.encoders[col]
                try:
                    df[col] = le.transform(df[col].astype(str))
                except ValueError as e:
                    # Nilai baru yang tidak dikenali saat training ditangani sementara dengan label 0
                    print(f"⚠️ Peringatan: Terdapat nilai tidak dikenali pada kolom {col}. Memakai default 0.")
                    df[col] = 0
                    
        # Prediksi
        pred_numeric = self.model.predict(df)[0]
        
        # Mengembalikan dari output algoritma XGBoost (numerik) ke string (misal: "Harian")
        target_le = self.encoders.get('Frekuensi')
        if target_le:
            pred_label = target_le.inverse_transform([pred_numeric])[0]
            return pred_label
        
        return str(pred_numeric)

# Script Testing sederhana jika di-run secara langsung
if __name__ == "__main__":
    try:
        predictor = IntervalClassifierPredictor()
        
        # Simulasi dummy aset yang sering rusak parah
        dummy_asset = {
            'Kategori': 'Mechanical',
            'Sub Kategori': 'Plumbing',
            'Tipe': 'Pompa Air',
            'Tingkat Kekritisan': 'Critical',
            'Jumlah_Komplain': 8,
            'Rata_Severity': 3.8
        }
        
        print("====== UJI COBA INFERENSI ======")
        print(f"Data Input Aset:\n{dummy_asset}\n")
        
        hasil = predictor.predict(dummy_asset)
        print(f"🚀 Hasil Prediksi Jadwal Maintenance: {hasil}")
        print("=================================")
        
    except Exception as e:
        print(f"Gagal melakukan inferensi: {str(e)}")
