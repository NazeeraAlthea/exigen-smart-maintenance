import os
import time
import torch
import joblib
from faster_whisper import WhisperModel, download_model
from ml_models.ticketing.utils import super_clean_text

class ExigenInferenceEngine:
    def __init__(self):
        self.model_nlp_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../models/ticketing/ticket_v1.1.1_tfidf.pkl"))
        self.whisper_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../models/whisper-large-v3"))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. Load Model NLP Tiket Pintar
        if os.path.exists(self.model_nlp_path):
            self.pipeline_nlp = joblib.load(self.model_nlp_path)
        else:
            raise FileNotFoundError(f"⚠️ Model pkl tidak ditemukan di: {self.model_nlp_path}")
            
        # 2. Load Model Whisper Offline
        path_model_lokal = download_model("large-v3", output_dir=self.whisper_dir)
        self.model_stt = WhisperModel(path_model_lokal, device=self.device, compute_type="int8_float16")

    def transkripsi_audio(self, audio_path: str) -> str:
        if not os.path.exists(audio_path):
            return ""
        segments, _ = self.model_stt.transcribe(
            audio_path, language="id", beam_size=5,
            condition_on_previous_text=False, vad_filter=True
        )
        return " ".join([segment.text for segment in segments]).strip()

    def predict_ticket(self, input_data: str, mode: str = "text") -> dict:
        """Pintu masuk utama prediksi Fase 1 (Mendukung Teks Langsung / Path File Audio)"""
        t0 = time.time()
        
        # Jalur Routing Input
        if mode == "voice":
            teks_mentah = self.transkripsi_audio(input_data)
        else:
            teks_mentah = input_data

        if not teks_mentah or teks_mentah.strip() == "":
            return {"error": "Tidak ada teks/suara komplain yang valid terdeteksi."}

        # Bersihkan teks
        teks_bersih = super_clean_text(teks_mentah)
        
        # Inferensi AI Multi-Output
        prediction = self.pipeline_nlp.predict([teks_bersih])[0]
        
        # Ekstrak Probabilitas (Confidence) untuk Jalur AI
        confidences = [1.0] * 6
        if hasattr(self.pipeline_nlp, "predict_proba"):
            try:
                probas = self.pipeline_nlp.predict_proba([teks_bersih])
                if isinstance(probas, list):
                    confidences = [float(p.max(axis=1)[0]) for p in probas]
                else:
                    confidences = [float(probas.max(axis=1)[0])] * 6
            except Exception:
                pass

        # Format prediksi ke dalam dictionary
        predictions = {
            "tipe_aset": prediction[0],
            "lokasi_gedung": prediction[1],
            "lokasi_lantai": prediction[2],
            "lokasi_zona": prediction[3],
            "kategori_dept": prediction[4],
            "severity_awal": prediction[5]
        }
        
        # Deteksi field lokasi yang kosong/belum terisi
        missing_locations = []
        lokasi_fields = {
            "gedung": {"value": predictions["lokasi_gedung"], "conf": confidences[1]},
            "lantai": {"value": predictions["lokasi_lantai"], "conf": confidences[2]},
            "zona": {"value": predictions["lokasi_zona"], "conf": confidences[3]}
        }
        
        # Ambang batas AI (jika tidak disebut spesifik, namun AI yakin di atas 60%)
        THRESHOLD = 0.60 

        empty_values = ["", "unknown", "none", "-", "tidak diketahui", "tidak terdeteksi", "null"]
        teks_lower = teks_bersih.lower()
        # Buat versi teks tanpa underscore untuk mempermudah pencarian kata kunci murni
        teks_tanpa_underscore = teks_lower.replace("_", " ")
        
        for field, info in lokasi_fields.items():
            is_valid = True
            value = info["value"]
            conf = info["conf"]
            val_str = str(value)
            val_lower = val_str.lower().replace("_", " ") # Normalisasi nilai prediksi juga

            if not value or val_lower in empty_values:
                is_valid = False
            else:
                # PENDEKATAN HYBRID YANG DIPERBAIKI:
                # Cek langsung ke teks yang sudah dinormalisasi (tanpa spasi/underscore yang mengganggu)
                kata_kunci = val_lower.replace(field, "").strip()
                
                keyword_match = (
                    val_lower in teks_tanpa_underscore or 
                    kata_kunci in teks_tanpa_underscore.split() or
                    val_lower.replace(" ", "_") in teks_lower
                )
                
                # 2. Jalur Implisit (AI Confidence)
                confidence_match = conf >= THRESHOLD
                
                # Jika keduanya gagal, berarti lokasi tidak valid
                if not keyword_match and not confidence_match:
                    is_valid = False
            
            if not is_valid:
                missing_locations.append(field)
                predictions[f"lokasi_{field}"] = None # Normalisasi ke null

        is_complete = len(missing_locations) == 0

        # Kembalikan struktur dictionary yang optimal untuk frontend Next.js
        return {
            "status_tiket": "Open" if is_complete else "Draft",
            "is_complete": is_complete,
            "requires_follow_up": not is_complete,
            "missing_fields": missing_locations, # Berisi array string misal: ["lantai", "zona"]
            "teks_asli": teks_mentah,
            "teks_bersih": teks_bersih,
            "predictions": predictions,
            "confidences": {
                "tipe_aset": round(confidences[0], 3),
                "lokasi_gedung": round(confidences[1], 3),
                "lokasi_lantai": round(confidences[2], 3),
                "lokasi_zona": round(confidences[3], 3),
                "kategori_dept": round(confidences[4], 3),
                "severity_awal": round(confidences[5], 3)
            },
            "trigger_whatsapp_alert": True if prediction[5] in ["Berat", "Fatal", "Tinggi"] else False,
            "waktu_komputasi_detik": round(time.time() - t0, 3)
        }