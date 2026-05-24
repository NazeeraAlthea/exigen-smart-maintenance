import json
import random

def predict_ticket_from_text(ticket_id: str, text: str) -> dict:
    """
    Simulasi/Mock up Inference Model NLP Ticketing.
    Dalam produksi, Anda bisa me-load model transformer, atau menggunakan google-genai
    untuk mengekstrak field dari teks natural.
    """
    text_lower = text.lower()
    
    # Simple rule-based extraction for simulation
    kategori = "Umum"
    severity = "Ringan"
    tipe_aset = "Unknown"
    
    if "bocor" in text_lower or "air" in text_lower or "pipa" in text_lower:
        kategori = "Plumbing"
        severity = "Sedang"
        tipe_aset = "Sistem Perpipaan"
    elif "mati" in text_lower or "listrik" in text_lower or "konslet" in text_lower:
        kategori = "Electrical"
        severity = "Kritis"
        tipe_aset = "Sistem Kelistrikan"
    elif "ac" in text_lower or "panas" in text_lower:
        kategori = "HVAC"
        severity = "Sedang"
        tipe_aset = "AC Split"
    elif "pecah" in text_lower or "retak" in text_lower:
        kategori = "Sipil"
        severity = "Berat"
        tipe_aset = "Infrastruktur Gedung"

    if "parah" in text_lower or "meledak" in text_lower:
        severity = "Fatal"

    return {
        "status_proses": "Sukses",
        "tiket_id": ticket_id,
        "hasil_prediksi_ai": {
            "tipe_aset": tipe_aset,
            "lokasi_gedung": "Gedung Utama (TBD)",
            "lokasi_lantai": "Lantai 1 (TBD)",
            "lokasi_zona": "Zona A (TBD)",
            "kategori_departemen": kategori,
            "tingkat_severity": severity
        },
        "saran_tindakan_sistem": f"Jadwalkan inspeksi teknisi {kategori} sesegera mungkin berdasarkan keluhan: {text}"
    }
