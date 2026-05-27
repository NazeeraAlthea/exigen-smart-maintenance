# Rencana Arsitektur Staging Tabel (Komplain Perbaikan)

Ide Anda untuk membuat **tabel staging** (`KomplainPerbaikan`) adalah pendekatan arsitektur yang sangat tepat dan paling aman! 

Dengan menggunakan tabel staging, kita bisa memisahkan "Komplain Mentah hasil NLP" (Fase 1) dari "Riwayat Perbaikan Aset yang Sah" (Fase 2). Ini menjamin kebersihan data yang nantinya akan disuapkan ke model Prediktif Maintenance Jose dan Melvin.

## Proposed Architecture

1. **Fase 1 (NLP & Staging):** Saat pengguna (Karyawan) mengirim keluhan via Voice/Text, Next.js akan memanggil NLP (FastAPI) dan hasilnya disimpan ke dalam tabel **`KomplainPerbaikan`**. Di tabel ini belum ada ikatan mutlak dengan ID Aset secara fisik.
2. **Fase 1.5 (Tinjauan / Assign Aset):** Teknisi atau Admin melihat daftar `KomplainPerbaikan` yang masuk, lalu memvalidasinya dengan memilih Aset yang tepat dari `MasterAsset` (misalnya memindai QR Code atau memilih dari dropdown).
3. **Fase 2 (Tiket Resmi):** Setelah Aset divalidasi, data dipindahkan/diteruskan ke tabel **`AssetComplaint`** (atau nama lainnya, tabel utama perbaikan) tempat tiket akan dieksekusi (Servis -> Selesai -> Perhitungan Biaya -> dll). Model Prediktif Jose akan membaca murni dari `AssetComplaint` dan `MasterAsset`.

## Proposed Database Schema Changes

Saya akan membatalkan tambahan kolom lokasi di `AssetComplaint` sebelumnya, dan membuat tabel baru murni untuk Staging:

#### [NEW] Model `KomplainPerbaikan` di `schema.prisma`
```prisma
model KomplainPerbaikan {
  id                  String   @id
  teksKeluhan         String
  predTipeAset        String
  predLokasiGedung    String
  predLokasiLantai    String
  predLokasiZona      String
  predKategoriDept    String
  predSeverityAwal    String
  
  // Field pendukung
  missingFields       String?  // JSON string jika butuh follow-up
  isComplete          Boolean
  requiresFollowUp    Boolean
  statusStaging       String   @default("OPEN") // OPEN, DRAFT, ASSIGNED
  tanggalDibuat       DateTime @default(now())
}
```

## ⚠️ User Review Required

Apakah rancangan alur (Fase 1 -> Staging -> Fase 2) di atas sudah sesuai dengan bayangan Anda dan kebutuhan tim Data Science (Jose/Melvin)? 

**Jika Anda setuju:**
1. Saya akan membuat tabel `KomplainPerbaikan` ini.
2. Mengubah endpoint API (`/api/ticket/predict`) untuk menyimpan payload NLP ke tabel ini, bukan lagi memaksakan pembuatan *dummy asset* `AST-PENDING` di tabel `AssetComplaint`.
3. Menyiapkan UI/Halaman baru di dashboard bernama **"Inbox Komplain"** (atau sejenisnya) agar admin bisa melihat komplain dari NLP dan meneruskannya (assign ke `AssetComplaint`).
