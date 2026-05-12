# Face Recognition Service API Documentation

Semua endpoint utama (kecuali *health checks*) membutuhkan autentikasi menggunakan header `X-API-Key`.

---

## Daftar Isi
1. [Health Checks](#1-health-checks)
2. [Face Operations](#2-face-operations)
   - [Detect Faces](#post-v1facesdetect)
   - [Enroll Face](#post-v1facesenroll)
   - [Verify Face (1:1)](#post-v1facesverify)
   - [Recognize Face (1:N)](#post-v1facesrecognize)
3. [Subject Management](#3-subject-management)
   - [Get Subject Info](#get-v1facessubjectssubject_id)
   - [Delete Subject](#delete-v1facessubjectssubject_id)

---

## 1. Health Checks

### `GET /healthz`
Mengecek status ketersediaan layanan (*liveness probe*).

**Request Headers:** None

**Response (200 OK):**
```json
{
  "status": "ok",
  "service": "face-service",
  "version": "1.0.0"
}
```

### `GET /readyz`
Mengecek apakah layanan sudah siap, database terkoneksi, dan model AI sudah dimuat (*readiness probe*).

**Request Headers:** None

**Response (200 OK):**
```json
{
  "status": "ready",
  "database": "connected",
  "model": "loaded"
}
```

---

## 2. Face Operations

### `POST /v1/faces/detect`
Mendeteksi letak wajah dan fitur *landmarks* pada gambar tanpa melakukan pencocokan.

**Request Headers:**
- `X-API-Key`: `<API_SECRET_KEY>`

**Request Body (multipart/form-data):**
- `image` (File, required): File gambar/foto yang akan diproses.

**Response (200 OK):**
```json
{
  "faces": [
    {
      "box": [120.5, 80.2, 250.0, 310.5],
      "confidence": 0.998,
      "landmarks": {
        "left_eye": [150.2, 120.1],
        "right_eye": [210.4, 118.9],
        "nose": [180.5, 170.3],
        "mouth_left": [155.0, 220.1],
        "mouth_right": [205.8, 219.5]
      }
    }
  ],
  "image_quality": {
    "width": 640,
    "height": 480
  }
}
```

### `POST /v1/faces/enroll`
Mendaftarkan wajah (*embedding*) untuk seorang subjek (user/mahasiswa). Jika subjek belum ada, akan otomatis dibuat.

**Request Headers:**
- `X-API-Key`: `<API_SECRET_KEY>`

**Request Body (multipart/form-data):**
- `subject_id` (String, required): ID identitas pengguna (contoh: NPM mahasiswa).
- `image` (File, required): File foto wajah (pastikan hanya ada 1 wajah yang jelas).

**Response (200 OK):**
```json
{
  "subject_id": "12345678",
  "embedding_id": 1,
  "faces_detected": 1,
  "quality_score": 0.995,
  "status": "stored"
}
```

### `POST /v1/faces/verify`
Melakukan verifikasi wajah 1:1. Memastikan apakah foto yang dikirim cocok dengan data `subject_id` yang terdaftar.

**Request Headers:**
- `X-API-Key`: `<API_SECRET_KEY>`

**Request Body (multipart/form-data):**
- `subject_id` (String, required): ID pengguna yang ingin divalidasi.
- `image` (File, required): Foto wajah terbaru (misal: hasil selfie saat absen).
- `threshold` (Float, optional): Batas minimum skor kecocokan (default dikonfigurasi di server).

**Response (200 OK):**
```json
{
  "subject_id": "12345678",
  "score": 0.88,
  "threshold": 0.75,
  "match": true
}
```
*(Catatan: Jika `match` adalah `true`, berarti wajah tersebut terverifikasi milik `subject_id` yang diminta).*

### `POST /v1/faces/recognize`
Mencari wajah (1:N recognition). Mengirimkan foto wajah tunggal, dan mengembalikan K subjek teratas yang paling mirip.

**Request Headers:**
- `X-API-Key`: `<API_SECRET_KEY>`

**Request Body (multipart/form-data):**
- `image` (File, required): Foto wajah untuk dicari identitasnya.
- `top_k` (Integer, optional): Jumlah maksimum hasil pencarian teratas (default: 5).

**Response (200 OK):**
```json
{
  "matches": [
    {
      "subject_id": "12345678",
      "score": 0.89
    },
    {
      "subject_id": "87654321",
      "score": 0.45
    }
  ]
}
```

---

## 3. Subject Management

### `GET /v1/faces/subjects/{subject_id}`
Melihat status dan jumlah data wajah (embeddings) dari seorang pengguna terdaftar.

**Request Headers:**
- `X-API-Key`: `<API_SECRET_KEY>`

**Path Parameters:**
- `subject_id` (String): ID pengguna.

**Response (200 OK):**
```json
{
  "subject_id": "12345678",
  "status": "active",
  "total_embeddings": 3,
  "active_embeddings": 3,
  "created_at": "2026-05-12T10:00:00Z"
}
```

### `DELETE /v1/faces/subjects/{subject_id}`
Menghapus subjek beserta semua data foto/wajah (*embeddings*) yang terkait dengannya secara permanen.

**Request Headers:**
- `X-API-Key`: `<API_SECRET_KEY>`

**Path Parameters:**
- `subject_id` (String): ID pengguna yang akan dihapus.

**Response (200 OK):**
```json
{
  "subject_id": "12345678",
  "deleted": true,
  "message": "Subject '12345678' and all templates deleted."
}
```
