---
title: Production Backend Deployment (Multi-Input Pure API)
description: Alur bertahap menguji API FastAPI multi-input secara lokal dan menyiapkan berkas untuk hosting gratisan.
---

# Steps

1. **Step 1: Dependency Freeze & Requirements (Kesiapan Berkas)**
   - Agent wajib mengecek `venv` lokal dan mengabaikan library global Mac M2.
   - Generate file `requirements.txt` yang berisi library esensial: `fastapi`, `uvicorn`, `spacy`, `pydantic`, dan `python-multipart` (wajib untuk menangani upload file/form).

2. **Step 2: Local Server Test & Verification (Uji Coba Swagger UI)**
   - Jalankan server FastAPI lokal menggunakan perintah `uvicorn app:app --reload`.
   - Buka dokumentasi otomatis di `http://127.0.0.1:8000/docs` (Swagger UI).
   - Pandu user melakukan uji coba interaktif di browser:
     - Mengirim teks via form (`text_input`).
     - Mengunggah file rekap lokal via file picker (`file_input`).
   - Pastikan model `model-best` termuat dengan mulus ke RAM tanpa kendala.

3. **Step 3: Git Sanity Check & Cleanup (Pembersihan Git)**
   - Jalankan pengecekan pada file `.gitignore`.
   - Pastikan folder raksasa `venv/`, `output_model/`, dan file biner `*.spacy` sudah aman terkunci agar tidak ikut ter-push ke GitHub.

4. **Step 4: Git Push & Deployment Config (Meluncur ke Awan)**
   - Pandu user melakukan commit akhir dan menjalankan `git push origin main`.
   - Berikan panduan konfigurasi *Start Command* pada dashboard hosting target (Render / Hugging Face): `uvicorn app:app --host 0.0.0.0 --port $PORT`.