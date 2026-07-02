---
name: parser_helper
description: Mengotomatisasi pengecekan venv, proteksi berkas .gitignore, dan pembuatan requirements.txt di Mac M2.
---

# Parser Helper Skill

Gunakan skill ini ketika user meminta bantuan terkait kesiapan berkas deployment, pengecekan dependensi Python, atau saat terminal lokal mengalami kendala.

## 🐍 Cara Menggunakan Script Python Internal:
Agent dapat mengeksekusi fungsi otomatis yang ada di dalam folder `scripts/parser_helpers.py` untuk membantu tugas-tugas berikut:

1. **Cek Proteksi Git:** Jalankan fungsi `verify_git_protection()` untuk memastikan folder `venv/` dan `output_model/` aman dari push GitHub.
2. **Cek Status Kotak Pasir:** Jalankan fungsi `check_venv_status()` untuk memverifikasi keaktifan environment.
3. **Generate Requirements:** Jalankan fungsi `generate_clean_requirements()` untuk membuat berkas `requirements.txt` yang bersih.

## 🚨 Penanganan Kendala Terminal (Mode Hemat Token)
Jika eksekusi script di atas mengalami kendala hak akses (permission) atau terbentur batasan sandbox lingkungan lokal, JANGAN memutar kode. Segera tuliskan perintah terminal (`command-line`) utuh di bawah ini agar user bisa mengeksukisnya secara manual:
- `source venv/bin/activate && uvicorn app:app --reload`
