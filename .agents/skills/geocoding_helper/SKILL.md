---
name: geocoding_helper
description: Skill khusus untuk memvalidasi alamat mentah hasil parser dan mengubahnya menjadi koordinat GPS via Geopy.
---

# Code Review & Integration Skill for Geocoding

## Kapan Menggunakan Skill Ini?
- Ketika ada data lokasi baru yang baru selesai di-parse dari teks/OCR.
- Ketika mempersiapkan payload JSON untuk di-push ke database Supabase.

## Panduan Penanganan Frontend Safety (Anti-Crash):
- Pastikan skema JSON yang dihasilkan menyertakan objek `data_tambahan` dalam bentuk yang aman.
- Gunakan fallback value atau pastikan properti krusial seperti `latitude` dan `longitude` bertipe data `float` atau `null` (jangan biarkan kosong atau bertipe string).

## Cara Penggunaan Script Helper:
- Jalankan script `.agents/skills/geocoding_helper/scripts/clean_geo.py` di dalam folder ini untuk memvalidasi string alamat secara lokal sebelum melakukan hit ke API.
