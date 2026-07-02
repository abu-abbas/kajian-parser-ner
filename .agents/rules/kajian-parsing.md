---
name: aturan-kurasi-kajian
description: Aturan Dynamic NER & Kebal Typo khusus untuk Pure Text Parsing.
activation: always_on
---
# Aturan Pemrosesan Teks Kajian
1. **Kebal Typo OCR/Text:** Jangan ubah typo bawaan teks (Contoh: 'U5tadz', 'Ma5j1d'). Masukkan langsung ke label entitas target.
2. **NER Dinamis:** - Wajib cari: `PEMATERI`, `LOKASI`, `WAKTU`, `TEMA`, `STATUS`.
   - Otomatis buat label baru jika ada info berharga: `BANK`, `NOREK`, `KONTAK`, `LINK_STREAMING`, `FASILITAS`.
3. **Presisi Index:** Hitung start & end index karakter secara 0-indexed tanpa *overlapping*.

# 💻 ENVIRONMENT SAFETY (ATURAN WAJIB MAC M2)
1. **Dilarang Install Global:** Agent TIDAK BOLEH menjalankan perintah `pip install` secara global di sistem macOS.
2. **Wajib Virtual Environment (venv):** Setiap kali Agent ingin mengeksekusi script Python (`build_dataset.py` atau `generate_dataset.py`) atau menambah library baru, Agent WAJIB memastikan bahwa Virtual Environment (`venv`) sudah aktif.
3. **Pengecekan Tanda Aktif:** Pastikan terminal berada di direktori proyek dan jalankan `source venv/bin/activate` sebelum melakukan eksekusi perintah Python apa pun.
