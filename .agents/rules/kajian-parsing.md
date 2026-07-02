---
trigger: always_on
description: Aturan Dynamic NER & Kebal Typo khusus untuk Pure Text Parsing.
---

# 🚨 RESPECT BAHASA INDONESIA
Wajib menggunakan bahasa indonesi setiap kali memberikan feedback atau penjelasan apapun

# 🚨 PRIORITAS UTAMA: MULTI-INPUT PURE TEXT PARSING ONLY
Untuk fase pengembangan saat ini, fokus 100% pada pemrosesan inputan user berupa teks langsung (Form) atau unggahan file (.txt). Jalur pemrosesan gambar/OCR sengaja dinonaktifkan sementara (di-bypass dengan HTTP 501).

# 📑 ATURAN PEMROSESAN TEKS KAJIAN
1. **Kebal Typo OCR/Text:** Jangan ubah typo bawaan teks (Contoh: 'U5tadz', 'Ma5j1d'). Masukkan langsung ke label entitas target.
2. **NER Dinamis:** - Wajib cari: `PEMATERI`, `LOKASI`, `WAKTU`, `TEMA`, `STATUS`.
   - Otomatis buat label baru jika ada info berharga: `BANK`, `NOREK`, `KONTAK`, `LINK_STREAMING`, `FASILITAS`.
3. **Presisi Index:** Hitung start & end index karakter secara 0-indexed tanpa *overlapping*.
4. **Relation Extraction Check:** Kelompokkan entitas secara spasial berdasarkan blok sesinya masing-masing agar data antar sesi tidak tertukar dalam satu output JSON.

# 💻 ENVIRONMENT SAFETY & TOKEN EFFICIENCY (ATURAN WAJIB MAC M2)
1. **Dilarang Install Global:** Agent TIDAK BOLEH menjalankan perintah `pip install` secara global di sistem macOS.
2. **Wajib Virtual Environment (venv):** Setiap kali Agent ingin mengeksekusi script Python (`build_dataset.py`, `generate_dataset.py`, `app.py`) atau menambah library baru, Agent WAJIB memastikan bahwa Virtual Environment (`venv`) sudah aktif.
3. **Pengecekan Tanda Aktif:** Pastikan terminal berada di direktori proyek dan jalankan `source venv/bin/activate` sebelum melakukan eksekusi perintah Python apa pun.
4. **Mode Eksekusi Manual (Hemat Token):** Jika Agent kepentok batasan *sandbox* lingkungan lokal, dilarang keras mencoba kodingan keliling/memutar yang menghabiskan token context. Cukup tuliskan perintah terminal (`command-line`) atau instruksi eksekusinya dengan jelas agar user yang mengeksekusinya secara manual di terminal Mac M2.
