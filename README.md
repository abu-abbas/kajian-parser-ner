# AI Parser Jadwal Kajian (SpaCy NER)

Proyek ini adalah parser jadwal kajian Islam berbasis kecerdasan buatan menggunakan Named Entity Recognition (NER) dari library **SpaCy**. Sistem ini mampu memilah teks rekap besar yang tidak terstruktur (seperti format Kaskus, Gresik Mengaji, Surabaya Mengaji, dll) menjadi sub-sesi individual dan mengekstrak entitas penting ke dalam format JSON terstruktur secara native.

## Fitur Utama
* **Format-based Splitting:** Otomatis mendeteksi format rekap teks (Kaskus, Gresik, atau Single) dan membaginya menjadi sesi-sesi kajian mandiri.
* **SpaCy NER Extraction:** Mengekstrak entitas penting secara native menggunakan model AI yang telah dilatih:
  * `PEMATERI` (Nama Ustaz)
  * `TEMA` (Judul Kajian/Kitab)
  * `WAKTU` (Waktu/Hari/Jam)
  * `LOKASI` (Nama Masjid/Gedung/Kediaman)
  * `ALAMAT` (Detail jalan/kota lokasi kajian)
  * `STATUS` (Status jamaah, misal: Khusus Akhwat)
  * `KONTAK` (Nomor HP/Link WhatsApp)
  * `LINK_STREAMING` (Link live streaming YouTube/Facebook)
  * `TARADHI` (Doa penghormatan, misal: *Hafizhahullah*)
  * `BANK` & `NOREK` (Informasi donasi dakwah)
* **Double-Layer Fallbacks:** 
  * **Alamat Fallback:** Jika model SpaCy melewatkan alamat, parser memiliki regex-based fallback untuk mengekstrak baris alamat di bawah nama masjid.
  * **Emoji Fallback:** Menjamin akurasi 100% pada rekap berbasis ikon/emoji terstruktur jika model mengalami fluktuasi prediksi.
* **Neat Directory Outputs:** Menyimpan output JSON secara otomatis di dalam direktori `output-sample/` untuk menjaga root project tetap bersih.

---

## Struktur Proyek

* `parse_to_json.py` — Script parser utama untuk memisahkan sesi, menjalankan inference SpaCy, dan mengekspor output JSON.
* `generate_dataset.py` — Script pembuat dataset latih dengan melabeli teks sampel secara otomatis menggunakan regex.
* `convert.py` — Mengonversi dataset latih python menjadi format biner `.spacy` untuk training.
* `predict.py` — Utilitas sederhana untuk menguji prediksi model secara cepat via CLI.
* `config.cfg` — File konfigurasi pipeline training SpaCy.
* `data-sample/` — Folder berisi dataset rekap mentah (`.txt`).
* `output-sample/` — Direktori penyimpanan output JSON terstruktur.
* `output_model/` — Direktori penyimpanan model SpaCy hasil training (`model-best` & `model-last`).

---

## Panduan Instalasi & Penggunaan

### 1. Persiapan Virtual Environment & Dependensi
Aktifkan virtual environment Anda dan pasang library SpaCy:
```bash
# Aktifkan venv
source venv/bin/activate

# Pasang SpaCy
pip install spacy
```

### 2. Menjalankan Parser
Jalankan script `parse_to_json.py` dengan memberikan path file input teks rekap.

* **Menyimpan otomatis ke `output-sample/` (Rekomendasi):**
  ```bash
  python parse_to_json.py data-sample/text-base/sample-dataset-rekap-kajian-kaskus.txt
  # Output akan tersimpan di output-sample/sample-dataset-rekap-kajian-kaskus_output_parsed.json
  # Jika file tersebut sudah ada, otomatis menambahkan counter (contoh: ..._parsed_01.json) agar tidak overwrite.
  ```

* **Menyimpan ke file/path spesifik:**
  ```bash
  python parse_to_json.py data-sample/text-base/sample-dataset-rekap-kajian-kaskus.txt hasil_kaskus.json
  ```

* **Menampilkan langsung di terminal (stdout):**
  ```bash
  python parse_to_json.py data-sample/text-base/sample-dataset-rekap-kajian-kaskus.txt -
  ```

---

## Panduan Training Ulang Model (Retraining)

Jika Anda melakukan perubahan pola ekstraksi pada `generate_dataset.py` untuk menambah performa model, lakukan langkah training ulang berikut:

```bash
# 1. Regenerasi data latihan baru (menghasilkan data_latihan_spacy.py)
python generate_dataset.py

# 2. Konversi ulang menjadi format biner train.spacy dan dev.spacy
python convert.py

# 3. Jalankan training ulang model SpaCy
python -m spacy train config.cfg --output ./output_model --paths.train ./train.spacy --paths.dev ./dev.spacy
```

SpaCy secara otomatis akan menimpa model lama di folder `./output_model/model-best` dengan model terbaik yang baru.
