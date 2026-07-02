# AI Parser Jadwal Kajian (SpaCy Joint NER + RE)

Proyek ini adalah parser jadwal kajian Islam berbasis kecerdasan buatan menggunakan Named Entity Recognition (NER) dan Relation Extraction (RE) dari library **SpaCy**. Sistem ini mampu memilah teks rekap besar yang tidak terstruktur (seperti format Kaskus, Rekap Gresik, Rekap Batam, Surabaya Mengaji, dll) menjadi sub-sesi individual dan mengekstrak entitas penting ke dalam format JSON terstruktur secara native.

## Fitur Utama
* **Format-based Splitting:** Otomatis mendeteksi format rekap teks (Kaskus, Gresik, Batam, Surabaya Mengaji, atau Single Poster) dan membaginya menjadi sesi-sesi kajian mandiri menggunakan teknik *Strict Unit Partitioning*.
* **SpaCy Joint NER & RE Extraction:** Mengekstrak entitas penting secara native menggunakan model AI yang telah dilatih:
  * `PEMATERI` (Nama Ustaz)
  * `TEMA` (Judul Kajian/Kitab/Bahasan)
  * `WAKTU` (Waktu/Hari/Jam pelaksanaan)
  * `LOKASI` (Nama Masjid/Gedung/Kediaman)
  * `ALAMAT` (Detail jalan/kota lokasi kajian)
  * `KATEGORI` (Status segmen peserta, misal: *Khusus Akhwat & Anak-anak*)
  * `KONTAK` (Nomor HP/Link WhatsApp)
  * `LINK_STREAMING` (Tautan live streaming YouTube/Facebook/Telegram)
  * `LINK_MAPS` (Tautan resmi peta lokasi seperti Google Maps/Waze/Bit.ly peta)
  * `METODE` (Platform media online, misal: *Zoom*, *YouTube Live*, *Fatwa TV*)
  * `TARADHI` (Doa penghormatan pemateri, misal: *Hafizhahullah*, *Hafidzahullahu Ta'ala*)
  * `BANK` & `NOREK` (Informasi donasi dakwah)
* **Dual Training System (NER-only atau Joint NER + RE):** Anda dapat memilih untuk melatih model standard Named Entity Recognition saja atau melatih model gabungan NER + Relation Extraction menggunakan Neural Network kustom.
* **Double-Layer Fallbacks:** 
  * **Alamat Fallback:** Jika model SpaCy melewatkan alamat, parser memiliki regex-based fallback untuk mengekstrak baris alamat di bawah nama masjid.
  * **Emoji Fallback:** Menjamin akurasi 100% pada rekap berbasis ikon/emoji terstruktur jika model mengalami fluktuasi prediksi.
* **Neat Directory Outputs:** Menyimpan output JSON secara otomatis di dalam direktori `output/sampling/` untuk menjaga root project tetap bersih.

---

## Struktur Proyek

* **`src/`** — Kode program utama untuk produksi:
  * `parse_to_json.py` — Script parser utama untuk memisahkan sesi, menjalankan inference SpaCy, dan mengekspor output JSON.
  * `predict.py` — Utilitas sederhana untuk menguji prediksi model secara cepat via CLI.
  * `debug_ner.py` — Script utility untuk menganalisis dan mendebug hasil deteksi NER per berkas dan menyimpannya di folder `output/debugging/`.
* **`training/`** — Program pendukung untuk pembuatan dataset & training model:
  * `generate_dataset_via_llm.py` — Script pembuat dataset NER menggunakan LLM API (Gemini/Qwen) dilengkapi fitur caching file dan opsi pemrosesan berkas sampel spesifik.
  * `generate_dataset.py` — Script pembuat dataset NER menggunakan pemrosesan aturan Regex tradisional.
  * `generate_dataset_relations.py` — Script pembuat dataset relasi RE berbasis relasi spasial logis dari entitas terdeteksi.
  * `generate_perfect_dataset.py` — Script generator dataset acuan referensi terkurasi sempurna.
  * `build_dataset.py` — Script ingestion dasar untuk membaca file teks mentah sampel.
  * `data_latihan_spacy.py` — Dataset latih format Python list hasil ekstraksi NER terkurasi dengan exact substring matching.
  * `data_latihan_relations.jsonl` — Dataset latih relasi dalam format JSONL untuk pelatihan Relation Extraction.
  * `rel_component.py` — Kode komponen kustom (*custom component*) untuk mendaftarkan neural network relasi ke dalam sistem SpaCy v3.
  * `convert.py` — Mengonversi dataset latih NER python menjadi format biner `train.spacy` dan `dev.spacy`.
  * `convert_rel.py` — Mengonversi dataset relasi JSONL menjadi format biner `train_rel.spacy` dan `dev_rel.spacy`.
  * `config.cfg` — File konfigurasi pipeline training model NER saja.
  * `config_rel.cfg` — File konfigurasi pipeline training model Joint NER + RE.
* `input/sampling/` — Folder berisi dataset rekap mentah (`.txt`).
* `output/sampling/` — Direktori penyimpanan output JSON terstruktur.
* `output/debugging/` — Direktori penyimpanan hasil log debugging parser.
* `output/llm_tmp/` — Direktori penyimpanan cache berkas respon mentah JSON dari panggilan LLM API.
* `output/model_ner/` — Direktori penyimpanan model hasil training SpaCy NER-only.
* `output/model_rel/` — Direktori penyimpanan model hasil training SpaCy Joint NER + RE.

---

### 1. Persiapan Virtual Environment & Dependensi
Aktifkan virtual environment Anda dan pasang dependensi proyek:
```bash
# Aktifkan venv
source venv/bin/activate

# Pasang dependensi dari requirements.txt (termasuk SpaCy dan Google Generative AI SDK)
pip install -r requirements.txt
```

### 2. Setup Gemini API Key (Penting untuk Data Labeling)
Untuk menghasilkan data latihan SpaCy menggunakan pelabelan otomatis berbasis AI, Anda wajib mengonfigurasi API Key Gemini Anda:
1. Buka file **`.env`** yang terletak di root direktori proyek Anda.
2. Ganti nilai placeholder dengan API Key Gemini riil Anda:
   ```ini
   GEMINI_API_KEY="AIzaSyYourActualAPIKeyHere"
   ```
*(Catatan: Berkas `.env` telah ditambahkan ke `.gitignore` sehingga API Key Anda aman dari kebocoran ke repositori git).*

### 3. Menjalankan Parser
Jalankan script `src/parse_to_json.py` dengan memberikan path file input teks rekap.

* **Menyimpan otomatis ke `output/sampling/` menggunakan Model NER Saja (Default):**
  ```bash
  python src/parse_to_json.py input/sampling/text-base/sample-dataset-rekap-kajian-kaskus.txt
  ```

* **Menyimpan otomatis menggunakan Model Joint NER + RE (Gunakan flag `--use-rel`):**
  ```bash
  python src/parse_to_json.py input/sampling/text-base/sample-dataset-rekap-kajian-kaskus.txt --use-rel
  ```

* **Menyimpan ke file/path spesifik:**
  ```bash
  python src/parse_to_json.py input/sampling/text-base/sample-dataset-rekap-kajian-kaskus.txt hasil_kaskus.json
  ```

* **Menampilkan langsung di terminal (stdout):**
  ```bash
  python src/parse_to_json.py input/sampling/text-base/sample-dataset-rekap-kajian-kaskus.txt -
  ```

---

## Panduan Retraining Otomatis (`train.sh`)

Untuk mempermudah dan mempercepat proses retraining tanpa harus mengetik perintah yang sangat panjang, Anda dapat menggunakan script otomatis **`train.sh`** di root folder. 

Script ini akan otomatis mengaktifkan `venv`, meregenerasi dataset entities & relasi, mengonversinya ke format biner `.spacy`, dan menjalankan training SpaCy dalam **satu langkah**.

### 1. Berikan Hak Akses Executable (Hanya Sekali)
Jalankan perintah ini di terminal Anda untuk mengizinkan script berjalan:
```bash
chmod +x train.sh
```

### 2. Jalankan Training Sesuai Opsi Pilihan Anda:

* **Opsi A: Latih Model Joint NER + RE (Rekomendasi)**
  ```bash
  ./train.sh rel
  ```
  *Model terbaik akan disimpan ke folder `output/model_rel/model-best`.*

* **Opsi B: Latih Model NER Saja (Standar)**
  ```bash
  ./train.sh ner
  ```
  *Model terbaik akan disimpan ke folder `output/model_ner/model-best`.*

---

## Pembuatan Dataset via LLM API (`generate_dataset_via_llm.py`)

Skrip `training/generate_dataset_via_llm.py` digunakan untuk mengekstrak entitas-entitas penting dari seluruh file teks sampel di `input/sampling/text-base/` menggunakan LLM API (Gemini atau Qwen Cloud) untuk membuat dataset latihan SpaCy `training/data_latihan_spacy.py`.

### Fitur & Opsi CLI:
* **Caching Respon LLM:** Secara default, respon LLM disimpan di dalam folder `output/llm_tmp/`. Pemrosesan berikutnya untuk berkas/sesi yang sama akan langsung menggunakan cache ini guna menghemat waktu dan biaya token API.
* **Opsi `--force`:** Memaksa pemanggilan ulang LLM API dan menimpa cache yang sudah ada.
  ```bash
  python training/generate_dataset_via_llm.py --force
  ```
* **Opsi `--sample <pattern>`:** Memproses berkas sampel spesifik saja yang nama berkasnya mengandung pola tertentu (tidak men-scan seluruh file).
  ```bash
  python training/generate_dataset_via_llm.py --sample 03
  ```
* **Kombinasi Opsi:** Anda dapat menggabungkan opsi di atas untuk memperbarui cache berkas spesifik:
  ```bash
  python training/generate_dataset_via_llm.py --sample 03 --force
  ```
