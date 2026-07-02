---
title: Pure Text Parsing Dataset Builder
description: Alur bertahap memproses file teks lokal menjadi data latihan SpaCy NER.
---
# Steps
1. **Step 1 (Ingestion):** Baca seluruh file `.txt` di folder `data-sample/text-base/` seutuhnya.
2. **Step 2 (NER Tagging):** Cari entitas wajib & dinamis pake `@aturan-kurasi-kajian` beserta index hurufnya.
3. **Step 3 (Relation Alignment):** Jodohkan entitas (Pemateri, Waktu, Tempat) agar terikat kuat per sesi kajian, jangan sampai tertukar.
4. **Step 4 (Export):** Bungkus jadi format array Python `TRAIN_DATA` dan simpan ke file `training/data_latihan_spacy.py`.
