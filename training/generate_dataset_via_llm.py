#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_dataset_via_llm.py
═══════════════════════════
Reads raw text files, splits them into logical sessions, sends each session to Gemini/Qwen API,
receives structured annotations (entities), calculates character boundaries,
and writes the output to training/data_latihan_spacy.py.

Opsi CLI:
  --force           Abaikan cache dan panggil LLM ulang dari awal.
  --sample <name>   Hanya memproses file sampel tertentu yang cocok (misal: '03' atau 'sample-dataset-03.txt').
"""

import os
import re
import sys
import glob
import json
import requests
from typing import List, Optional
import google.generativeai as genai
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# 1. Setup API Key and LLM Provider
load_dotenv()  # Muat variabel dari berkas .env
provider = os.environ.get("LLM_PROVIDER", "gemini").lower()

if provider == "gemini":
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or api_key == "AIzaSyYourGeminiAPIKeyHere":
        print("❌ Error: GEMINI_API_KEY belum disetel dengan benar di berkas .env.")
        print("💡 Silakan buka berkas .env di root proyek dan ganti 'AIzaSyYourGeminiAPIKeyHere' dengan API Key Gemini riil Anda.")
        sys.exit(1)
    genai.configure(api_key=api_key)
elif provider == "qwen":
    api_key = os.environ.get("QWEN_API_KEY")
    if not api_key or api_key == "sk-YourQwenAPIKeyHere" or not api_key.strip():
        print("❌ Error: QWEN_API_KEY belum disetel dengan benar di berkas .env.")
        print("💡 Silakan buka berkas .env di root proyek dan ganti 'sk-YourQwenAPIKeyHere' dengan API Key Qwen riil Anda.")
        sys.exit(1)
else:
    print(f"❌ Error: LLM Provider '{provider}' tidak dikenal. Gunakan 'gemini' or 'qwen'.")
    sys.exit(1)

# 2. Define Pydantic Schema for Structured Outputs
class Entity(BaseModel):
    text: str = Field(description="Exact substring of the entity text from the input")
    label: str = Field(description="Label entitas. Wajib cari: PEMATERI, LOKASI, WAKTU, TEMA, STATUS. Anda bebas membuat label baru lainnya (UPPERCASE dengan underscore) jika menemukan informasi bermanfaat lainnya seperti BANK, NOREK, KONTAK, LINK_STREAMING, FASILITAS, dll.")

class SessionAnnotation(BaseModel):
    entities: List[Entity]

# 3. Helper to split text to sessions (same logic as perfect dataset)
def split_text_to_sessions(text, filename):
    fn = filename.lower()
    if "gresik" in fn:
        blocks = re.split(r'\n\n+', text)
        return [b.strip() for b in blocks if b.strip().startswith('⏰')]
    elif "kaskus" in fn:
        matches = re.findall(r'((?:🕌|🏢|🏡).+?)(?:\n\*\s*$)', text, re.DOTALL | re.MULTILINE)
        return [m.strip() for m in matches if m.strip()]
    elif "13" in fn:
        blocks = re.split(r'\n\n+', text)
        sessions = [b.strip() for b in blocks if b.strip().startswith('⏰')]
        donation_block = next((b.strip() for b in blocks if "💳" in b or "BSI -" in b), "")
        if donation_block:
            sessions.append(donation_block)
        return sessions
    else:
        return [text.strip()]

def get_entities_via_gemini(session_text: str, filename_prefix: str = "") -> List[Entity]:
    """Calls Gemini API with structured schema to extract entities from text."""
    system_instruction = (
        "Anda adalah pakar annotator NLP bahasa Indonesia yang sangat teliti.\n"
        "Tugas Anda mengekstrak entitas-entitas penting dari kutipan teks jadwal kajian sunnah yang diberikan.\n\n"
        "Entitas target yang wajib ditemukan (jika ada):\n"
        "- PEMATERI: Nama Ustadz atau Ustadzah (contoh: 'Ustadz Abdurrahman Keken, Lc., M.H'). Jangan menyertakan doa taradhi di dalamnya.\n"
        "- TEMA: Judul kajian atau kitab/bahasan (contoh: 'TUJUAN HIDUP MUSLIM SETIAP HARI', 'Kitab Al Wajiz').\n"
        "- WAKTU: Keterangan tanggal, hari, jam, atau sesi waktu kajian (contoh: 'Ba'da Subuh', 'AHAD, 25 JANUARI 2026', '16.00 - Selesai').\n"
        "- LOKASI: Nama tempat fisik atau masjid (contoh: 'MASJID DARUSSAKINAH').\n"
        "- STATUS: Informasi status/keterangan penting (contoh: 'GRATIS', 'Khusus Akhwat', 'Terbuka untuk umum', 'Ditunda').\n\n"
        "NER DINAMIS:\n"
        "Anda sangat disarankan untuk secara dinamis membuat label baru (ditulis dalam format UPPERCASE dengan underscore) apabila menemukan informasi penting dan bermanfaat lainnya di dalam teks. Contoh:\n"
        "- ALAMAT: Alamat jalan detail tempat lokasi kajian fisik berada.\n"
        "- KATEGORI: Status target/audiens (contoh: 'Ikhwan & Akhwat').\n"
        "- METODE: Nama platform online kajian (contoh: 'Zoom', 'YouTube', 'AshiilTV').\n"
        "- LINK_STREAMING: URL link media sosial/streaming (contoh: 'https://bit.ly/ashiilapp').\n"
        "- LINK_MAPS: URL tautan peta lokasi fisik (contoh: 'https://maps.app.goo.gl/...').\n"
        "- BANK: Nama bank transfer donasi (contoh: 'BANK SYARIAH INDONESIA').\n"
        "- NOREK: Nomor rekening transfer saja (contoh: '81 755 0707 9').\n"
        "- KONTAK: Nomor telepon/WhatsApp narahubung saja (contoh: '0812-70805555').\n"
        "- FASILITAS: Fasilitas penunjang yang disediakan di lokasi.\n"
        "- TARADHI: Lafadz doa penghormatan untuk asatidzah/sahabat (contoh: 'حفظه الله', 'hafidzahullah').\n"
        "- ...atau label logis lainnya yang bermanfaat untuk diekstrak.\n\n"
        "ATURAN MUTLAK:\n"
        "1. Nilai 'text' HARUS persis sama karakter-demi-karakter (exact substring) dengan teks di dalam input. Jangan ubah huruf besar/kecil, typo, atau tanda baca.\n"
        "2. Jangan pernah menebak atau mengarang data jika tidak tertulis di input."
    )

    # Use gemini-2.5-flash for fast and cost-effective extraction
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=SessionAnnotation,
            temperature=0.1,  # Low temperature for highly deterministic output
        ),
        system_instruction=system_instruction
    )

    import time
    max_retries = 5
    base_delay = 5.0

    for attempt in range(max_retries):
        try:
            response = model.generate_content(session_text)
            # Jeda pasif 2 detik di setiap pemanggilan sukses untuk menjaga kestabilan rate limit gratisan
            time.sleep(2.0)

            data = json.loads(response.text)
            if filename_prefix:
                save_llm_response(session_text, data, filename_prefix)
            entities = [Entity(**e) for e in data.get("entities", [])]
            return entities
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "ResourceExhausted" in err_str or "quota" in err_str.lower():
                delay = base_delay * (2 ** attempt)
                print(f"  ⚠️ Terkena batas kuota (Rate Limit/429). Menunggu {delay} detik sebelum mencoba kembali (Percobaan {attempt+1}/{max_retries})...")
                time.sleep(delay)
            else:
                print(f"  ⚠️ Gagal memanggil Gemini API: {e}")
                return []

    print("  ❌ Gagal mendapatkan respon dari Gemini setelah batas percobaan karena kuota habis.")
    return []

def get_entities_via_qwen(session_text: str, filename_prefix: str = "") -> List[Entity]:
    """Calls Qwen Cloud API with JSON Mode to extract entities from text."""
    api_key = os.environ.get("QWEN_API_KEY")
    base_url = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model_name = os.environ.get("QWEN_MODEL", "qwen-plus")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    system_instruction = (
        "Anda adalah pakar annotator NLP bahasa Indonesia yang sangat teliti.\n"
        "Tugas Anda mengekstrak entitas-entitas penting dari kutipan teks jadwal kajian sunnah yang diberikan.\n\n"
        "Kembalikan data dalam format JSON dengan struktur berikut:\n"
        "{\n"
        "  \"entities\": [\n"
        "    {\"text\": \"exact substring dari input\", \"label\": \"NAMA_LABEL\"}\n"
        "  ]\n"
        "}\n\n"
        "Entitas target yang wajib ditemukan (jika ada):\n"
        "- PEMATERI: Nama Ustadz atau Ustadzah (contoh: 'Ustadz Abdurrahman Keken, Lc., M.H'). Jangan menyertakan doa taradhi di dalamnya.\n"
        "- TEMA: Judul kajian atau kitab/bahasan (contoh: 'TUJUAN HIDUP MUSLIM SETIAP HARI', 'Kitab Al Wajiz').\n"
        "- WAKTU: Keterangan tanggal, hari, jam, atau sesi waktu kajian (contoh: 'Ba'da Subuh', 'AHAD, 25 JANUARI 2026', '16.00 - Selesai').\n"
        "- LOKASI: Nama tempat fisik atau masjid (contoh: 'MASJID DARUSSAKINAH').\n"
        "- STATUS: Informasi status/keterangan penting (contoh: 'GRATIS', 'Khusus Akhwat', 'Terbuka untuk umum', 'Ditunda').\n\n"
        "NER DINAMIS:\n"
        "Anda sangat disarankan untuk secara dinamis membuat label baru (ditulis dalam format UPPERCASE dengan underscore) apabila menemukan informasi penting dan bermanfaat lainnya di dalam teks. Contoh:\n"
        "- ALAMAT: Alamat jalan detail tempat lokasi kajian fisik berada.\n"
        "- KATEGORI: Status target/audiens (contoh: 'Ikhwan & Akhwat').\n"
        "- METODE: Nama platform online kajian (contoh: 'Zoom', 'YouTube', 'AshiilTV').\n"
        "- LINK_STREAMING: URL link media sosial/streaming (contoh: 'https://bit.ly/ashiilapp').\n"
        "- LINK_MAPS: URL tautan peta lokasi fisik (contoh: 'https://maps.app.goo.gl/...').\n"
        "- BANK: Nama bank transfer donasi (contoh: 'BANK SYARIAH INDONESIA').\n"
        "- NOREK: Nomor rekening transfer saja (contoh: '81 755 0707 9').\n"
        "- KONTAK: Nomor telepon/WhatsApp narahubung saja (contoh: '0812-70805555').\n"
        "- FASILITAS: Fasilitas penunjang yang disediakan di lokasi.\n"
        "- TARADHI: Lafadz doa penghormatan untuk asatidzah/sahabat (contoh: 'حفظه الله', 'hafidzahullah').\n"
        "- ...atau label logis lainnya yang bermanfaat untuk diekstrak.\n\n"
        "ATURAN MUTLAK:\n"
        "1. Nilai 'text' HARUS persis sama karakter-demi-karakter (exact substring) dengan teks di dalam input. Jangan ubah huruf besar/kecil, typo, atau tanda baca.\n"
        "2. Kembalikan HANYA objek JSON yang valid tanpa markdown formatting block (tanpa ```json ... ```)."
    )

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": session_text}
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"}
    }

    import time
    max_retries = 5
    base_delay = 5.0
    url = f"{base_url.rstrip('/')}/chat/completions"

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)

            if response.status_code == 429:
                delay = base_delay * (2 ** attempt)
                print(f"  ⚠️ Terkena batas kuota Qwen Cloud (429). Menunggu {delay} detik (Percobaan {attempt+1}/{max_retries})...")
                time.sleep(delay)
                continue

            response.raise_for_status()
            res_data = response.json()
            content = res_data["choices"][0]["message"]["content"]

            # Jeda pasif 2 detik
            time.sleep(2.0)

            data = json.loads(content)
            if filename_prefix:
                save_llm_response(session_text, data, filename_prefix)
            entities = [Entity(**e) for e in data.get("entities", [])]
            return entities
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate limit" in err_str.lower():
                delay = base_delay * (2 ** attempt)
                print(f"  ⚠️ Terkena batas kuota Qwen (429). Menunggu {delay} detik...")
                time.sleep(delay)
            else:
                print(f"  ⚠️ Gagal memanggil Qwen API: {e}")
                time.sleep(base_delay)

    print("  ❌ Gagal mendapatkan respon dari Qwen setelah batas percobaan.")
    return []

def save_llm_response(session_text: str, data: dict, filename_prefix: str):
    """Saves raw LLM input text and parsed JSON response to output/llm_tmp for debugging."""
    try:
        os.makedirs("output/llm_tmp", exist_ok=True)
        safe_prefix = re.sub(r'[^a-zA-Z0-9_\-]', '_', filename_prefix)
        if not safe_prefix:
            import hashlib
            safe_prefix = hashlib.md5(session_text.encode('utf-8')).hexdigest()

        filepath = os.path.join("output/llm_tmp", f"{safe_prefix}.json")
        debug_payload = {
            "session_text": session_text,
            "response_data": data
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(debug_payload, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"  ⚠️ Gagal menyimpan debug output LLM: {e}")

def load_llm_response(filename_prefix: str, session_text: str) -> Optional[List[Entity]]:
    """Loads cached LLM response from output/llm_tmp if it exists and is valid."""
    try:
        safe_prefix = re.sub(r'[^a-zA-Z0-9_\-]', '_', filename_prefix)
        if not safe_prefix:
            import hashlib
            safe_prefix = hashlib.md5(session_text.encode('utf-8')).hexdigest()
            
        filepath = os.path.join("output/llm_tmp", f"{safe_prefix}.json")
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            response_data = data.get("response_data", {})
            entities = [Entity(**e) for e in response_data.get("entities", [])]
            return entities
    except Exception as e:
        print(f"  ⚠️ Gagal memuat cache untuk {filename_prefix}: {e}")
    return None

def get_entities(session_text: str, filename_prefix: str = "", force: bool = False) -> List[Entity]:
    """Router helper to select between Gemini and Qwen Cloud providers, with caching."""
    if not force and filename_prefix:
        cached = load_llm_response(filename_prefix, session_text)
        if cached is not None:
            print(f"    💾 Loaded from cache (output/llm_tmp/{filename_prefix}.json)")
            return cached

    if provider == "gemini":
        return get_entities_via_gemini(session_text, filename_prefix)
    elif provider == "qwen":
        return get_entities_via_qwen(session_text, filename_prefix)
    return []

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate training dataset via LLM API")
    parser.add_argument("--force", action="store_true", help="Abaikan cache di output/llm_tmp dan panggil LLM ulang dari awal")
    parser.add_argument("--sample", type=str, default=None, help="Nama berkas sampel spesifik yang ingin diproses saja (misal: sample-dataset-03.txt atau 03)")
    args = parser.parse_args()
    
    force_call = args.force
    sample_filter = args.sample

    model_title = "Gemini" if provider == "gemini" else "Qwen"
    print("=" * 70)
    print(f"🤖 Generating Clean Dataset via {model_title} LLM API (Structured Outputs)")
    print("=" * 70)

    input_folder = "input/sampling/text-base"
    files = sorted(glob.glob(os.path.join(input_folder, "*.txt")))

    all_data = []

    for fpath in files:
        fname = os.path.basename(fpath)
        
        # Filter berdasarkan parameter --sample jika diisi oleh user
        if sample_filter and sample_filter not in fname:
            continue

        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()
            # Clean surrogate characters just in case
            content = "".join(c for c in content if not (0xD800 <= ord(c) <= 0xDFFF))

        sessions = split_text_to_sessions(content, fname)
        print(f"📄 Processing {fname} ({len(sessions)} session blocks)...")

        for idx, session in enumerate(sessions):
            print(f"  ↳ Calling LLM ({provider}) for Session {idx+1}/{len(sessions)} ...")
            fn_base = fname[:-4] if fname.endswith('.txt') else fname
            raw_entities = get_entities(session, filename_prefix=f"{fn_base}_session_{idx+1}", force=force_call)

            # Resolve exact character coordinates (0-indexed)
            used_ranges = []
            resolved_ents = []

            # Sort by length of text descending to avoid substring collision
            sorted_raw = sorted(raw_entities, key=lambda x: len(x.text), reverse=True)

            for ent in sorted_raw:
                sub = ent.text.strip()
                if not sub:
                    continue
                label = ent.label

                # Search for substring position safely
                start_search = 0
                while True:
                    pos = session.find(sub, start_search)
                    if pos == -1:
                        # Try to find case-insensitive as fallback if LLM altered case slightly
                        pos = session.lower().find(sub.lower(), start_search)
                        if pos == -1:
                            break

                    end_pos = pos + len(sub)
                    # Check overlap
                    overlap = False
                    for us, ue in used_ranges:
                        if not (end_pos <= us or pos >= ue):
                            overlap = True
                            break

                    if not overlap:
                        resolved_ents.append((pos, end_pos, label))
                        used_ranges.append((pos, end_pos))
                        break

                    start_search = pos + 1

            # Sort final entities back by start index
            resolved_ents.sort(key=lambda x: x[0])

            if resolved_ents:
                all_data.append((session, resolved_ents, fname, idx))
                print(f"    ✅ Resolved {len(resolved_ents)} entities.")
            else:
                print("    ⚠️ No entities resolved for this session.")

    # Generate synthetic variations to enrich data (copying previous logic for safety)
    for session, entities, orig_fname, orig_idx in list(all_data):
        if "Masjid al-Ikhlash" in session and ", Jl." in session:
            variations = [
                ("Masjid An-Nur", "Jl. Diponegoro No. 12, Kel. Tegalsari, Kec. Tegalsari, Surabaya"),
                ("Masjid Al-Barkah", "Jl. Slamet Riyadi No. 5, Laweyan, Surakarta"),
                ("Musholla At-Taqwa", "Jl. Pemuda No. 45, Sekayu, Kec. Semarang Tengah, Kota Semarang"),
                ("Masjid Istiqlal", "Jl. Taman Wijaya Kusuma, Pasar Baru, Kec. Sawah Besar, Kota Jakarta Pusat"),
            ]
            for var_idx, (loc_name, addr_name) in enumerate(variations):
                old_loc = "Masjid al-Ikhlash - Delta Sari Indah"
                old_addr = "Jl. Anggrek VI no.36-38, Kureksari, Waru, Sidoarjo"
                new_session = session.replace(old_loc, loc_name).replace(old_addr, addr_name)
                # Parse using gemini for synthetic
                print(f"🤖 Parsing synthetic variation via LLM ({provider})...")
                fn_base = orig_fname[:-4] if orig_fname.endswith('.txt') else orig_fname
                synthetic_ents_raw = get_entities(new_session, filename_prefix=f"{fn_base}_session_{orig_idx+1}_synthetic_{var_idx+1}", force=force_call)

                # Resolve coordinates for synthetic
                s_used = []
                s_resolved = []
                s_sorted = sorted(synthetic_ents_raw, key=lambda x: len(x.text), reverse=True)
                for ent in s_sorted:
                    sub = ent.text.strip()
                    if not sub:
                        continue
                    label = ent.label
                    start_s = 0
                    while True:
                        pos = new_session.find(sub, start_s)
                        if pos == -1:
                            break
                        end_pos = pos + len(sub)
                        overlap = False
                        for us, ue in s_used:
                            if not (end_pos <= us or pos >= ue):
                                overlap = True
                                break
                        if not overlap:
                            s_resolved.append((pos, end_pos, label))
                            s_used.append((pos, end_pos))
                            break
                        start_s = pos + 1

                s_resolved.sort(key=lambda x: x[0])

                if s_resolved:
                    all_data.append((new_session, s_resolved, f"{orig_fname}_synthetic_{var_idx+1}", orig_idx))
            break

    # 4. Write data to training/data_latihan_spacy.py
    output_path = "training/data_latihan_spacy.py"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write('# -*- coding: utf-8 -*-\n')
        f.write('"""\n')
        f.write('data_latihan_spacy.py\n')
        f.write('═════════════════════\n')
        if provider == "gemini":
            model_info = "Gemini API (gemini-2.5-flash)"
        else:
            model_name = os.environ.get("QWEN_MODEL", "qwen-plus")
            model_info = f"Qwen API ({model_name})"
        f.write(f'Auto-generated by generate_dataset_via_llm.py using {model_info}.\n')
        f.write('SpaCy NER training data for Jadwal Kajian parsing.\n')
        f.write('"""\n\n')
        f.write('TRAIN_DATA = [\n')

        for i, (text, ents, *_) in enumerate(all_data):
            f.write(f'\n    # ── Entry {i + 1} ──\n')
            f.write(f'    (\n')
            f.write(f'        {text!r},\n')
            f.write(f'        {{"entities": [\n')
            for s, e, label in ents:
                snippet = text[s:e].replace('\n', '\\n')
                f.write(f'            ({s}, {e}, "{label}"),  # {snippet!r}\n')
            f.write(f'        ]}}\n')
            f.write(f'    ),\n')

        f.write(']\n')

    print("-" * 70)
    print(f"📊 Completed! Total training examples: {len(all_data)}")
    print(f"💾 Written to {output_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
