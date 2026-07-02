#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parse_to_json.py
════════════════
Parses a rekap file (e.g. Kaskus, Gresik, or Single Session) into a structured JSON file.
Uses the trained SpaCy model to extract entities and groups them per session.
"""

import sys
import argparse
import os
import re
import json
import spacy
from typing import Any

# ═════════════════════════════════════════════════════════════
# Session Splitting Logic (adapted from generate_dataset.py)
# ═════════════════════════════════════════════════════════════

def _split_gresik(text):
    """Split Gresik rekap: sessions are ⏰-blocks separated by blank lines."""
    blocks = re.split(r'\n\n+', text)
    return [b.strip() for b in blocks if b.strip().startswith('⏰')]

def _split_kaskus(text):
    """Split Kaskus rekap: each session starts with 🕌/🏢/🏡, ends with standalone *."""
    matches = re.findall(
        r'((?:🕌|🏢|🏡).+?)(?:\n\*\s*$)',
        text,
        re.DOTALL | re.MULTILINE,
    )
    return [m.strip() for m in matches if m.strip()]

def _split_sample13_sessions(text):
    """Split Surabaya Mengaji rekap into sessions (⏰ blocks) and the donation footer."""
    blocks = re.split(r'\n\n+', text)
    sessions = [b.strip() for b in blocks if b.strip().startswith('⏰')]
    donation_block = ""
    for b in blocks:
        if "💳" in b or "BSI -" in b:
            donation_block = b.strip()
            break
    if donation_block:
        sessions.append(donation_block)
    return sessions

def _split_batam(text):
    """Split Batam rekap: sessions are blocks starting with a line like *MASJID...* or *MUSHOLLA...*"""
    # Pattern to match starting of a masjid or musholla block
    pattern = r'(?=\*(?:MASJID|MUSHOLLA|KAJIAN)\s+[^*]+\*)'
    blocks = re.split(pattern, text)
    
    header = ""
    # Check if first block contains date header
    first_block = blocks[0].strip()
    if not (first_block.startswith("*MASJID") or first_block.startswith("*MUSHOLLA") or first_block.startswith("*KAJIAN")):
        header = first_block
        blocks = blocks[1:]
        
    final_blocks = []
    for b in blocks:
        b_strip = b.strip()
        if b_strip:
            if "TIDAK ADA KAJIAN" in b_strip:
                continue
            # Context Injection: inject date header context at the top of each block
            if header:
                final_blocks.append(header + "\n\n" + b_strip)
            else:
                final_blocks.append(b_strip)
    return final_blocks

def split_sessions_by_content(text, filename=""):
    """Detect file format based on content, split it into sessions, and handle sub-sessions."""
    text_clean = text.strip()
    fn = filename.lower()
    
    # Priority 1: Filename matches
    if "gresik" in fn:
        raw_sessions = _split_gresik(text_clean)
    elif "kaskus" in fn:
        raw_sessions = _split_kaskus(text_clean)
    elif "13" in fn:
        raw_sessions = _split_sample13_sessions(text_clean)
    elif "03" in fn:
        raw_sessions = _split_batam(text_clean)
    else:
        # Priority 2: Content patterns
        if "》Pemateri" in text or "》Tema" in text:
            raw_sessions = _split_kaskus(text_clean)
        elif "JADWAL KAJIAN SUNNAH KOTA BATAM" in text:
            raw_sessions = _split_batam(text_clean)
        elif text_clean.startswith("⏰") or "\n⏰" in text:
            # Check if Surabaya Mengaji (sample-dataset-13) or Gresik
            if "Surabaya Mengaji" in text or "💳" in text:
                raw_sessions = _split_sample13_sessions(text_clean)
            else:
                raw_sessions = _split_gresik(text_clean)
        else:
            # Fallback: Single session (whole text)
            raw_sessions = [text_clean]
            
    # Sub-split sessions that contain multiple sub-session blocks (e.g. - SESI X or Batam sub-sessions)
    final_sessions = []
    for s in raw_sessions:
        if re.search(r'-\s*SESI\s*\d+', s, flags=re.IGNORECASE):
            parts = re.split(r'(-\s*SESI\s*\d+)', s, flags=re.IGNORECASE)
            header = parts[0]
            for i in range(1, len(parts), 2):
                sesi_title = parts[i]
                sesi_content = parts[i+1]
                combined = header.strip() + "\n" + sesi_title + "\n" + sesi_content.strip()
                final_sessions.append(combined)
        elif len(re.findall(r'(?:Kajian\s+Ba\'da|(?<!Kajian\s)Ba\'da)\s+(?:Subuh|Dzuhur|Ashar|Maghrib|Isya)', s, flags=re.IGNORECASE)) > 1:
            # Batam sub-splitting using lookahead with negative lookbehind
            parts = re.split(r'(?=(?:Kajian\s+Ba\'da|(?<!Kajian\s)Ba\'da)\s+(?:Subuh|Dzuhur|Ashar|Maghrib|Isya))', s, flags=re.IGNORECASE)
            header = parts[0]
            for sub_content in parts[1:]:
                if sub_content.strip():
                    combined = header.strip() + "\n\n" + sub_content.strip()
                    final_sessions.append(combined)
        else:
            final_sessions.append(s)
            
    return final_sessions

# ═════════════════════════════════════════════════════════════
# Entity Structuring & Cleaning
# ═════════════════════════════════════════════════════════════

def extract_address_from_text(text):
    """Extract address lines directly following the location indicator line."""
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    start_idx = -1
    for i, line in enumerate(lines):
        if any(emoji in line for emoji in ['🕌', '🏢', '🏡', '🏠', '📍']) or 'LOKASI:' in line:
            start_idx = i
            break
            
    if start_idx == -1:
        return None
        
    address_lines = []
    stop_indicators = [
        'G-maps', '🌏', '》', '—', '–', '-', 'http', 'https', '☎️', '📲', '🎙️', '📚', '⏰', '👤', '📲', 'LIVE'
    ]
    
    for i in range(start_idx + 1, len(lines)):
        line = lines[i]
        if any(marker in line for marker in stop_indicators) or line.startswith(('*', '👉', '▶️', '☎️', '💡', '🔴', '📢', '🔊')):
            break
        if line == '*':
            break
        address_lines.append(line)
        
    if address_lines:
        return ", ".join(address_lines)
    return None


def structure_session(doc) -> dict[str, Any]:
    """Convert SpaCy doc entities into a structured dictionary."""
    session_data: dict[str, Any] = {
        "sesi": None,
        "pemateri": None,
        "tema": None,
        "waktu": None,
        "lokasi": None,
        "alamat": None,
        "link_maps": None,
        "metode": "offline",
        "kategori": None,
        "kontak": [],
        "link_streaming": [],
        "taradhi": [],
        "bank_transfer": {
            "bank": None,
            "norek": None
        }
    }
    
    # Extract address from text lines as fallback
    fallback_address = extract_address_from_text(doc.text)
    
    # Extract session identifier (e.g. "SESI 1") if present in the text
    sesi_match = re.search(r'-\s*(SESI\s*\d+)', doc.text, flags=re.IGNORECASE)
    if sesi_match:
        session_data["sesi"] = sesi_match.group(1).strip().upper()
        
    pemateris = []
    temas = []
    waktus = []
    lokasis = []
    alamats = []
    kategoris = []
    banks = []
    noreks = []
    link_maps_list = []
    metodes = []
    
    # Specific keywords to filter out false-positive TEMA entities
    disclaimer_keywords = [
        "kami tim jadwal",
        "bukan sebagai pihak penyelenggara",
        "mohon cek kembali info rekapan",
        "hanya memberikan informasi"
    ]
    
    for ent in doc.ents:
        val = ent.text.strip()
        label = ent.label_
        
        # Post-processing: Filter out false-positive TEMA (like disclaimers)
        if label == "TEMA":
            val_lower = val.lower()
            if any(kw in val_lower for kw in disclaimer_keywords):
                continue
                
        if label == "PEMATERI":
            pemateris.append(val)
        elif label == "TEMA":
            temas.append(val)
        elif label == "WAKTU":
            waktus.append(val)
        elif label == "LOKASI":
            # Jika lokasi terdeteksi online platform, masukkan ke METODE
            if val.upper() in ["ZOOM", "GOOGLE MEET", "MEET", "YOUTUBE"]:
                metodes.append(val)
            else:
                lokasis.append(val)
        elif label == "KATEGORI" or label == "STATUS":
            kategoris.append(val)
        elif label == "KONTAK":
            # Clean up and split multiple contact numbers in the same string
            parts = re.split(r'[\s/]+', val)
            for part in parts:
                cleaned_kontak = re.sub(r'[^\d\+\-]', '', part).strip()
                if len(re.sub(r'[^\d]', '', cleaned_kontak)) >= 9:
                    if cleaned_kontak not in session_data["kontak"]:
                        session_data["kontak"].append(cleaned_kontak)
        elif label == "LINK_STREAMING":
            if val not in session_data["link_streaming"]:
                session_data["link_streaming"].append(val)
        elif label == "LINK_MAPS":
            # Validate that it is indeed a URL or maps link to avoid statistical leaks
            if any(k in val.lower() for k in ["http", "https", "maps", "goo.gl", "bit.ly"]):
                link_maps_list.append(val)
        elif label == "METODE":
            metodes.append(val)
        elif label == "TARADHI":
            if val not in session_data["taradhi"]:
                session_data["taradhi"].append(val)
        elif label == "ALAMAT":
            alamats.append(val)
        elif label == "BANK":
            banks.append(val)
        elif label == "NOREK":
            # Clean up non-digit numbers, limit to first line to prevent leakage to contacts below
            first_line = val.split('\n')[0].strip()
            cleaned_norek = re.sub(r'[^\d]', '', first_line)
            if cleaned_norek:
                noreks.append(cleaned_norek)
            
    # Resolve multiple values to clean strings
    if pemateris:
        session_data["pemateri"] = " / ".join(pemateris) if len(pemateris) > 1 else pemateris[0]
    if temas:
        session_data["tema"] = " / ".join(temas) if len(temas) > 1 else temas[0]
    if waktus:
        session_data["waktu"] = " / ".join(waktus) if len(waktus) > 1 else waktus[0]
    if lokasis:
        session_data["lokasi"] = " / ".join(lokasis) if len(lokasis) > 1 else lokasis[0]
    if alamats:
        session_data["alamat"] = " / ".join(alamats) if len(alamats) > 1 else alamats[0]
    else:
        session_data["alamat"] = fallback_address
        
    if link_maps_list:
        session_data["link_maps"] = link_maps_list[0]
    if metodes:
        session_data["metode"] = metodes[0]
        
    if kategoris:
        session_data["kategori"] = " / ".join(kategoris) if len(kategoris) > 1 else kategoris[0]
        
    if banks:
        session_data["bank_transfer"]["bank"] = banks[0]
    if noreks:
        session_data["bank_transfer"]["norek"] = noreks[0]
        
    # ── Emoji-based Fallbacks ──
    # If key fields are still missing due to statistical model fluctuation,
    # extract them directly using structural emojis.
    def extract_by_emoji(emoji_list):
        for emoji in emoji_list:
            m = re.search(fr'^{emoji}\ufe0f?\s*(.+)$', doc.text, re.MULTILINE)
            if m:
                val = m.group(1).strip()
                return re.sub(r'[\*_]', '', val).strip()
        return None

    if not session_data["tema"]:
        session_data["tema"] = extract_by_emoji(['📚', '📖', '📜', '📗', '📝'])
        
    if not session_data["pemateri"]:
        pemateri_val = extract_by_emoji(['👤', '🎙️', '🎙', '🩺'])
        if pemateri_val:
            taradhi_words = ["hafidzahullah", "hafizhahullah", "hafidzahullahu", "hafizhahullahu", "ta'ala", "حفظه", "الله", "shallallahu", "alaihi", "wasallam"]
            clean_p = pemateri_val.split(',')[0].strip()
            for tw in taradhi_words:
                clean_p = re.sub(fr'\s*{re.escape(tw)}\s*', '', clean_p, flags=re.IGNORECASE)
            session_data["pemateri"] = clean_p.strip(", ")
            
    if not session_data["waktu"]:
        session_data["waktu"] = extract_by_emoji(['⏰', '🕰️', '🕰', '📅', '📆', '🕗'])
        
    if not session_data["lokasi"]:
        session_data["lokasi"] = extract_by_emoji(['📍', '🕌', '🏢', '🏡', '🏠'])
        
    if not session_data["kategori"]:
        session_data["kategori"] = extract_by_emoji(['🚨'])
        
    # Fallback to extract link maps from text if link_maps is None
    if not session_data["link_maps"]:
        maps_match = re.search(r'(https?://(?:maps\.google\.com|goo\.gl/maps|maps\.app\.goo\.gl)\S*)', doc.text)
        if maps_match:
            session_data["link_maps"] = maps_match.group(1)
        else:
            bitly_match = re.search(r'(http://bit\.ly/(?:ahmaddahlangresik|AtTauhid_Betiting|AlJihad_Cerme|PSofyan_Driyorejo))', doc.text)
            if bitly_match:
                session_data["link_maps"] = bitly_match.group(1)
                
    # Fallback contact numbers based on phone/whatsapp emojis
    if not session_data["kontak"]:
        contact_matches = re.findall(fr'^(?:📱|📞|☎️|📲)\ufe0f?\s*(.+)$', doc.text, re.MULTILINE)
        for val in contact_matches:
            parts = re.split(r'[\s/]+', val)
            for part in parts:
                cleaned = re.sub(r'[^\d\+\-]', '', part).strip()
                if len(re.sub(r'[^\d]', '', cleaned)) >= 9:
                    if cleaned not in session_data["kontak"]:
                        session_data["kontak"].append(cleaned)
                        
    # Fallback bank transfer info (bank & norek)
    if not session_data["bank_transfer"]["bank"]:
        bank_match = re.search(r'\b(BSI|BANK\s+SYARIAH\s+INDONESIA|MANDIRI|BCA|BRI|BANK\s+MUAMALAT|MUAMALAT)\b', doc.text, re.IGNORECASE)
        if bank_match:
            session_data["bank_transfer"]["bank"] = bank_match.group(1).upper()
            
    if not session_data["bank_transfer"]["norek"] or len(session_data["bank_transfer"]["norek"]) > 25:
        norek_match = re.search(r'(?:No\.\s*Rek\s*:\s*|No\s*rekening\s*:\s*|BSI\s*-\s*)([\d\s]+)', doc.text, re.IGNORECASE)
        if norek_match:
            raw_norek = norek_match.group(1).split('\n')[0].strip()
            cleaned_norek = re.sub(r'[^\d]', '', raw_norek)
            if cleaned_norek:
                session_data["bank_transfer"]["norek"] = cleaned_norek
                
    # Fallback online detection
    if session_data["metode"] == "offline":
        if any(kw in doc.text.upper() for kw in ["ZOOM", "GOOGLE MEET", "MEET", "YOUTUBE LIVE", "LIVE YOUTUBE"]):
            for kw in ["ZOOM", "GOOGLE MEET", "MEET", "YOUTUBE"]:
                if kw in doc.text.upper():
                    session_data["metode"] = kw
                    break
            if session_data["metode"] == "offline":
                session_data["metode"] = "online"
                
    return session_data

# ═════════════════════════════════════════════════════════════
# Main Pipeline
# ═════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="AI Parser Jadwal Kajian (SpaCy)")
    parser.add_argument("input_file", help="Path ke berkas teks rekap input (.txt)")
    parser.add_argument("output_file", nargs="?", default=None, help="Path ke berkas output JSON (default: output/sampling/..._parsed.json)")
    parser.add_argument("--use-rel", action="store_true", help="Gunakan model Joint NER + Relation Extraction (default: model NER)")
    
    args = parser.parse_args()
    input_path = args.input_file
    output_path = args.output_file
    use_rel = args.use_rel
    
    if output_path is None:
        # Generate default name under output-sample/ directory
        out_dir = "output/sampling"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(out_dir, f"{base_name}_output_parsed.json")
        
        # If the file already exists, append _01.json, _02.json, etc.
        if os.path.exists(output_path):
            counter = 1
            while True:
                candidate_name = f"{base_name}_output_parsed_{counter:02d}.json"
                candidate_path = os.path.join(out_dir, candidate_name)
                if not os.path.exists(candidate_path):
                    output_path = candidate_path
                    break
                counter += 1
    
    def log(msg):
        if output_path == "-":
            print(msg, file=sys.stderr)
        else:
            print(msg)
            
    if not os.path.exists(input_path):
        log(f"Error: Input file '{input_path}' not found.")
        sys.exit(1)
        
    if use_rel:
        model_path = "output/model_rel/model-best"
    else:
        model_path = "output/model_ner/model-best"
        
    if not os.path.exists(model_path):
        log(f"Error: Model not found at '{model_path}'. Please train the model first.")
        sys.exit(1)
        
    log(f"🤖 Loading SpaCy model from: {model_path} ...")
    
    # Coba impor custom component Relation Extraction secara dinamis
    # agar registry SpaCy mengenali 'relation_extractor' jika model adalah Joint NER + RE.
    try:
        src_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(src_dir)
        training_dir = os.path.join(project_root, "training")
        if training_dir not in sys.path:
            sys.path.insert(0, training_dir)
        import rel_component
        log("🧩 Registered custom 'relation_extractor' component from training/rel_component.py")
    except Exception as e:
        log(f"ℹ️ Custom relation_extractor not registered (NER-only fallback): {e}")

    nlp = spacy.load(model_path)
    
    log(f"📄 Reading {input_path} ...")
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Step 1: Split rekap into session blocks
    sessions = split_sessions_by_content(content, os.path.basename(input_path))
    log(f"✂️  Split document into {len(sessions)} session(s).")
    
    # Step 2: Parse each session block
    parsed_sessions = []
    for idx, session_text in enumerate(sessions):
        doc = nlp(session_text)
        structured = structure_session(doc)
        
        # Only add if we found at least some meaningful information
        if any([structured["pemateri"], structured["tema"], structured["lokasi"]]):
            parsed_sessions.append(structured)
            
    # Step 3: Export to JSON
    if output_path == "-":
        print(json.dumps(parsed_sessions, indent=2, ensure_ascii=False))
    else:
        log(f"💾 Saving {len(parsed_sessions)} parsed sessions to {output_path} ...")
        with open(output_path, "w", encoding="utf-8") as out_f:
            json.dump(parsed_sessions, out_f, indent=2, ensure_ascii=False)
        log("✅ Completed successfully!")

if __name__ == "__main__":
    main()
