#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parse_to_json.py
════════════════
Parses a rekap file (e.g. Kaskus, Gresik, or Single Session) into a structured JSON file.
Uses the trained SpaCy model to extract entities and groups them per session.
"""

import sys
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
    else:
        # Priority 2: Content patterns
        if "》Pemateri" in text or "》Tema" in text:
            raw_sessions = _split_kaskus(text_clean)
        elif text_clean.startswith("⏰") or "\n⏰" in text:
            # Check if Surabaya Mengaji (sample-dataset-13) or Gresik
            if "Surabaya Mengaji" in text or "💳" in text:
                raw_sessions = _split_sample13_sessions(text_clean)
            else:
                raw_sessions = _split_gresik(text_clean)
        else:
            # Fallback: Single session (whole text)
            raw_sessions = [text_clean]
            
    # Sub-split sessions that contain multiple "- SESI X" blocks
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
        "status": None,
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
    statuses = []
    banks = []
    noreks = []
    
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
            lokasis.append(val)
        elif label == "STATUS":
            statuses.append(val)
        elif label == "KONTAK":
            # Clean up trailing/leading symbols or whitespaces from contact numbers
            cleaned_kontak = re.sub(r'[^\d\s\-\+/a-zA-Z]', '', val).strip()
            if cleaned_kontak and cleaned_kontak not in session_data["kontak"]:
                session_data["kontak"].append(cleaned_kontak)
        elif label == "LINK_STREAMING":
            if val not in session_data["link_streaming"]:
                session_data["link_streaming"].append(val)
        elif label == "TARADHI":
            if val not in session_data["taradhi"]:
                session_data["taradhi"].append(val)
        elif label == "ALAMAT":
            alamats.append(val)
        elif label == "BANK":
            banks.append(val)
        elif label == "NOREK":
            # Clean up non-digit numbers
            cleaned_norek = re.sub(r'[^\d]', '', val)
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
    if statuses:
        session_data["status"] = " / ".join(statuses) if len(statuses) > 1 else statuses[0]
        
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
        

            
    return session_data

# ═════════════════════════════════════════════════════════════
# Main Pipeline
# ═════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_to_json.py <input_text_file> [output_json_file]")
        sys.exit(1)
        
    input_path = sys.argv[1]
    
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    else:
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
        
    model_path = "output/model/model-best"
    if not os.path.exists(model_path):
        log(f"Error: Model not found at '{model_path}'. Please train the model first.")
        sys.exit(1)
        
    log(f"🤖 Loading SpaCy model from: {model_path} ...")
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
