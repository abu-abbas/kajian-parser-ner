#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_perfect_dataset.py
═══════════════════════════
Reads files from input/sampling/text-base/, parses them per session, 
applies robust and curated regexes, cleans the entity boundaries 
dynamically (trimming control characters, newlines, and symbols), 
and exports the dataset directly to training/data_latihan_spacy.py.

Introduces:
  - KATEGORI (replacing STATUS)
  - LINK_MAPS (specifically for maps links, distinct from LINK_STREAMING)
  - METODE (for online platform names like ZOOM, Live YouTube, etc.)
"""

import os
import re
import glob

def clean_entity_text(text, start, end, label):
    """
    Cleans the extracted entity text (removes trailing/leading newlines, whitespace,
    and markdown indicators like *, _, and emojis) and returns adjusted (start, end).
    """
    val = text[start:end]
    
    # Trim leading/trailing whitespace
    ls = len(val) - len(val.lstrip())
    rs = len(val) - len(val.rstrip())
    start += ls
    end -= rs
    val = text[start:end]
    
    # Loop to strip typical formatting and control symbols from the boundaries
    strip_chars = "*_~,;.\n\r⏰📚🎙️🕌🏠🏡🏢🚨📱🖇️👤🕰️📆🕗📍▶️👉"
    while True:
        changed = False
        val_strip = val.strip(strip_chars).strip()
        if len(val_strip) < len(val):
            # Adjust indices
            ls_diff = val.find(val_strip)
            if ls_diff > 0:
                start += ls_diff
            end = start + len(val_strip)
            val = text[start:end]
            changed = True
        
        # If there is a trailing or leading newline or space inside
        if val.startswith('\n') or val.endswith('\n'):
            val = val.strip('\n')
            start = text.find(val, start)
            end = start + len(val)
            changed = True
            
        if not changed:
            break
            
    return start, end, val

def extract_entities_from_session(session_text, filename):
    """
    Applies curated patterns to extract entities from session text.
    Handles KATEGORI, LINK_MAPS, and METODE.
    """
    entities = []
    
    # 1. Match regex patterns
    patterns = [
        # WAKTU
        (r'⏰\s*(.+)$', 'WAKTU'),
        (r'🕰️?\s*(.+)$', 'WAKTU'),
        (r'📅\s*(.+)$', 'WAKTU'),
        (r'📆\s*(.+)$', 'WAKTU'),
        (r'🕗\s*(.+)$', 'WAKTU'),
        (r'》Waktu\s*:\s*(.+)$', 'WAKTU'),
        (r'Hari\s+Jum\'at,\s+\d+\s+Desember\s+\d+.*$', 'WAKTU'),
        (r'Pukul\s*\d+\.\d+\s*WIB\s*s/d\s*selesai', 'WAKTU'),
        
        # TEMA (Exclude introductory text)
        (r'📚\s*(.+)$', 'TEMA'),
        (r'📖\s*(.+)$', 'TEMA'),
        (r'📜\s*(.+)$', 'TEMA'),
        (r'📗\s*(.+)$', 'TEMA'),
        (r'📝\s*(.+)$', 'TEMA'),
        (r'》Tema\s*:\s*(.+(?:\n(?!》|\*|🕌|🏢|🏡|🌏|- ).+)*)', 'TEMA'),
        (r'TEMA\s*:\s*(.+)$', 'TEMA'),
        (r'Kitab\s+Tauhid', 'TEMA'),
        (r'Tazkiyatun\s+Nafs', 'TEMA'),
        (r'^([A-Z\s]{10,})$', 'TEMA'),
        
        # PEMATERI
        (r'(?:🎙️)?\s*(Ustadz\s+[A-Za-z\s\.,]+|Ustadzah\s+[A-Za-z\s\.,]+)', 'PEMATERI'),
        (r'👤\s*(Ustadz\s+[A-Za-z\s\.,]+|Ustadzah\s+[A-Za-z\s\.,]+)', 'PEMATERI'),
        (r'Nara Sumber:\s*(.+)$', 'PEMATERI'),
        (r'》Pemateri\s*:\s*(.+)$', 'PEMATERI'),
        (r'Bersama\s+(Mu’allim\s+Tartil\s+Academy\s+-\s+QITA\s+Ikhwan)', 'PEMATERI'),
        
        # LOKASI
        (r'🕌\s*(.+)$', 'LOKASI'),
        (r'🏠\s*(.+)$', 'LOKASI'),
        (r'🏡\s*(.+)$', 'LOKASI'),
        (r'🏢\s*(.+)$', 'LOKASI'),
        (r'Disiarkan\s+langsung\s+dari\s*:\s*\n(.+)', 'LOKASI'),
        
        # ALAMAT
        (r'^\s+(Jl\.\s+.+)$', 'ALAMAT'),
        (r'^\s+(Gg\.\s+.+)$', 'ALAMAT'),
        (r'^\s+(Personal\s+Meeting\s+Room)$', 'ALAMAT'),
        
        # KATEGORI (formerly STATUS)
        (r'🚨\s*(.+)$', 'KATEGORI'),
        (r'(TERBUKA UNTUK UMUM)', 'KATEGORI'),
        (r'\(Terbuka untuk umum[^)]*\)', 'KATEGORI'),
        (r'(Umum\s*:\s*Ikhwan\s*-\s*Akhwat)', 'KATEGORI'),
        (r'(Khusus\s*:\s*Akhwat\s*&\s*Anak-anak)', 'KATEGORI'),
        (r'(Khusus\s*:\s*Akhwat)', 'KATEGORI'),
        (r'(Khusus\s*:\s*Ikhwan)', 'KATEGORI'),
        (r'(Khusus\s*:\s*Akhwat\s*&\s*Remaja)', 'KATEGORI'),
        (r'(Umum\s*:\s*Ikhwan\s*-\s*AkAkhwat)', 'KATEGORI'),
        
        # CONTACT
        (r'》CP\s*:\s*(.+)$', 'KONTAK'),
        (r'📱\s*(\d[\d\s-]+\d)', 'KONTAK'),
        (r'via wa\s+(\d[\d\s-]+\d)', 'KONTAK'),
        (r'NARAHUBUNG:\s*(\d[\d\s-]+\d)', 'KONTAK'),
        (r'No\.\s*Rek\s*:\s*([\d\s]+)', 'NOREK'),
        (r'No rekening:\s*(\d+)', 'NOREK'),
        (r'BSI\s*-\s*(\d+)', 'NOREK'),
        
        # BANK
        (r'(BANK\s+[A-Z]+)', 'BANK'),
        
        # LINK_MAPS
        (r'(https?://(?:maps\.google\.com|goo\.gl/maps|maps\.app\.goo\.gl)\S*)', 'LINK_MAPS'),
        (r'(http://bit\.ly/(?:ahmaddahlangresik|AtTauhid_Betiting|AlJihad_Cerme|PSofyan_Driyorejo))', 'LINK_MAPS'),
        (r'🌏\s*G-maps\s*:\s*(\S+)', 'LINK_MAPS'),
        
        # LINK_STREAMING
        (r'(https?://(?:www\.)?youtube\.com\S*)', 'LINK_STREAMING'),
        (r'(https?://(?:www\.)?facebook\.com\S*)', 'LINK_STREAMING'),
        (r'(https?://(?:www\.)?instagram\.com\S*)', 'LINK_STREAMING'),
        (r'(https?://t\.me\S*)', 'LINK_STREAMING'),
        (r'(https?://whatsapp\.com/channel\S*)', 'LINK_STREAMING'),
        (r'(https?://bit\.ly/ashiilapp\S*)', 'LINK_STREAMING'),
        
        # METODE
        (r'\b(𝗭𝗢𝗢𝗠|ZOOM|Zoom|Google Meet|GoogleMeet|Meet|YouTube Live|LIVE YouTube|Live Streaming|AshiilTV|Televisi Streaming)\b', 'METODE')
    ]
    
    # Taradhi
    taradhi_patterns = [
        r"Hafi(?:d|z|dz|zh)ahullah(?:u(?:\s+Ta'ala)?)?",
        r"حفظ[هة]\s+الله(?:\s+تعal[ىي])?",
        r"shallallahu\s+[^a-zA-Z\s]{0,3}\s*alaihi\s+wa\s*sallam",
        r"shallallahu\s+[^a-zA-Z\s]{0,3}\s*alaihi\s+wasallam",
        r"Shallallahu\s+‘alaihi\s+wa\s+sallam",
    ]
    for pat in taradhi_patterns:
        for m in re.finditer(pat, session_text, re.IGNORECASE):
            entities.append((m.start(), m.end(), 'TARADHI'))

    for regex, label in patterns:
        for m in re.finditer(regex, session_text, re.MULTILINE):
            grp = 1 if m.lastindex and m.lastindex >= 1 else 0
            s, e = m.start(grp), m.end(grp)
            entities.append((s, e, label))
            
    # Post-process and clean boundaries
    cleaned_entities = []
    for s, e, label in entities:
        if 0 <= s < e <= len(session_text):
            s_c, e_c, val = clean_entity_text(session_text, s, e, label)
            
            # Post-cleaning validations:
            # 1. Skip if the entity is empty
            if not val or not val.strip():
                continue
            # 2. Skip false positive TEMA
            if label == 'TEMA' and ("Insya Allah" in val or "Untuk Informasi" in val or "Kajian rutin" in val or "Kajian Umum" in val):
                continue
            # 3. Skip false positive LOKASI that are actually Zoom/Online platform (they will be caught by METODE)
            if label == 'LOKASI' and val in ["𝗭𝗢𝗢𝗠", "Zoom", "Zoom Meeting"]:
                continue
            # 4. Skip LINK_MAPS that are actually other social media links
            if label == 'LINK_MAPS' and not any(k in val for k in ["maps", "goo.gl", "bit.ly/ahmaddahlangresik", "bit.ly/AtTauhid", "bit.ly/AlJihad", "bit.ly/PSofyan"]):
                continue
                
            cleaned_entities.append((s_c, e_c, label))
            
    # Resolve overlapping entities (keep longer span, prioritize certain labels)
    cleaned_entities.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    final_entities = []
    last_end = -1
    for s, e, label in cleaned_entities:
        if s >= last_end:
            final_entities.append((s, e, label))
            last_end = e
            
    return final_entities

def split_text_to_sessions(text, filename):
    """Splits document text into logical session units."""
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

def main():
    print("🚀 Generating Perfect Dataset for SpaCy...")
    input_folder = "input/sampling/text-base"
    files = sorted(glob.glob(os.path.join(input_folder, "*.txt")))
    
    all_data = []
    
    for fpath in files:
        fname = os.path.basename(fpath)
        # Skip empty files
        if fname == "sample-dataset-03.txt":
            continue
            
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()
            # Clean surrogate characters (0xD800 - 0xDFFF) to prevent SpaCy UnicodeEncodeError
            content = "".join(c for c in content if not (0xD800 <= ord(c) <= 0xDFFF))
            
        sessions = split_text_to_sessions(content, fname)
        print(f"  📄 Processing {fname} ({len(sessions)} session blocks)...")
        
        for i, session in enumerate(sessions):
            entities = extract_entities_from_session(session, fname)
            if entities:
                all_data.append((session, entities))
                
    # Generate synthetic variations for comma-separated LOKASI and ALAMAT
    # to enrich spatial understanding
    for session, entities in list(all_data):
        if "Masjid al-Ikhlash" in session and ", Jl." in session:
            variations = [
                ("Masjid An-Nur", "Jl. Diponegoro No. 12, Kel. Tegalsari, Kec. Tegalsari, Surabaya"),
                ("Masjid Al-Barkah", "Jl. Slamet Riyadi No. 5, Laweyan, Surakarta"),
                ("Musholla At-Taqwa", "Jl. Pemuda No. 45, Sekayu, Kec. Semarang Tengah, Kota Semarang"),
                ("Masjid Istiqlal", "Jl. Taman Wijaya Kusuma, Pasar Baru, Kec. Sawah Besar, Kota Jakarta Pusat"),
            ]
            for loc_name, addr_name in variations:
                old_loc = "Masjid al-Ikhlash - Delta Sari Indah"
                old_addr = "Jl. Anggrek VI no.36-38, Kureksari, Waru, Sidoarjo"
                new_session = session.replace(old_loc, loc_name).replace(old_addr, addr_name)
                new_entities = extract_entities_from_session(new_session, "synthetic")
                if new_entities:
                    all_data.append((new_session, new_entities))
            break
            
    # Export to training/data_latihan_spacy.py
    output_path = "training/data_latihan_spacy.py"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write('# -*- coding: utf-8 -*-\n')
        f.write('"""\n')
        f.write('data_latihan_spacy.py\n')
        f.write('═════════════════════\n')
        f.write('Auto-generated by generate_perfect_dataset.py\n')
        f.write('SpaCy NER training data for Jadwal Kajian parsing.\n')
        f.write('\n')
        f.write('Entity Labels:\n')
        f.write('  Wajib  : PEMATERI, LOKASI, WAKTU, TEMA, KATEGORI\n')
        f.write('  Dinamis: BANK, NOREK, KONTAK, LINK_STREAMING, LINK_MAPS, METODE, TARADHI\n')
        f.write('"""\n\n')
        f.write('TRAIN_DATA = [\n')
        
        for i, (text, ents) in enumerate(all_data):
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
        
    print(f"📊 Completed! Total training examples: {len(all_data)}")
    print(f"💾 Written to {output_path}")

if __name__ == "__main__":
    main()
