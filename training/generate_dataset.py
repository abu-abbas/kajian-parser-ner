#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_dataset.py
═══════════════════
Workflow /build-dataset — Steps 1–4
Processes text files from data-sample/text-base/ into SpaCy NER TRAIN_DATA.

Rules (@aturan-kurasi-kajian):
  1. Kebal Typo: Keep OCR/text typos as-is (e.g. 'U5tadz', 'AkAkhwat')
  2. NER Dinamis:
     - Wajib:  PEMATERI, LOKASI, WAKTU, TEMA, STATUS
     - Dinamis: BANK, NOREK, KONTAK, LINK_STREAMING, FASILITAS
  3. Presisi Index: 0-indexed character start/end, no overlapping

Usage:
    source venv/bin/activate
    python training/generate_dataset.py
"""

import re
import os
import glob
import sys


# ═════════════════════════════════════════════════════════════
# Step 1: Ingestion — Read all .txt files
# ═════════════════════════════════════════════════════════════

def read_all_files(folder="data-sample/text-base"):
    """Read all non-empty .txt files from the folder."""
    files = sorted(glob.glob(os.path.join(folder, "*.txt")))
    result = {}
    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()
        if content.strip():  # Skip empty files (e.g. sample-dataset-03.txt)
            result[os.path.basename(fpath)] = content
    return result


# ═════════════════════════════════════════════════════════════
# Step 2a: Session Splitting
# ═════════════════════════════════════════════════════════════

def split_sessions(text, filename):
    """Split text into individual session blocks based on file format."""
    fn = filename.lower()
    if "gresik" in fn:
        return _split_gresik(text)
    elif "kaskus" in fn:
        return _split_kaskus(text)
    elif "13" in fn:
        return _split_sample13_sessions(text)
    else:
        # Single-session files (sample-dataset-01, sample-dataset-02)
        return [text]


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


# ═════════════════════════════════════════════════════════════
# Step 2b: NER Tagging — Entity Extraction
# ═════════════════════════════════════════════════════════════

def extract_entities(text, filename):
    """Extract NER entities from session text using format-specific patterns."""
    fn = filename.lower()
    if "gresik" in fn:
        entities = _extract_gresik(text)
    elif "kaskus" in fn:
        entities = _extract_kaskus(text)
    elif "01" in filename:
        entities = _extract_sample01(text)
    elif "02" in filename:
        entities = _extract_sample02(text)
    elif "04" in filename:
        entities = _extract_sample04(text)
    elif "05" in filename:
        entities = _extract_sample05(text)
    elif "06" in filename:
        entities = _extract_sample06(text)
    elif "07" in filename:
        entities = _extract_sample07(text)
    elif "08" in filename:
        entities = _extract_sample08(text)
    elif "09" in filename:
        entities = _extract_sample09(text)
    elif "10" in filename:
        entities = _extract_sample10(text)
    elif "11" in filename:
        entities = _extract_sample11(text)
    elif "12" in filename:
        entities = _extract_sample12(text)
    elif "13" in filename:
        entities = _extract_sample13(text)
    else:
        entities = []

    # Extract TARADHI globally from the text
    taradhi_patterns = [
        r"Hafi(?:d|z|dz|zh)ahullah(?:u(?:\s+Ta'ala)?)?",
        r"حفظ[هة]\s+الله(?:\s+تعal[ىي])?",
        r"shallallahu\s+[^a-zA-Z\s]{0,3}\s*alaihi\s+wa\s*sallam",
        r"shallallahu\s+[^a-zA-Z\s]{0,3}\s*alaihi\s+wasallam",
    ]
    
    taradhi_found = []
    for pat in taradhi_patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            s, e = m.start(), m.end()
            taradhi_found.append((s, e, 'TARADHI'))

    # Post-processing: Shrink PEMATERI boundary if it swallows trailing TARADHI words
    cleaned_entities = []
    taradhi_words = {"hafidzahullah", "hafizhahullah", "hafidzahullahu", "hafizhahullahu", "ta'ala", "حفظه", "الله", "shallallahu", "alaihi", "wasallam"}
    
    for s, e, label in entities:
        if label == 'PEMATERI':
            val = text[s:e]
            while True:
                m_last = re.search(r'(\s*\S+)\s*$', val)
                if not m_last:
                    break
                last_word = m_last.group(1).strip()
                last_word_clean = last_word.lower().strip(",._()[] ")
                if last_word_clean in taradhi_words or not any(c.isalnum() for c in last_word_clean):
                    val = val[:m_last.start(1)]
                else:
                    break
            val_clean = val.rstrip(",_()[] ")
            e = s + len(val_clean)
        if s < e:
            cleaned_entities.append((s, e, label))

    # Append all found TARADHI entities
    cleaned_entities.extend(taradhi_found)
    
    return cleaned_entities




def _apply_patterns(text, patterns):
    """Apply a list of (regex, label) patterns to text.

    Each regex should have exactly one capturing group (group 1).
    Returns list of (start, end, label) tuples with trimmed whitespace.
    """
    entities = []
    for pat, label in patterns:
        for m in re.finditer(pat, text, re.MULTILINE):
            grp = 1 if m.lastindex and m.lastindex >= 1 else 0
            s, e = m.start(grp), m.end(grp)
            val = text[s:e]
            # Trim leading/trailing whitespace, adjusting indices
            ls = len(val) - len(val.lstrip())
            rs = len(val) - len(val.rstrip())
            s += ls
            if rs > 0:
                e -= rs
            if s < e:
                entities.append((s, e, label))
    return entities


# ── Gresik ──
def _extract_gresik(text):
    """Extract entities from a Gresik session block.

    Format:
        ⏰  [time]
        📚  [topic]
        🎙️  [speaker]         (🎙 with or without variation selector)
        🕌/🏠  [location]
        🗺️  [map url]
        🚨  [audience status]
        📱  [contact]          (optional)
    """
    pats = [
        (r'^⏰\s+(.+)$',               'WAKTU'),
        (r'^📚\s+(.+)$',               'TEMA'),
        (r'^🎙\ufe0f?\s+(.+)$',        'PEMATERI'),
        (r'^(?:🕌|🏠)\s+(.+)$',        'LOKASI'),
        # ALAMAT (indented address lines under location icon)
        (r'^(?:🕌|🏠)\s+.+?\n\s+([A-Za-z\d\s\.,]+?)(?=\n\s*(?:🗺️|🚨|📱|⏰|📚|🎙))', 'ALAMAT'),
        (r'^🚨\s+(.+)$',               'STATUS'),
        (r'^📱\s+(.+)$',               'KONTAK'),
    ]
    return _apply_patterns(text, pats)


# ── Kaskus ──
def _extract_kaskus(text):
    """Extract entities from a Kaskus session block.

    Format:
        Possibility of:
        🕌/🏢/🏡 [location name]
        [address lines]
        🌏 G-maps : [url]
        [optional: - SESI 1 / - SESI 2]
        》Pemateri : [speaker]
        》Tema : [topic]           (may have continuation lines)
        》Waktu : [time]
        》CP : [contact]
    """
    pats = [
        # Location name (first line only)
        (r'^(?:🕌|🏢|🏡)\s*(.+)$',     'LOKASI'),
        # Address lines (captures lines between location and g-maps/other markers)
        (r'^(?:🕌|🏢|🏡)[\s\S]+?\n\s*([\s\S]+?)(?=\n\s*(?:🌏 G-maps|》|–|- SESI|\*))', 'ALAMAT'),
        # Speaker(s)
        (r'》Pemateri\s*:\s*(.+)$',      'PEMATERI'),
        # Topic — may span continuation lines (lines not starting with markers)
        (r'》Tema\s*:\s*(.+(?:\n(?!》|\*|🕌|🏢|🏡|🌏|- ).+)*)', 'TEMA'),
        # Time
        (r'》Waktu\s*:\s*(.+)$',         'WAKTU'),
        # Contact Person
        (r'》CP\s*:\s*(.+)$',            'KONTAK'),
    ]
    return _apply_patterns(text, pats)


# ── Sample 01 ──
def _extract_sample01(text):
    """Extract entities from sample-dataset-01.

    Single-session WhatsApp-style poster with emoji markers and *bold* formatting.
    Rich in dynamic entities (BANK, NOREK, LINK_STREAMING).
    """
    pats = [
        # PEMATERI — after "Nara Sumber:"
        (r'Nara Sumber:\s*(.+)$',                          'PEMATERI'),
        # WAKTU — date between asterisks on 📅 line
        (r'📅\s*\*(.+?)\*',                                'WAKTU'),
        # WAKTU — time range between asterisks on 🕰 line
        (r'🕰\ufe0f?\s*\*(.+?)\*',                         'WAKTU'),
        # TEMA — kitab name between asterisks on 📚 line
        (r'📚\s*\*(.+?)\*',                                'TEMA'),
        # TEMA — specific topic after "TEMA :"
        (r'TEMA\s*:\s*(.+?)\*',                            'TEMA'),
        # LOKASI — after "LOKASI:"
        (r'LOKASI:\s*(.+?)\*',                             'LOKASI'),
        # STATUS
        (r'(TERBUKA UNTUK UMUM)',                          'STATUS'),
        # BANK
        (r'\*\s*(BANK MUAMALAT)',                          'BANK'),
        # NOREK
        (r'No rekening:\s*(\d+)',                          'NOREK'),
        # KONTAK — phone after "via wa"
        (r'via wa\s+([\d][\d\s-]+[\d])',                   'KONTAK'),
        # KONTAK — phone after "NARAHUBUNG:"
        (r'NARAHUBUNG:\s*([\d][\d\s-]+[\d])',              'KONTAK'),
        # LINK_STREAMING — social media URLs
        (r'(https?://youtube\.com\S+)',                    'LINK_STREAMING'),
        (r'(https?://www\.facebook\.com\S+)',              'LINK_STREAMING'),
        (r'(https?://www\.instagram\.com\S+)',             'LINK_STREAMING'),
    ]
    return _apply_patterns(text, pats)


# ── Sample 02 ──
def _extract_sample02(text):
    """Extract entities from sample-dataset-02.

    Single-session with ┃ separator headers, _italic_ markers,
    Arabic text, and multi-platform live streaming links.
    """
    entities = []

    # PEMATERI — name on line after "Pemateri" header, before Arabic honorific
    m = re.search(r'Pemateri\s*:\*\s*\n_(.+?)\s*حفظه', text)
    if m:
        entities.append((m.start(1), m.end(1), 'PEMATERI'))

    # TEMA — quoted Indonesian translation after "Materi" header
    m = re.search(r'Materi\s*:\*.+?_"(.+?)"', text, re.DOTALL)
    if m:
        entities.append((m.start(1), m.end(1), 'TEMA'))

    # WAKTU — date (line after "Hari & Tanggal" header)
    m = re.search(r'Hari & Tanggal\s*:\*\s*\n_(.+?)_\s*$', text, re.MULTILINE)
    if m:
        entities.append((m.start(1), m.end(1), 'WAKTU'))

    # WAKTU — time range (line after "┃Waktu" header)
    m = re.search(r'┃Waktu\s*:\*\s*\n_(.+?)_\s*$', text, re.MULTILINE)
    if m:
        entities.append((m.start(1), m.end(1), 'WAKTU'))

    # LOKASI & ALAMAT — under Tempat
    m = re.search(r'Tempat\s*:\*\s*\n_(.+?)_\s*\n_(.+?)_(?:\s*\n_(.+?)_)?(?=\s*\n_https?://)', text)
    if m:
        entities.append((m.start(1), m.end(1), 'LOKASI'))
        entities.append((m.start(2), m.end(2), 'ALAMAT'))
        if m.group(3):
            entities.append((m.start(3), m.end(3), 'ALAMAT'))

    # KONTAK — wa.me contact
    m = re.search(r'Contact Person\s*:\*\s*\n_(wa\.me/\d+)', text)
    if m:
        entities.append((m.start(1), m.end(1), 'KONTAK'))

    # LINK_STREAMING — URLs between underscores for known platforms
    for m in re.finditer(r'_(https?://[^_\s]+)', text):
        url = m.group(1)
        if any(d in url for d in [
            'solomengaji', 'youtube', 'facebook',
            'instagram', 't.me', 'whatsapp'
        ]):
            entities.append((m.start(1), m.end(1), 'LINK_STREAMING'))

    return entities


# ── Sample 04 ──
def _extract_sample04(text):
    """Extract entities from sample-dataset-04.

    Single-session with BANK, NOREK, KONTAK, LINK_STREAMING, PEMATERI, TEMA.
    """
    pats = [
        # TEMA — First line
        (r'^(TUJUAN HIDUP MUSLIM SETIAP HARI)',              'TEMA'),
        # PEMATERI
        (r'(Ustadz\s+[A-Za-z\s]+(?:,\s*[A-Z][a-z]*\.?)+)',   'PEMATERI'),
        # LINK_STREAMING
        (r'(https?://\S+)',                                  'LINK_STREAMING'),
        # BANK
        (r'🏧\s*(BANK\s+[A-Za-z\s]+)',                       'BANK'),
        # NOREK
        (r'No\.\s*Rek\s*:\s*([\d\s]+)',                      'NOREK'),
        # KONTAK
        (r'(\d{4}-\d+)',                                     'KONTAK'),
        (r'(wa\.me/\d+)',                                    'KONTAK'),
    ]
    return _apply_patterns(text, pats)


# ── Sample 05 ──
def _extract_sample05(text):
    """Extract entities from sample-dataset-05."""
    pats = [
        # TEMA
        (r'^(Jejak Rasul 5\.0 Perjanjian Hudaybiyyah)',        'TEMA'),
        # KONTAK
        (r'(\d{11,13})',                                      'KONTAK'),
    ]
    return _apply_patterns(text, pats)


# ── Sample 06 ──
def _extract_sample06(text):
    """Extract entities from sample-dataset-06."""
    pats = [
        # WAKTU
        (r'📆\s*(.+)$',                                       'WAKTU'),
        (r'🕰️\s*(.+)$',                                       'WAKTU'),
        # PEMATERI
        (r'🩺\s*(.+)$',                                       'PEMATERI'),
        # TEMA
        (r'📜\s*\*(.+?)\*',                                   'TEMA'),
        # LINK_STREAMING
        (r'(https?://\S+)',                                  'LINK_STREAMING'),
        (r'Website HSI Berbagi\s*:\s*(\S+)',                 'LINK_STREAMING'),
    ]
    return _apply_patterns(text, pats)


# ── Sample 07 ──
def _extract_sample07(text):
    """Extract entities from sample-dataset-07."""
    pats = [
        # TEMA
        (r'^\*(Kajian Rutin Masjid Al-Fattah Jatinegara Jakarta Timur)\*', 'TEMA'),
        (r'📖\s*(.+?)(?:\s*\||$)',                            'TEMA'),
        # PEMATERI
        (r'\*(Ustadz Dr. Abdullah Roy, M.A.)',                'PEMATERI'),
        # WAKTU
        (r'🗓\s*(.+)$',                                        'WAKTU'),
        (r'🕗\s*(.+)$',                                        'WAKTU'),
        # LOKASI
        (r'🕌\s*(.+)$',                                       'LOKASI'),
        # ALAMAT
        (r'🕌\s*.+?\n\s*([A-Za-z\s]+?)(?=\n\s*(?:https?://|•••))', 'ALAMAT'),
        # LINK_STREAMING
        (r'(youtube\.com/\S+)',                               'LINK_STREAMING'),
        (r'(instagram\.com/\S+)',                             'LINK_STREAMING'),
        (r'(t\.me/\S+)',                                      'LINK_STREAMING'),
        (r'(radiohsi\.com)',                                  'LINK_STREAMING'),
        (r'(facebook\.com/\S+)',                              'LINK_STREAMING'),
        (r'(twitter\.com/\S+)',                               'LINK_STREAMING'),
        (r'(threads\.net/\S+)',                               'LINK_STREAMING'),
        (r'(whatsapp\.com/channel/\S+)',                       'LINK_STREAMING'),
        (r'(https?://\S+)',                                  'LINK_STREAMING'),
    ]
    return _apply_patterns(text, pats)


# ── Sample 08 ──
def _extract_sample08(text):
    """Extract entities from sample-dataset-08."""
    pats = [
        # TEMA
        (r'^(Kajian Umum Sirah Nabawiyyah)',                  'TEMA'),
        (r'📖\s*\*(.+?)\*',                                   'TEMA'),
        # PEMATERI
        (r'\*(Ustadz Dr. Abdullah Roy, M.A.)',                'PEMATERI'),
        # WAKTU
        (r'🗓\s*(.+)$',                                        'WAKTU'),
        (r'🕕\s*(.+)$',                                        'WAKTU'),
        # LOKASI
        (r'🕌\s*(.+)$',                                       'LOKASI'),
        # ALAMAT
        (r'🕌\s*.+?\n\s*([A-Za-z\s]+?)(?=\n\s*(?:https?://|•••))', 'ALAMAT'),
        # LINK_STREAMING
        (r'(youtube\.com/\S+)',                               'LINK_STREAMING'),
        (r'(instagram\.com/\S+)',                             'LINK_STREAMING'),
        (r'(t\.me/\S+)',                                      'LINK_STREAMING'),
        (r'(radiohsi\.com)',                                  'LINK_STREAMING'),
        (r'(facebook\.com/\S+)',                              'LINK_STREAMING'),
        (r'(twitter\.com/\S+)',                               'LINK_STREAMING'),
        (r'(threads\.net/\S+)',                               'LINK_STREAMING'),
        (r'(whatsapp\.com/channel/\S+)',                       'LINK_STREAMING'),
        (r'(https?://\S+)',                                  'LINK_STREAMING'),
    ]
    return _apply_patterns(text, pats)


# ── Sample 09 ──
def _extract_sample09(text):
    """Extract entities from sample-dataset-09."""
    pats = [
        # TEMA
        (r'^(Live Interaktif Fatwa TV \| Jalan Lurus)',       'TEMA'),
        # PEMATERI
        (r'\*(Ustadz Dr. Abdullah Roy, M.A.)',                'PEMATERI'),
        # WAKTU
        (r'📆\s*(.+)$',                                       'WAKTU'),
        (r'🕗\s*(.+)$',                                       'WAKTU'),
        # KONTAK
        (r'(\d{3}\s*-\s*\d{4}\s*\d{4})',                      'KONTAK'),
        # LINK_STREAMING
        (r'(youtube\.com/\S+)',                               'LINK_STREAMING'),
        (r'(instagram\.com/\S+)',                             'LINK_STREAMING'),
        (r'(t\.me/\S+)',                                      'LINK_STREAMING'),
        (r'(radiohsi\.com)',                                  'LINK_STREAMING'),
        (r'(facebook\.com/\S+)',                              'LINK_STREAMING'),
        (r'(twitter\.com/\S+)',                               'LINK_STREAMING'),
        (r'(threads\.net/\S+)',                               'LINK_STREAMING'),
        (r'(whatsapp\.com/channel/\S+)',                       'LINK_STREAMING'),
        (r'(https?://\S+)',                                  'LINK_STREAMING'),
    ]
    return _apply_patterns(text, pats)


# ── Sample 10 ──
def _extract_sample10(text):
    """Extract entities from sample-dataset-10."""
    pats = [
        # TEMA
        (r'\*(TAHSIN TILAWAH UMUM IKHWAN)\*',                 'TEMA'),
        # PEMATERI
        (r'Bersama\s+(Mu’allim Tartil Academy - QITA Ikhwan)', 'PEMATERI'),
        # WAKTU
        (r'-\s*(Ahad,\s*\d+\s*[A-Za-z]+\s*\d{4})',            'WAKTU'),
        (r'(Pukul\s*\d+\.\d+\s*WIB\s*s/d\s*selesai)',          'WAKTU'),
        # LOKASI
        (r'(Zoom)',                                           'LOKASI'),
        # LINK_STREAMING
        (r'(https?://\S+)',                                  'LINK_STREAMING'),
        # STATUS
        (r'\*(Terbuka untuk umum khusus Ikhwan/Lelaki minimal 11 tahun)\*', 'STATUS'),
    ]
    return _apply_patterns(text, pats)


# ── Sample 11 ──
def _extract_sample11(text):
    """Extract entities from sample-dataset-11."""
    pats = [
        # TEMA
        (r'Tema:\s*"([^"]+)"',                                'TEMA'),
        (r'📗\s*(.+)$',                                       'TEMA'),
        # PEMATERI
        (r'👤\s*(Ustadz\s+[A-Za-z\s]+(?:,\s*(?!Hafidzahullah|Hafizhahullah|Hafidzahullahu|Hafizhahullahu)[A-Za-z\.]+)*)',   'PEMATERI'),
        # WAKTU
        (r'🕰\s*(.+)$',                                       'WAKTU'),
        (r'📆\s*(.+)$',                                       'WAKTU'),
        # LOKASI
        (r'🕌\s*(.+)$',                                       'LOKASI'),
        # ALAMAT
        (r'^🕌\s*[\s\S]+?\n\s*([\s\S]+?)(?=\n\s*(?:📲|🖇|👤|🕰|📆|📗|📮))', 'ALAMAT'),
        # KONTAK / WA
        (r'(https://wa\.me/\+\d+)',                           'KONTAK'),
        # LINK_STREAMING
        (r'(https?://\S+)',                                  'LINK_STREAMING'),
        # STATUS
        (r'(TERBUKA UNTUK UMUM)',                              'STATUS'),
    ]
    return _apply_patterns(text, pats)


# ── Sample 12 ──
def _extract_sample12(text):
    """Extract entities from sample-dataset-12."""
    pats = [
        # TEMA
        (r'📝\s*(.+)$',                                       'TEMA'),
        (r'📚\s*\|\s*(.+)$',                                   'TEMA'),
        # PEMATERI
        (r'🎙️\s*(Ustadz\s+[A-Za-z\s]+,\s*[A-Z\.]+\s*,\s*[A-Z\.]+)', 'PEMATERI'),
        # WAKTU
        (r'📆\s*\|\s*(.+)$',                                   'WAKTU'),
        (r'🕗\s*\|\s*(.+)$',                                   'WAKTU'),
        # LOKASI
        (r'Disiarkan langsung dari\s*:\s*\n(.+)',             'LOKASI'),
        # ALAMAT
        (r'Disiarkan langsung dari\s*:\s*\n.+?\n\s*(.+)',    'ALAMAT'),
        # LINK_STREAMING
        (r'(https?://\S+)',                                  'LINK_STREAMING'),
    ]
    return _apply_patterns(text, pats)


# ── Sample 13 ──
def _split_sample13_sessions(text):
    """Split Surabaya Mengaji rekap into sessions (⏰ blocks) and the donation footer."""
    blocks = re.split(r'\n\n+', text)
    sessions = [b.strip() for b in blocks if b.strip().startswith('⏰')]

    # Check if there is a donation block at the end
    donation_block = ""
    for b in blocks:
        if "💳" in b or "BSI -" in b:
            donation_block = b.strip()
            break

    if donation_block:
        sessions.append(donation_block)

    return sessions


def _extract_sample13(text):
    """Extract entities from sample-dataset-13 session or donation block."""
    if text.startswith('⏰'):
        pats = [
            (r'^⏰\s*(.+)$',                                       'WAKTU'),
            (r'^📚\s*(.+)$',                                       'TEMA'),
            (r'^👤\s*(Ustadz\s+[A-Za-z\s]+(?:,\s*[A-Za-z\.]+)*)',   'PEMATERI'),
            (r'^🔗\s*(\S+)',                                      'LINK_STREAMING'),
        ]
        entities = _apply_patterns(text, pats)
        
        # Parse LOKASI and ALAMAT dynamically from 📍 line
        m = re.search(r'^📍\s*(.+)$', text, re.MULTILINE)
        if m:
            full_loc = m.group(1).strip()
            s_loc_line = m.start(1)
            
            # Check if there is a street address marker (e.g. " - Jl.", ", Jl.", " - Jalan", ", Jalan")
            split_match = re.search(r'\s*[\-,]\s*(?:Jl\.?|Jalan)\b', full_loc, flags=re.IGNORECASE)
            if split_match:
                split_idx = split_match.start()
                loc_name = full_loc[:split_idx].strip()
                addr_part = full_loc[split_idx:].strip()
                addr_part = re.sub(r'^[\s\-,]+', '', addr_part).strip()
                
                # Add location name entity
                s_loc = s_loc_line
                e_loc = s_loc + len(loc_name)
                entities.append((s_loc, e_loc, 'LOKASI'))
                
                # Add address entity
                s_addr = s_loc_line + full_loc.find(addr_part)
                e_addr = s_addr + len(addr_part)
                entities.append((s_addr, e_addr, 'ALAMAT'))
            else:
                # Whole line is LOKASI
                e_loc = s_loc_line + len(full_loc)
                entities.append((s_loc_line, e_loc, 'LOKASI'))
                
        return entities
    else:
        # Donation block
        pats = [
            (r'💳\s*([A-Za-z]+)',                                  'BANK'),
            (r'-\s*([\d\s]+)',                                     'NOREK'),
            (r'Konfirmasi:\s*(\d+)',                               'KONTAK'),
        ]
        return _apply_patterns(text, pats)



# ═════════════════════════════════════════════════════════════
# Step 3: Relation Alignment — Validate, Clean, Dedup
# ═════════════════════════════════════════════════════════════

def validate_and_clean(text, entities):
    """Sort entities by position, remove overlaps, filter noise, validate."""
    # 1. Filter out meaningless KONTAK entries (just dashes/emojis/no digits)
    filtered = []
    for s, e, label in entities:
        val = text[s:e].strip()
        if label == 'KONTAK':
            cleaned = re.sub(r'[–\-\s🚹🚺/]', '', val)
            if not cleaned or not any(c.isdigit() for c in cleaned):
                continue
        filtered.append((s, e, label))

    # 2. Sort by start position, then by end position
    filtered.sort(key=lambda x: (x[0], x[1]))

    # 3. Remove overlapping entities (keep first occurrence)
    result = []
    last_end = -1
    for s, e, label in filtered:
        if s >= last_end:
            # Validate indices
            if 0 <= s < e <= len(text):
                result.append((s, e, label))
                last_end = e

    return result


# ═════════════════════════════════════════════════════════════
# Step 4: Export — Write TRAIN_DATA to Python file
# ═════════════════════════════════════════════════════════════

def export_train_data(all_data, output_path="training/data_latihan_spacy.py"):
    """Write all training data as a Python file with TRAIN_DATA variable."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write('# -*- coding: utf-8 -*-\n')
        f.write('"""\n')
        f.write('data_latihan_spacy.py\n')
        f.write('═════════════════════\n')
        f.write('Auto-generated by generate_dataset.py (Workflow /build-dataset)\n')
        f.write('SpaCy NER training data for Jadwal Kajian parsing.\n')
        f.write('\n')
        f.write('Entity Labels:\n')
        f.write('  Wajib  : PEMATERI, LOKASI, WAKTU, TEMA, STATUS\n')
        f.write('  Dinamis: BANK, NOREK, KONTAK, LINK_STREAMING\n')
        f.write('"""\n\n')
        f.write('TRAIN_DATA = [\n')

        for i, (text, ents) in enumerate(all_data):
            f.write(f'\n    # ── Entry {i + 1} ──\n')
            f.write(f'    (\n')
            f.write(f'        {text!r},\n')
            f.write(f'        {{"entities": [\n')
            for s, e, label in ents:
                snippet = text[s:e][:60].replace('\n', '\\n')
                f.write(f'            ({s}, {e}, "{label}"),  # {snippet!r}\n')
            f.write(f'        ]}}\n')
            f.write(f'    ),\n')

        f.write(']\n')

    return output_path


# ═════════════════════════════════════════════════════════════
# Verification — Sanity-check the generated data
# ═════════════════════════════════════════════════════════════

def verify_train_data(all_data):
    """Verify that all entity indices are correct."""
    errors = 0
    for i, (text, ents) in enumerate(all_data):
        for s, e, label in ents:
            val = text[s:e]
            if not val.strip():
                print(f"  ❌ Entry {i+1}: empty entity ({s},{e},{label})")
                errors += 1
            if s < 0 or e > len(text) or s >= e:
                print(f"  ❌ Entry {i+1}: bad index ({s},{e},{label})")
                errors += 1
    if errors == 0:
        print("  ✅ All entity indices verified OK")
    else:
        print(f"  ❌ Found {errors} error(s)")
    return errors


def synthesize_comma_separated_variations(all_data):
    """Generates synthetic examples of comma-separated LOKASI and ALAMAT."""
    synthetic_data = []
    for session, entities in all_data:
        if "Masjid al-Ikhlash" in session and ", Jl." in session:
            variations = [
                ("Masjid An-Nur", "Jl. Diponegoro No. 12, Kel. Tegalsari, Kec. Tegalsari, Surabaya"),
                ("Masjid Al-Barkah", "Jl. Slamet Riyadi No. 5, Laweyan, Surakarta"),
                ("Musholla At-Taqwa", "Jl. Pemuda No. 45, Sekayu, Kec. Semarang Tengah, Kota Semarang"),
                ("Masjid Istiqlal", "Jl. Taman Wijaya Kusuma, Pasar Baru, Kec. Sawah Besar, Kota Jakarta Pusat"),
                ("Masjid Raya Baiturrahman", "Jl. Moh. Jam No.1, Kampung Baru, Kec. Baiturrahman, Kota Banda Aceh"),
            ]
            for loc_name, addr_name in variations:
                old_loc = "Masjid al-Ikhlash - Delta Sari Indah"
                old_addr = "Jl. Anggrek VI no.36-38, Kureksari, Waru, Sidoarjo"
                
                new_session = session.replace(old_loc, loc_name).replace(old_addr, addr_name)
                new_entities = _extract_sample13(new_session)
                new_entities = validate_and_clean(new_session, new_entities)
                if new_entities:
                    synthetic_data.append((new_session, new_entities))
            break
    return synthetic_data


# ═════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("📊 Workflow /build-dataset: Building NER Training Data")
    print("=" * 60)

    # ── Step 1: Ingestion ──
    print("\n📥 Step 1: Ingestion")
    files = read_all_files()
    print(f"   Found {len(files)} non-empty file(s)")
    for fname in files:
        print(f"   · {fname} ({len(files[fname]):,} chars)")

    # ── Step 2 & 3: NER Tagging & Relation Alignment ──
    print("\n🏷️  Step 2-3: NER Tagging & Relation Alignment")
    all_data = []
    for fname, content in files.items():
        sessions = split_sessions(content, fname)
        print(f"\n   📄 {fname} → {len(sessions)} session(s)")
        for j, session in enumerate(sessions):
            entities = extract_entities(session, fname)
            entities = validate_and_clean(session, entities)
            if entities:
                all_data.append((session, entities))
                labels = sorted(set(l for _, _, l in entities))
                print(f"      #{j+1:>2d}: {len(entities)} entities {labels}")

    # ── Data Augmentation ──
    synthetic_vars = synthesize_comma_separated_variations(all_data)
    if synthetic_vars:
        print(f"\n➕ Added {len(synthetic_vars)} synthetic comma-separated location variations for training")
        all_data.extend(synthetic_vars)

    # ── Verification ──
    print("\n🔍 Verification:")
    verify_train_data(all_data)

    # ── Step 4: Export ──
    print(f"\n📤 Step 4: Export")
    out = export_train_data(all_data)
    print(f"   → Written to: {out}")

    # ── Summary ──
    print(f"\n{'=' * 60}")
    total_e = sum(len(e) for _, e in all_data)
    label_cnt = {}
    for _, ents in all_data:
        for _, _, l in ents:
            label_cnt[l] = label_cnt.get(l, 0) + 1
    print(f"📊 Summary: {len(all_data)} training examples, {total_e} total entities")
    for l, c in sorted(label_cnt.items()):
        print(f"   {l:.<20s} {c}")
    print("=" * 60)
    print("\n🎯 Done! Jalankan di venv:")
    print("   source venv/bin/activate")
    print("   python training/generate_dataset.py")


if __name__ == "__main__":
    main()
