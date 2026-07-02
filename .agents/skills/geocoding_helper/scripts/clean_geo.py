# Helper script untuk membersihkan teks alamat sebelum ditembak ke Mapbox/OSM
import re

def clean_address_text(text):
    # Buang karakter aneh atau emoji yang mengacaukan Geocoding API
    cleaned = re.sub(r'[^\w\s,.-]', '', text)
    return cleaned.strip()
