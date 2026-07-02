#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_dataset_relations.py
═════════════════════════════
Reads RAW_DATA from training/data_latihan_spacy.py, resolves exact substring indices
for entities, automatically establishes logical relations between entities of the same session,
and exports the dataset as a standardized JSONL file for Relation Extraction training.

Relations Established:
  - PEMATERI -> TEMA (label: "PEMATERI_TEMA")
  - PEMATERI -> WAKTU (label: "PEMATERI_WAKTU")
  - PEMATERI -> LOKASI (label: "PEMATERI_LOKASI")
  - LOKASI -> ALAMAT (label: "LOKASI_ALAMAT")
  - LOKASI -> LINK_MAPS (label: "LOKASI_LINK_MAPS")
  - LOKASI -> METODE (label: "LOKASI_METODE")
  - BANK -> NOREK (label: "BANK_NOREK")
"""

import os
import sys
import json

# Ensure training directory is in path to import data_latihan_spacy
_TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
if _TRAINING_DIR not in sys.path:
    sys.path.insert(0, _TRAINING_DIR)

try:
    from data_latihan_spacy import TRAIN_DATA
except ImportError:
    print("Error: Could not import TRAIN_DATA from data_latihan_spacy.py")
    sys.exit(1)

def generate_relations_jsonl(output_path="training/data_latihan_relations.jsonl"):
    print("🚀 Generating Relation Extraction Dataset (JSONL)...")
    
    jsonl_records = []
    
    for idx, (text, entity_coords) in enumerate(TRAIN_DATA):
        # 1. Resolve entity coordinates directly (from new start, end format)
        ents_resolved = []
        entities_list = entity_coords.get("entities", [])
        
        for start, end, label in entities_list:
            substring = text[start:end]
            ents_resolved.append({
                "start": start,
                "end": end,
                "label": label,
                "text": substring
            })
            
        # Sort entities by start index
        ents_resolved.sort(key=lambda x: x["start"])
        
        # 2. Build map of labels to entity indices in the resolved list
        label_to_indices = {}
        for ent_idx, ent in enumerate(ents_resolved):
            lbl = ent["label"]
            if lbl not in label_to_indices:
                label_to_indices[lbl] = []
            label_to_indices[lbl].append(ent_idx)
            
        # 3. Establish relations
        relations = []
        
        # Connect PEMATERI to TEMA, WAKTU, LOKASI
        if "PEMATERI" in label_to_indices:
            for p_idx in label_to_indices["PEMATERI"]:
                if "TEMA" in label_to_indices:
                    for t_idx in label_to_indices["TEMA"]:
                        relations.append({"head": p_idx, "child": t_idx, "label": "PEMATERI_TEMA"})
                if "WAKTU" in label_to_indices:
                    for w_idx in label_to_indices["WAKTU"]:
                        relations.append({"head": p_idx, "child": w_idx, "label": "PEMATERI_WAKTU"})
                if "LOKASI" in label_to_indices:
                    for l_idx in label_to_indices["LOKASI"]:
                        relations.append({"head": p_idx, "child": l_idx, "label": "PEMATERI_LOKASI"})
                        
        # Connect LOKASI to ALAMAT, LINK_MAPS, METODE
        if "LOKASI" in label_to_indices:
            for l_idx in label_to_indices["LOKASI"]:
                if "ALAMAT" in label_to_indices:
                    for a_idx in label_to_indices["ALAMAT"]:
                        relations.append({"head": l_idx, "child": a_idx, "label": "LOKASI_ALAMAT"})
                if "LINK_MAPS" in label_to_indices:
                    for m_idx in label_to_indices["LINK_MAPS"]:
                        relations.append({"head": l_idx, "child": m_idx, "label": "LOKASI_LINK_MAPS"})
                if "METODE" in label_to_indices:
                    for mt_idx in label_to_indices["METODE"]:
                        relations.append({"head": l_idx, "child": mt_idx, "label": "LOKASI_METODE"})
                        
        # Connect BANK to NOREK
        if "BANK" in label_to_indices and "NOREK" in label_to_indices:
            for b_idx in label_to_indices["BANK"]:
                for n_idx in label_to_indices["NOREK"]:
                    relations.append({"head": b_idx, "child": n_idx, "label": "BANK_NOREK"})
                    
        # Remove text field from entities list to match standard RE formats
        ents_clean = [{"start": e["start"], "end": e["end"], "label": e["label"]} for e in ents_resolved]
        
        record = {
            "text": text,
            "ents": ents_clean,
            "rels": relations
        }
        jsonl_records.append(record)
        
    # Write to output file
    with open(output_path, "w", encoding="utf-8") as f_out:
        for r in jsonl_records:
            f_out.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print(f"📊 Completed! Total relation extraction examples: {len(jsonl_records)}")
    print(f"💾 Written to: {output_path}")

if __name__ == "__main__":
    generate_relations_jsonl()
