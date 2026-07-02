# -*- coding: utf-8 -*-
"""
convert_rel.py
══════════════
Converts JSONL Relation Extraction data in data_latihan_relations.jsonl
into SpaCy's binary .spacy format (train_rel.spacy and dev_rel.spacy)
with custom entity relations resolved in doc._.rel.
"""

import os
import sys
import json
import random
import spacy
from spacy.tokens import DocBin, Doc, Span

# Ensure rel_component is imported to register extensions
_TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
if _TRAINING_DIR not in sys.path:
    sys.path.insert(0, _TRAINING_DIR)

import rel_component  # Registers relation_extractor and doc._.rel extension

def load_jsonl(file_path):
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def convert_records(records, output_path, nlp):
    db = DocBin(store_user_data=True)
    skipped = 0
    success = 0
    
    for i, record in enumerate(records):
        text = record["text"]
        # Clean surrogate characters (0xD800 - 0xDFFF) to prevent SpaCy UnicodeEncodeError
        text_clean = "".join(c for c in text if not (0xD800 <= ord(c) <= 0xDFFF))
        doc = nlp.make_doc(text_clean)
        
        # 1. Resolve and assign entities
        ents = []
        for ent_data in record["ents"]:
            start = ent_data["start"]
            end = ent_data.get("end") or ent_data.get("output")
            if end is None:
                continue
            label = ent_data["label"]
            
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is not None:
                ents.append(span)
            else:
                skipped += 1
                
        # Handle overlaps and set entities
        try:
            doc.ents = ents
        except Exception as e:
            # Skip doc if entities overlap/fail
            skipped += 1
            continue
            
        # 2. Resolve and assign relations (map entity list indices to character coordinates)
        relations = {}
        for rel_data in record["rels"]:
            head_idx = rel_data["head"]
            child_idx = rel_data["child"]
            label = rel_data["label"]
            
            # Map back to doc entities
            if head_idx < len(doc.ents) and child_idx < len(doc.ents):
                ent_head = doc.ents[head_idx]
                ent_child = doc.ents[child_idx]
                
                key = (ent_head.start_char, ent_head.end_char, ent_child.start_char, ent_child.end_char)
                if key not in relations:
                    relations[key] = {}
                relations[key][label] = 1.0  # Training probability is 1.0 (True)
                
        doc._.rel = relations
        db.add(doc)
        success += 1
        
    db.to_disk(output_path)
    print(f"  ✅ Saved {success} documents to {output_path} (skipped errors: {skipped})")

def main():
    print("=" * 60)
    print("📦 SpaCy Joint NER + RE Dataset Converter")
    print("=" * 60)

    # 1. Load Indonesian language base
    try:
        nlp = spacy.blank("id")
    except Exception:
        print("  ⚠️ SpaCy 'id' model not available. Falling back to multi-language 'xx'.")
        nlp = spacy.blank("xx")

    # 2. Load JSONL records
    jsonl_path = "training/data_latihan_relations.jsonl"
    if not os.path.exists(jsonl_path):
        print(f"❌ Error: {jsonl_path} does not exist.")
        sys.exit(1)
        
    records = load_jsonl(jsonl_path)
    
    # 3. Shuffle & Split
    random.seed(42)
    random.shuffle(records)
    
    split_idx = int(len(records) * 0.8)
    train_records = records[:split_idx]
    dev_records = records[split_idx:]
    
    print(f"📥 Total records loaded: {len(records)}")
    print(f"   ↳ Train split: {len(train_records)}")
    print(f"   ↳ Dev split  : {len(dev_records)}")
    print("-" * 60)
    
    # 4. Save to binary files
    print("🔄 Converting Train set (with relations)...")
    convert_records(train_records, "training/train_rel.spacy", nlp)
    
    print("🔄 Converting Dev set (with relations)...")
    convert_records(dev_records, "training/dev_rel.spacy", nlp)
    
    print("=" * 60)
    print("🎯 Relation Dataset Conversion Complete!")
    print("   ↳ Outputs: training/train_rel.spacy & training/dev_rel.spacy")
    print("=" * 60)

if __name__ == "__main__":
    main()
