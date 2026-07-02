#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_ner.py
════════════
Helper script to print raw SpaCy predictions (Entities & Relations) directly to terminal for debugging.
Outputs entities, their boundaries, and labels without any JSON post-processing.
"""

import sys
import os
import argparse
import spacy
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Helper script to print raw SpaCy predictions for debugging.")
    parser.add_argument("input_file", help="Path ke berkas teks rekap input (.txt)")
    parser.add_argument("output_file", nargs="?", default=None, help="Path ke berkas log output (.txt)")
    parser.add_argument("--use-rel", action="store_true", help="Gunakan model Joint NER + Relation Extraction (default: model NER)")
    
    args = parser.parse_args()
    input_path = args.input_file
    output_path = args.output_file
    use_rel = args.use_rel
    
    if output_path is None:
        # Dapatkan base name dari file input (tanpa ekstensi)
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        # Buat format datetime YYYYMMDD_HHMMSS
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Arahkan default output ke output/debugging/[nama_file_input]_[timestamp].txt
        output_path = f"output/debugging/{base_name}_{timestamp}.txt"
        
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        sys.exit(1)
        
    if use_rel:
        model_path = "output/model_rel/model-best"
    else:
        model_path = "output/model_ner/model-best"
        
    if not os.path.exists(model_path):
        print(f"Error: Model not found at '{model_path}'. Please train it first.")
        sys.exit(1)
        
    print(f"🤖 Loading SpaCy model from: {model_path} ...")
    
    # Coba impor custom component Relation Extraction secara dinamis jika model RE digunakan
    try:
        src_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(src_dir)
        training_dir = os.path.join(project_root, "training")
        if training_dir not in sys.path:
            sys.path.insert(0, training_dir)
        import rel_component
        print("🧩 Registered custom 'relation_extractor' component from training/rel_component.py")
    except Exception as e:
        print(f"ℹ️ Custom relation_extractor not registered: {e}")
        
    nlp = spacy.load(model_path)
    
    print(f"📄 Reading {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Bersihkan surrogate characters jika ada agar SpaCy make_doc tidak crash
    content_clean = "".join(c for c in content if not (0xD800 <= ord(c) <= 0xDFFF))
    doc = nlp(content_clean)
    
    # Kumpulkan output teks
    output_lines = []
    output_lines.append("="*80)
    output_lines.append(f"📊 DEBUG NER & RE LOG FOR: {os.path.basename(input_path)}")
    output_lines.append(f"🤖 Model Used: {model_path}")
    output_lines.append("="*80)
    
    output_lines.append("\n📝 [RAW INPUT TEXT]")
    output_lines.append("-" * 80)
    output_lines.append(content_clean.strip())
    output_lines.append("-" * 80)
    
    output_lines.append("\n🏷️  [RAW OUTPUT ENTITIES DETECTED]")
    output_lines.append("-" * 80)
    if not doc.ents:
        output_lines.append("❌ No entities detected by the model.")
    else:
        for ent in doc.ents:
            output_lines.append(f"  [{ent.label_:<15}] ({ent.start_char:>4}, {ent.end_char:>4}) -> {repr(ent.text)}")
            
    # Jika model relasi RE aktif, cetak relasi mentah dari model RE
    if "relation_extractor" in nlp.pipe_names and hasattr(doc._, "rel") and doc._.rel:
        output_lines.append("\n🔗 [RAW RELATIONS DETECTED (SpaCy RE)]")
        output_lines.append("-" * 80)
        for (s1, e1, s2, e2), rel_dict in doc._.rel.items():
            for label, prob in rel_dict.items():
                output_lines.append(f"  - {doc.text[s1:e1]!r} --({label}: {prob:.2f})--> {doc.text[s2:e2]!r}")
                
    output_lines.append("="*80)
    output_lines.append(f"📈 Total entities detected: {len(doc.ents)}")
    output_lines.append("="*80)
    
    # Print ke terminal
    print("\n" + "\n".join(output_lines))
    
    # Simpan ke file (selalu aktif ke target default atau custom)
    if output_path:
        out_dir = os.path.dirname(output_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
            
        with open(output_path, "w", encoding="utf-8") as f_out:
            f_out.write("\n".join(output_lines) + "\n")
        print(f"💾 Raw debugging output saved to: {output_path}")

if __name__ == "__main__":
    main()
