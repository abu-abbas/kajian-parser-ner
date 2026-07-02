#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_ner.py
════════════
Helper script to print raw SpaCy predictions directly to terminal for debugging.
Outputs entities, their boundaries, and labels without any JSON post-processing.
"""

import sys
import os
import spacy

def main():
    if len(sys.argv) < 2:
        print("Usage: python src/debug_ner.py <input_text_file> [output_text_file]")
        sys.exit(1)
        
    input_path = sys.argv[1]
    
    # Import modul datetime untuk timestamp
    from datetime import datetime
    
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    else:
        # Dapatkan base name dari file input (tanpa ekstensi)
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        # Buat format datetime YYYYMMDD_HHMMSS
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Arahkan default output ke output/debugging/[nama_file_input]_[timestamp].txt
        output_path = f"output/debugging/{base_name}_{timestamp}.txt"
    
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        sys.exit(1)
        
    model_path = "output/model/model-best"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at '{model_path}'. Please train it first.")
        sys.exit(1)
        
    print(f"🤖 Loading model from {model_path}...")
    nlp = spacy.load(model_path)
    
    print(f"📄 Reading {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    doc = nlp(content)
    
    # Kumpulkan output teks
    output_lines = []
    output_lines.append("="*70)
    output_lines.append(f"📊 RAW ENTITY PREDICTIONS FOR: {os.path.basename(input_path)}")
    output_lines.append("="*70)
    
    if not doc.ents:
        output_lines.append("❌ No entities detected by the model.")
    else:
        for ent in doc.ents:
            output_lines.append(f"🏷️  [{ent.label_:<15}] ({ent.start_char:>4}, {ent.end_char:>4}) -> {repr(ent.text)}")
            
    output_lines.append("="*70)
    output_lines.append(f"📈 Total entities detected: {len(doc.ents)}")
    output_lines.append("="*70)
    
    # Print ke terminal
    print("\n" + "\n".join(output_lines))
    
    # Simpan ke file (selalu aktif ke target default atau custom)
    if output_path:
        out_dir = os.path.dirname(output_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
            
        with open(output_path, "w", encoding="utf-8") as f_out:
            f_out.write("\n".join(output_lines) + "\n")
        print(f"💾 Output saved to: {output_path}")

if __name__ == "__main__":
    main()
