#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict.py
══════════
Run inference on a text file using the trained SpaCy NER model.
"""

import sys
import os
import spacy

def main():
    if len(sys.argv) > 1:
        fpath = sys.argv[1]
    else:
        fpath = "data-sample/text-base/sample-dataset-01.txt"

    if not os.path.exists(fpath):
        print(f"Error: File not found at {fpath}")
        sys.exit(1)

    model_path = "output_model/model-best"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please run training first.")
        sys.exit(1)

    print(f"🤖 Loading SpaCy model from: {model_path} ...")
    nlp = spacy.load(model_path)

    print(f"📄 Reading input file: {fpath} ...")
    with open(fpath, "r", encoding="utf-8") as f:
        text = f.read()

    print("\n--- Raw Text ---")
    print(text.strip())
    print("----------------\n")

    doc = nlp(text)

    print("🏷️  Extracted Entities:")
    if not doc.ents:
        print("  (No entities found)")
    else:
        for ent in doc.ents:
            print(f"  [{ent.label_:<15}] -> {ent.text!r} ({ent.start_char}, {ent.end_char})")

if __name__ == "__main__":
    main()
