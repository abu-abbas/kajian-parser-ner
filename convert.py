# -*- coding: utf-8 -*-
"""
convert.py
══════════
Converts Python list TRAIN_DATA in data_latihan_spacy.py
into SpaCy's binary .spacy format (train.spacy and dev.spacy) for training.
"""

import os
import random
import spacy
from spacy.tokens import DocBin
from data_latihan_spacy import TRAIN_DATA

def convert(data, output_path, nlp):
    db = DocBin()
    skipped = 0
    success = 0
    for i, (text, annot) in enumerate(data):
        doc = nlp.make_doc(text)
        ents = []
        for start, end, label in annot["entities"]:
            # Create a span in the doc
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None:
                # Warning if index doesn't align with token boundaries
                # contract mode will shrink the boundary slightly to match token if possible,
                # but if it returns None, it means tokenization didn't align.
                print(f"  ⚠️ Warning: Entry {i+1} has misaligned entity [{text[start:end]}] ({start}, {end}, {label}). Skipped this entity.")
                skipped += 1
            else:
                ents.append(span)
        
        # Resolve overlaps just in case (keep longer spans or first spans)
        try:
            doc.ents = ents
            db.add(doc)
            success += 1
        except Exception as e:
            print(f"  ❌ Error adding Doc Entry {i+1}: {e}")
            skipped += 1
            
    db.to_disk(output_path)
    print(f"  ✅ Saved {success} documents to {output_path} (entities skipped: {skipped})")

def main():
    print("=" * 60)
    print("📦 SpaCy Dataset Converter & Train/Dev Splitter")
    print("=" * 60)
    
    # 1. Load empty Indonesian language model for tokenization
    try:
        nlp = spacy.blank("id")
    except Exception:
        print("  ⚠️ SpaCy 'id' model not available. Falling back to multi-language 'xx'.")
        nlp = spacy.blank("xx")

    # 2. Shuffle data for unbiased split
    random.seed(42)  # For reproducibility
    shuffled_data = list(TRAIN_DATA)
    random.shuffle(shuffled_data)

    # 3. Split: 80% Train, 20% Dev (Validation)
    split_idx = int(len(shuffled_data) * 0.8)
    train_data = shuffled_data[:split_idx]
    dev_data = shuffled_data[split_idx:]

    print(f"📥 Total examples: {len(shuffled_data)}")
    print(f"   ↳ Train split: {len(train_data)}")
    print(f"   ↳ Dev split  : {len(dev_data)}")
    print("-" * 60)

    # 4. Convert and save
    print("🔄 Converting Train set...")
    convert(train_data, "train.spacy", nlp)
    
    print("🔄 Converting Dev set...")
    convert(dev_data, "dev.spacy", nlp)
    
    print("=" * 60)
    print("🎯 Preparation Complete! Next steps:")
    print("   1. Install spacy (if not done yet):")
    print("      pip install spacy")
    print("   2. Generate training config:")
    print("      python -m spacy init config config.cfg --pipeline ner --lang id --force")
    print("   3. Run training:")
    print("      python -m spacy train config.cfg --output ./output_model --paths.train ./train.spacy --paths.dev ./dev.spacy")
    print("=" * 60)

if __name__ == "__main__":
    main()
