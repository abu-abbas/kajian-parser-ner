#!/bin/bash
# ==============================================================================
# train.sh
# ════════
# Bash script to automate the entire training pipeline for Abu Abbas Jadwal Parser.
# Usage:
#   ./train.sh ner   -> Run training for NER-only model (saves to output/model_ner)
#   ./train.sh rel   -> Run training for Joint NER + RE model (saves to output/model_rel)
# ==============================================================================

# Ensure script stops on first error
set -e

# 1. Automatically activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "🔌 Activating virtual environment..."
    source venv/bin/activate
else
    echo "⚠️ Warning: venv folder not found. Running with current python interpreter."
fi

# 2. Parse mode argument (defaults to 'rel')
MODE=${1:-"rel"}

if [ "$MODE" == "ner" ]; then
    echo ""
    echo "======================================================================"
    echo "🏋️‍♂️ Starting Retraining Pipeline: NER-ONLY Mode"
    echo "======================================================================"
    
    echo "🔄 Step 1: Regenerating clean entities dataset via Gemini LLM..."
    python training/generate_dataset_via_llm.py
    
    echo "🔄 Step 2: Converting Python dataset to binary train/dev.spacy..."
    python training/convert.py
    
    echo "🏋️‍♂️ Step 3: Running SpaCy NER training (saving to output/model_ner)..."
    python -m spacy train training/config.cfg --output ./output/model_ner --paths.train ./training/train.spacy --paths.dev ./training/dev.spacy

    echo "======================================================================"
    echo "✅ NER-ONLY training completed successfully!"
    echo "📂 Model saved in: output/model_ner/model-best"
    echo "💡 Run parser with: python src/parse_to_json.py <file>"
    echo "======================================================================"

elif [ "$MODE" == "rel" ]; then
    echo ""
    echo "======================================================================"
    echo "🏋️‍♂️ Starting Retraining Pipeline: Joint NER + RE Mode"
    echo "======================================================================"
    
    echo "🔄 Step 1: Regenerating clean entities dataset via Gemini LLM..."
    python training/generate_dataset_via_llm.py
    
    echo "🔄 Step 2: Regenerating relations dataset (JSONL)..."
    python training/generate_dataset_relations.py
    
    echo "🔄 Step 3: Converting relation dataset to binary train_rel/dev_rel.spacy..."
    python training/convert_rel.py
    
    echo "🏋️‍♂️ Step 4: Running SpaCy Joint NER + RE training (saving to output/model_rel)..."
    python -m spacy train training/config_rel.cfg --output ./output/model_rel --paths.train ./training/train_rel.spacy --paths.dev ./training/dev_rel.spacy --code training/rel_component.py

    echo "======================================================================"
    echo "✅ Joint NER + RE training completed successfully!"
    echo "📂 Model saved in: output/model_rel/model-best"
    echo "💡 Run parser with: python src/parse_to_json.py <file> --use-rel"
    echo "======================================================================"

else
    echo "❌ Error: Invalid training mode '$MODE'."
    echo "Usage: ./train.sh [ner|rel]"
    exit 1
fi
