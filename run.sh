#!/bin/bash
# ==============================================================================
# run.sh
# ══════
# Bash script to easily execute parsing and debugging tasks.
# Usage:
#   ./run.sh parse <input_file>      -> Run parser with NER-only model
#   ./run.sh parse-rel <input_file>  -> Run parser with Joint NER + RE model
#   ./run.sh debug <input_file>      -> Run debugger log with NER-only model
#   ./run.sh debug-rel <input_file>  -> Run debugger log with Joint NER + RE model
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

# 2. Parse arguments
ACTION=${1:-""}
INPUT_FILE=${2:-""}

if [ -z "$ACTION" ] || [ -z "$INPUT_FILE" ]; then
    echo "❌ Error: Missing arguments."
    echo "Usage:"
    echo "  ./run.sh parse <input_file>"
    echo "  ./run.sh parse-rel <input_file>"
    echo "  ./run.sh debug <input_file>"
    echo "  ./run.sh debug-rel <input_file>"
    exit 1
fi

case "$ACTION" in
    "parse")
        echo "🚀 Running parser (NER-only)..."
        python src/parse_to_json.py "$INPUT_FILE"
        ;;
    "parse-rel")
        echo "🚀 Running parser (Joint NER + RE)..."
        python src/parse_to_json.py "$INPUT_FILE" --use-rel
        ;;
    "debug")
        echo "🔍 Running debugger log (NER-only)..."
        python src/debug_ner.py "$INPUT_FILE"
        ;;
    "debug-rel")
        echo "🔍 Running debugger log (Joint NER + RE)..."
        python src/debug_ner.py "$INPUT_FILE" --use-rel
        ;;
    *)
        echo "❌ Error: Invalid action '$ACTION'."
        echo "Available actions: parse, parse-rel, debug, debug-rel"
        exit 1
        ;;
esac
