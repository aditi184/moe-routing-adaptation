#!/bin/bash
# Resume dolma tokenization by removing incomplete parts and re-running

DOCUMENTS_DIR="/combined"
DESTINATION_DIR="/combined/tokenized"
TOKENIZER="allenai/OLMoE-1B-7B-0924-Instruct"
MAX_SIZE="2_147_483_648"
SEED=0
PROCESSES=32

echo "Checking tokenization status..."

# Check for incomplete parts (parts that are significantly smaller than max_size)
# Remove underscores from MAX_SIZE for arithmetic
MAX_SIZE_NUM=$(echo "$MAX_SIZE" | tr '_' '')
MAX_SIZE_BYTES=$((MAX_SIZE_NUM * 2))  # uint16 = 2 bytes per token
THRESHOLD=$((MAX_SIZE_BYTES * 90 / 100))  # 90% of max_size

INCOMPLETE_PARTS=()
for npy_file in ${DESTINATION_DIR}/part-*.npy; do
    if [ -f "$npy_file" ]; then
        size=$(stat -c%s "$npy_file" 2>/dev/null)
        if [ "$size" -lt "$THRESHOLD" ]; then
            # This part is incomplete
            base=$(basename "$npy_file" .npy)
            INCOMPLETE_PARTS+=("$base")
            echo "  Found incomplete part: $base (size: $((size / 1024 / 1024 / 1024)) GB)"
        fi
    fi
done

if [ ${#INCOMPLETE_PARTS[@]} -eq 0 ]; then
    echo "All parts appear complete!"
    echo "Checking if there are remaining input files to process..."
    
    # Check which input files were processed
    PROCESSED_FILES=()
    for csv_file in ${DESTINATION_DIR}/*.csv.gz; do
        if [ -f "$csv_file" ]; then
            PROCESSED=$(zcat "$csv_file" 2>/dev/null | cut -d',' -f4 | sort -u)
            while IFS= read -r file; do
                if [[ ! " ${PROCESSED_FILES[@]} " =~ " ${file} " ]]; then
                    PROCESSED_FILES+=("$file")
                fi
            done <<< "$PROCESSED"
        fi
    done
    
    INPUT_FILES=($(ls -1 ${DOCUMENTS_DIR}/*.jsonl | sort))
    REMAINING=()
    for input_file in "${INPUT_FILES[@]}"; do
        found=false
        for processed_file in "${PROCESSED_FILES[@]}"; do
            if [ "$input_file" == "$processed_file" ]; then
                found=true
                break
            fi
        done
        if [ "$found" == false ]; then
            REMAINING+=("$input_file")
        fi
    done
    
    if [ ${#REMAINING[@]} -eq 0 ]; then
        echo "All input files have been processed!"
        exit 0
    else
        echo "Found ${#REMAINING[@]} unprocessed file(s):"
        for file in "${REMAINING[@]}"; do
            echo "  $file"
        done
    fi
else
    echo ""
    echo "Found ${#INCOMPLETE_PARTS[@]} incomplete part(s)."
    echo "Removing incomplete parts..."
    
    for part in "${INCOMPLETE_PARTS[@]}"; do
        rm -f "${DESTINATION_DIR}/${part}.npy"
        rm -f "${DESTINATION_DIR}/${part}.csv.gz"
        echo "  Removed: $part"
    done
    
    echo ""
    echo "Incomplete parts removed. Re-running dolma to continue from where it left off..."
fi

echo ""
echo "Running dolma tokens..."
dolma tokens \
    --documents "$DOCUMENTS_DIR" \
    --destination "$DESTINATION_DIR" \
    --tokenizer.name_or_path "$TOKENIZER" \
    --max_size "$MAX_SIZE" \
    --seed $SEED \
    --tokenizer.eos_token_id 0 \
    --tokenizer.pad_token_id 1 \
    --processes $PROCESSES

if [ $? -eq 0 ]; then
    echo ""
    echo "Tokenization completed successfully!"
else
    echo ""
    echo "Tokenization failed. Check the error messages above."
    exit 1
fi
