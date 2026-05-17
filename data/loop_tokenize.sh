#!/bin/bash

# List of languages
# languages=("ca" "et" "sk" "ur")  # Add more language codes as needed
languages=("sk" "ca" "et" "ur" "nl" "uk")
# Base paths
# base_input="/evaluation"
# base_output="/evaluation/tokenized"

base_input="/low_resource/"
base_output="/low_resource/tokenized"


# Loop over each language
for lang in "${languages[@]}"; do
  echo "Processing language: $lang"
  
  dolma tokens \
    --documents "${base_input}/${lang}_train_1.jsonl" \
    --destination "${base_output}/${lang}" \
    --tokenizer.name_or_path 'allenai/gpt-neox-olmo-dolma-v1_5' \
    --max_size '2_147_483_648' \
    --seed 0 \
    --tokenizer.eos_token_id 50279 \
    --tokenizer.pad_token_id 1 \
    --processes 32
done
