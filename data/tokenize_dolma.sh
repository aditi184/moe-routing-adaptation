dolma tokens \
    --documents /combined \
    --destination /combined/tokenized \
    --tokenizer.name_or_path allenai/OLMoE-1B-7B-0924-Instruct \
    --max_size '2_147_483_648' \
    --seed 0 \
    --tokenizer.eos_token_id 0 \
    --tokenizer.pad_token_id 1 \
    --processes 32