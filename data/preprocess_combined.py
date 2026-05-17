# python preprocess_combined.py --multilang_dir /multilang/text --low_resource_file /low_resource/text/multilang_train_1.jsonl --output_dir /combined

import argparse
import json
import os
import random

def preprocess_combined(multilang_dir, low_resource_file, output_dir, multilang_lines=3_000_000):
    """
    Create a combined dataset from multilang and low-resource sources, then shuffle.
    
    Args:
        multilang_dir: Directory containing multilang_train_1.jsonl, multilang_train_2.jsonl, multilang_train_3.jsonl
        low_resource_file: Path to low-resource combined multilang_train_1.jsonl file
        output_dir: Output directory for the combined dataset
        multilang_lines: Number of lines to read from multilang files (default: 3M)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_lines = []
    
    # Step 1: Read lines from multilang files (up to multilang_lines)
    multilang_files = [
        os.path.join(multilang_dir, "multilang_train_1.jsonl"),
        os.path.join(multilang_dir, "multilang_train_2.jsonl"),
        os.path.join(multilang_dir, "multilang_train_3.jsonl"),
    ]
    
    print(f"Reading {multilang_lines:,} lines from multilang files...")
    multilang_count = 0
    
    for file_path in multilang_files:
        if not os.path.exists(file_path):
            print(f"  Warning: File not found: {file_path}")
            continue
        
        print(f"  Processing {os.path.basename(file_path)}...")
        with open(file_path, "r", encoding="utf8") as f:
            for line in f:
                if multilang_count >= multilang_lines:
                    break
                if line.strip():
                    all_lines.append(line)
                    multilang_count += 1
                    if multilang_count % 100000 == 0:
                        print(f"    Read {multilang_count:,} / {multilang_lines:,} lines")
        
        if multilang_count >= multilang_lines:
            break
    
    print(f"  Read {multilang_count:,} lines from multilang files")
    
    # Step 2: Read all lines from low-resource file
    if not os.path.exists(low_resource_file):
        raise FileNotFoundError(f"Low-resource file not found: {low_resource_file}")
    
    print(f"\nReading all lines from {low_resource_file}...")
    low_resource_count = 0
    
    with open(low_resource_file, "r", encoding="utf8") as f:
        for line in f:
            if line.strip():
                all_lines.append(line)
                low_resource_count += 1
                if low_resource_count % 100000 == 0:
                    print(f"  Read {low_resource_count:,} lines")
    
    print(f"  Read {low_resource_count:,} lines from low-resource file")
    print(f"  Total lines: {len(all_lines):,}")
    
    # Step 3: Shuffle
    print(f"\nShuffling {len(all_lines):,} lines...")
    random.shuffle(all_lines)
    
    # Step 4: Write to output files (rotate every 1M samples)
    print(f"\nWriting shuffled data to output files...")
    output_path = os.path.join(output_dir, "combined_train.jsonl")
    output_file = open(output_path, "w", encoding="utf8")
    file_counter = 1
    samples_per_file = 1_000_000
    
    for idx, line in enumerate(all_lines):
        output_file.write(line)
        
        if (idx + 1) % samples_per_file == 0:
            output_file.close()
            file_counter += 1
            output_path = os.path.join(output_dir, f"combined_train_{file_counter}.jsonl")
            output_file = open(output_path, "w", encoding="utf8")
            print(f"  Written {idx+1:,} lines, rotated to file {file_counter}")
    
    output_file.close()
    
    # Summary
    print(f"\n=== Processing Complete ===")
    print(f"Total lines written: {len(all_lines):,}")
    print(f"Output files: {file_counter}")
    print(f"\nMultilang (high-resource):")
    print(f"  Lines: {multilang_count:,}")
    print(f"\nLow-resource:")
    print(f"  Lines: {low_resource_count:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a combined and shuffled dataset from multilang and low-resource sources"
    )
    parser.add_argument("--multilang_dir", required=True,
                       help="Directory containing multilang_train_1.jsonl, multilang_train_2.jsonl, multilang_train_3.jsonl")
    parser.add_argument("--low_resource_file", required=True,
                       help="Path to low-resource combined multilang_train_1.jsonl file")
    parser.add_argument("--output_dir", required=True,
                       help="Output directory for the combined dataset")
    parser.add_argument("--multilang_lines", type=int, default=3_000_000,
                       help="Number of lines to read from multilang files (default: 3M)")
    
    args = parser.parse_args()

    preprocess_combined(
        args.multilang_dir,
        args.low_resource_file,
        args.output_dir,
        args.multilang_lines
    )
