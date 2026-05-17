# python preprocess_raw.py --language uk --raw_dir /low_resource/raw_data --output_dir /low_resource
# python preprocess_raw.py --language ur --raw_dir /low_resource/raw_data --output_dir /low_resource
# python preprocess_raw.py --language sk --raw_dir /low_resource/raw_data --output_dir /low_resource
# python preprocess_raw.py --language ca --raw_dir /low_resource/raw_data --output_dir /low_resource
# python preprocess_raw.py --language et --raw_dir /low_resource/raw_data --output_dir /low_resource
# python preprocess_raw.py --language mr --raw_dir /low_resource/raw_data --output_dir /low_resource
# python preprocess_raw.py --language uk --raw_dir /low_resource/raw_data --output_dir /low_resource
# python preprocess_raw.py --language nl --raw_dir /low_resource/raw_data --output_dir /low_resource
# python preprocess_raw.py --language uk --raw_dir /low_resource/raw_data --output_dir /low_resource
# python preprocess_raw.py --language uk --raw_dir /low_resource/raw_data --output_dir /low_resource

# python preprocess_raw.py --language ur --raw_dir /low_resource/raw_data --output_dir /low_resource
# python preprocess_raw.py --language sk --raw_dir /low_resource/raw_data --output_dir /low_resource
# python preprocess_raw.py --language ca --raw_dir /low_resource/raw_data --output_dir /low_resource
# python preprocess_raw.py --language et --raw_dir /low_resource/raw_data --output_dir /low_resource
# python preprocess_raw.py --language mr --raw_dir /low_resource/raw_data --output_dir /low_resource
# python preprocess_raw.py --language nl --raw_dir /low_resource/raw_data --output_dir /low_resource
# python preprocess_raw.py --language uk --raw_dir /low_resource/raw_data --output_dir /low_resource




import argparse
import json
import os

def load_raw(path):
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            yield json.loads(line)

def create_doc(example, language, idx):
    return {
        "id": f"{language}_{idx}",
        "text": example["text"],
        "source": example["source"],
        "timestamp": example["timestamp"],
        "url": example["url"],
        "metadata": {},
        "language": language
    }

def preprocess(language, raw_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    raw_iter = load_raw(os.path.join(raw_dir, f"raw_{language}.jsonl"))

    # Save first 5000 samples as test
    test_path = os.path.join(output_dir, f"{language}_test.jsonl")
    with open(test_path, "w", encoding="utf8") as f:
        for idx in range(5000):
            try:
                ex = next(raw_iter)
                doc = create_doc(ex, language, idx)
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            except StopIteration:
                print(f"Warning: Less than 5000 samples available for {language}")
                break
    print(f"Saved test set for {language} to {test_path}")

    # Save remaining samples to train with rotation every 1M samples
    idx = 0
    file_counter = 1
    train_path = os.path.join(output_dir, f"{language}_train_{file_counter}.jsonl")
    train_file = open(train_path, "w", encoding="utf8")

    while True:
        try:
            ex = next(raw_iter)
        except StopIteration:
            train_file.close()
            print(f"Saved training data across {file_counter} file(s) to {output_dir}")
            return

        doc = create_doc(ex, language, idx)
        train_file.write(json.dumps(doc, ensure_ascii=False) + "\n")
        idx += 1

        # rotate every 1M samples
        if idx % 1000000 == 0:
            train_file.close()
            file_counter += 1
            train_path = os.path.join(output_dir, f"{language}_train_{file_counter}.jsonl")
            train_file = open(train_path, "w", encoding="utf8")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", required=True)
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    preprocess(args.language, args.raw_dir, args.output_dir)
