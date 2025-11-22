import os
import sys
import glob


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from balnlp.preprocessing.cleaner import BalochiTextCleaner
from balnlp.preprocessing.normalizer import BalochiTextNormalizer
from balnlp.dedup.exact import ExactDeduplicator
from balnlp.dedup.minhash import NearDeduplicator

# ==========================
# SETTINGS
# ==========================
INPUT_DIR = "/home/python-dev/BalNLP/data"
OUTPUT_PATH = "/home/python-dev/BalNLP/corpus/balochi_corpus.txt"
USE_NEAR_DEDUP = True


def main():
    print(f">>> Initializing Advanced Pipeline...")

    # 1. Init Components
    cleaner = BalochiTextCleaner()
    normalizer = BalochiTextNormalizer()
    exact_dedup = ExactDeduplicator()

    near_dedup = None
    if USE_NEAR_DEDUP:
        print(">>> Initializing LSH (MinHash) for Near-Deduplication...")
        near_dedup = NearDeduplicator(threshold=0.85)

    total_count = 0
    saved_count = 0
    dropped_exact = 0
    dropped_near = 0

    # 2. Find all .txt files in the directory
    # This will find raw_data.txt, tbp_nebeshtank.txt, etc.
    input_files = glob.glob(os.path.join(INPUT_DIR, "*.txt"))

    # Filter out the output file if it already exists in the same folder to avoid loop
    input_files = [f for f in input_files if os.path.basename(f) != os.path.basename(OUTPUT_PATH)]

    if not input_files:
        print(f"ERROR: No .txt files found in {INPUT_DIR}")
        return

    print(f">>> Found {len(input_files)} files: {[os.path.basename(f) for f in input_files]}")

    # 3. Open Output File ONCE
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f_out:

        # 4. Loop through each input file
        for file_path in input_files:
            print(f"    -> Processing file: {os.path.basename(file_path)}")

            try:
                with open(file_path, 'r', encoding='utf-8') as f_in:
                    for line in f_in:
                        total_count += 1
                        original_line = line.strip()

                        # --- STAGE 1: CLEANING ---
                        cleaned_text = cleaner.clean_text(
                            original_line,
                            remove_english=True,
                            remove_numbers=True
                        )
                        if cleaned_text is None:
                            continue  # Garbage detected

                        # --- STAGE 2: EXACT DEDUPLICATION ---
                        # (Deduplication works across ALL files because exact_dedup is outside the loop)
                        if not exact_dedup.process_single(cleaned_text):
                            dropped_exact += 1
                            continue

                        # --- STAGE 3: NEAR DEDUPLICATION ---
                        if near_dedup:
                            if not near_dedup.process_single(cleaned_text):
                                dropped_near += 1
                                continue

                        # --- STAGE 4: NORMALIZATION ---
                        final_text = normalizer.normalize(cleaned_text)

                        if len(final_text.split()) < 2:
                            continue

                        # --- SAVE ---
                        f_out.write(final_text + "\n")
                        saved_count += 1

                        if total_count % 1000 == 0:
                            print(f"       Processed {total_count} lines total...")

            except Exception as e:
                print(f"    [WARNING] Could not read file {file_path}: {e}")

    print("=" * 40)
    print(f"PIPELINE COMPLETE")
    print(f"Total Lines Processed: {total_count}")
    print(f"Exact Duplicates Removed: {dropped_exact}")
    print(f"Near Duplicates Removed:  {dropped_near}")
    print(f"Final Clean Lines:      {saved_count}")
    print(f"Saved to:               {OUTPUT_PATH}")
    print("=" * 40)


if __name__ == "__main__":
    main()