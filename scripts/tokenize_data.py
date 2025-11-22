import sys
import os
import numpy as np
from pathlib import Path

current_path = Path(__file__).resolve().parent.parent
sys.path.append(str(current_path))

from balnlp.bal_tokenizer.sentencepiece_tokenizer import BalSentencePieceTokenizer


def main():
    # --- CORRECT PATHS ---
    INPUT_CORPUS = current_path / "corpus" / "balochi_corpus.txt"
    TOKENIZER_MODEL = current_path / "models" / "tokenizer" / "balochi_bpe.model"
    OUTPUT_DATA = current_path / "data" / "balochi_training_data.npy"

    if not TOKENIZER_MODEL.exists():
        print("❌ Tokenizer not found! Run Step 1 (train_tokenizer.py) first.")
        return

    print(">>> Loading Tokenizer...")
    tokenizer = BalSentencePieceTokenizer(str(TOKENIZER_MODEL))

    print(f">>> Reading Text: {INPUT_CORPUS}")
    with open(INPUT_CORPUS, "r", encoding="utf-8") as f:
        lines = f.readlines()

    all_tokens = []
    print(f">>> Converting Text to Numbers...")

    for i, line in enumerate(lines):
        line = line.strip()
        if not line: continue

        # Add EOS (End of Sentence) ID
        ids = tokenizer.encode(line) + [tokenizer.eos_id]
        all_tokens.extend(ids)

        if i % 5000 == 0:
            print(f"    Processed {i} lines...", end="\r")

    # Save as highly compressed Numpy file
    data_array = np.array(all_tokens, dtype=np.uint16)
    np.save(OUTPUT_DATA, data_array)

    print(f"\n✅ DATASET READY: {OUTPUT_DATA}")
    print(f"   Total Tokens: {len(data_array)}")


if __name__ == "__main__":
    main()