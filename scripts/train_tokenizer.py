import sys
import os
from pathlib import Path

# Setup path to find 'balnlp'
current_path = Path(__file__).resolve().parent.parent
sys.path.append(str(current_path))

from balnlp.bal_tokenizer.sentencepiece_tokenizer import BalSentencePieceTokenizer


def main():
    # --- CORRECT PATHS FOR YOUR STRUCTURE ---
    # Input: The clean text you just built
    INPUT_CORPUS = current_path / "corpus" / "balochi_corpus.txt"

    # Output: Where to save the tokenizer model
    MODEL_DIR = current_path / "models" / "tokenizer"
    MODEL_PREFIX = str(MODEL_DIR / "balochi_bpe")

    # Check if corpus exists
    if not INPUT_CORPUS.exists():
        print(f"❌ Error: Corpus not found at {INPUT_CORPUS}")
        return

    # Create output folder
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"🚀 Training Tokenizer on: {INPUT_CORPUS}")

    tokenizer = BalSentencePieceTokenizer()

    # Train (Vocabulary = 32000)
    tokenizer.train(
        input_file=str(INPUT_CORPUS),
        model_prefix=MODEL_PREFIX,
        vocab_size=32000,
        model_type="bpe"
    )

    print(f"✅ Tokenizer Saved to: {MODEL_PREFIX}.model")


if __name__ == "__main__":
    main()