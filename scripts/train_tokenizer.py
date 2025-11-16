import os
import json
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def train_tokenizer(corpus_file: str, save_dir: str, vocab_size: int = 30000):
    # Ensure save_dir exists
    os.makedirs(save_dir, exist_ok=True)

    # Initialize a BPE tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    # Trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    )

    # Train tokenizer
    tokenizer.train([corpus_file], trainer=trainer)

    # Save the tokenizer as JSON
    tokenizer_path = os.path.join(save_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    print(f"Tokenizer saved to {tokenizer_path}")

    # Optionally, save vocab.txt separately (for compatibility)
    vocab_file = os.path.join(save_dir, "vocab.txt")
    with open(vocab_file, "w", encoding="utf-8") as f:
        for token, _ in tokenizer.get_vocab().items():
            f.write(token + "\n")
    print(f"Vocabulary saved to {vocab_file}")

# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("corpus_file")
    parser.add_argument("save_dir")
    parser.add_argument("--vocab_size", type=int, default=30000)
    args = parser.parse_args()

    train_tokenizer(args.corpus_file, args.save_dir, args.vocab_size)
