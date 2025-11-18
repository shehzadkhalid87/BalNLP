import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from balnlp.bal_tokenizer.word_tokenizer import BalochiWordTokenizer


# your rule-based tokenizer


# ---------------------------------------------------------
# Step 1: Tokenize your corpus with your own tokenizer
# ---------------------------------------------------------
def preprocess_corpus(input_file: str, temp_file: str):
    tokenizer = BalochiWordTokenizer()

    with open(input_file, "r", encoding="utf-8") as f, open(
        temp_file, "w", encoding="utf-8"
    ) as out:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens_info = tokenizer.tokenize_with_affixes(line)
            # Join tokens with spaces for BPE training
            tokens = [t["token"] for t in tokens_info]
            out.write(" ".join(tokens) + "\n")
    print(f"Preprocessed corpus saved to {temp_file}")


# ---------------------------------------------------------
# Step 2: Train a BPE tokenizer on pre-tokenized corpus
# ---------------------------------------------------------
def train_bpe_tokenizer(preprocessed_file: str, save_dir: str, vocab_size: int = 300):
    os.makedirs(save_dir, exist_ok=True)

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    )

    tokenizer.train([preprocessed_file], trainer=trainer)

    # Save tokenizer
    tokenizer_path = os.path.join(save_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    print(f"BPE Tokenizer saved to {tokenizer_path}")


# ---------------------------------------------------------
# Example usage
# ---------------------------------------------------------
if __name__ == "__main__":
    input_file = "/home/python-dev/BalNLP/data/data.txt"  # original text corpus
    temp_file = (
        "/home/python-dev/BalNLP/data/balochi_corpus_pre.txt"  # pre-tokenized for BPE
    )
    save_dir = "tokenizer"

    preprocess_corpus(input_file, temp_file)
    train_bpe_tokenizer(
        temp_file, save_dir, vocab_size=500
    )  # you can adjust vocab_size
