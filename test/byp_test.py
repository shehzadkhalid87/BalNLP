from balnlp import BalBPETokenizer

# 1. Training with different parameters
tokenizer = BalBPETokenizer(
    vocab_size=5000,  # Larger vocabulary
    special_tokens=["<pad>", "<unk>", "<s>", "</s>", "<mask>"],
)

# 2. Training on large corpus
with open("/test/data/cleaned_data.txt", "r", encoding="utf-8") as f:
    texts = [line.strip() for line in f if line.strip()]

tokenizer.train(texts, save_dir="data/balochi_bpe_tokenizer")


# 3. Using for LLM training
def prepare_llm_data(texts, tokenizer, max_length=512):
    """Prepare data for language model training."""
    encoded_data = []
    for text in texts:
        tokens = tokenizer.encode(text)
        if len(tokens) <= max_length:
            encoded_data.append(tokens)
        else:
            # Split long texts
            for i in range(0, len(tokens), max_length):
                chunk = tokens[i : i + max_length]
                encoded_data.append(chunk)
    return encoded_data


# 4. Batch encoding
def batch_encode(texts, tokenizer):
    """Encode multiple texts efficiently."""
    return [tokenizer.encode(text) for text in texts]


# 5. Vocabulary analysis
def analyze_vocabulary(tokenizer):
    """Analyze the learned vocabulary."""
    vocab = tokenizer.vocab
    print(f"Total vocabulary: {len(vocab)}")

    # Count special tokens
    special_count = sum(
        1 for token in vocab if token.startswith("<") and token.endswith(">")
    )
    print(f"Special tokens: {special_count}")

    # Count merged tokens (multi-character)
    merged_count = sum(
        1 for token in vocab if len(token) > 1 and not token.startswith("<")
    )
    print(f"Merged tokens: {merged_count}")

    # Show some merged tokens
    merged_tokens = [
        token for token in vocab if len(token) > 1 and not token.startswith("<")
    ]
    print(f"Sample merged tokens: {merged_tokens[:10]}")


# Usage
analyze_vocabulary(tokenizer)
