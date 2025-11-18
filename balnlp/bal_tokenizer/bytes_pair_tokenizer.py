import json
import os
from typing import List, Dict, Optional
from collections import Counter, defaultdict


class BalBPETokenizer:
    """
    Byte Pair Encoding (BPE) tokenizer for Balochi text.
    """

    def __init__(self, vocab_size: int = 5000, special_tokens: Optional[List[str]] = None):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or ["<pad>", "<unk>", "<s>", "</s>"]
        self.vocab = {}
        self.merges = {}
        self.inverse_vocab = {}
        self.trained = False

    def train(self, texts: List[str], save_dir: Optional[str] = None):
        """Train BPE tokenizer on Balochi text."""
        print("Starting BPE training...")

        # Extract all words with frequencies
        word_freqs = self._extract_word_frequencies(texts)
        print(f"Extracted {len(word_freqs)} unique words")

        # Initialize vocabulary with characters
        initial_vocab = self._initialize_vocab(word_freqs)
        print(f"Initial vocabulary size: {len(initial_vocab)}")

        # Learn BPE merges
        self.merges = self._learn_bpe(word_freqs)
        print(f"Learned {len(self.merges)} merges")

        # Build final vocabulary
        self._build_final_vocab()
        print(f"Final vocabulary size: {len(self.vocab)}")

        self.trained = True

        # Save if directory provided
        if save_dir:
            self.save(save_dir)
            print(f"Tokenizer saved to {save_dir}")

    def _extract_word_frequencies(self, texts: List[str]) -> Counter:
        """Extract words with their frequencies."""
        word_freqs = Counter()
        for text in texts:
            # Simple word splitting for Balochi
            words = text.split()
            for word in words:
                if word.strip():  # Only non-empty words
                    word_freqs[word] += 1
        return word_freqs

    def _initialize_vocab(self, word_freqs: Counter) -> Dict[str, int]:
        """Initialize vocabulary with individual characters."""
        vocab = {}
        chars = set()

        # Collect all unique characters
        for word in word_freqs:
            for char in word:
                chars.add(char)

        # Add characters to vocabulary
        for idx, char in enumerate(sorted(chars)):
            vocab[char] = idx

        # Add space as a special character
        vocab[' '] = len(vocab)

        return vocab

    def _learn_bpe(self, word_freqs: Counter, num_merges: int = None) -> Dict[tuple, int]:
        """Learn BPE merges."""
        if num_merges is None:
            num_merges = self.vocab_size - len(self.special_tokens) - 100  # Reserve space for chars

        vocab = {word: list(word) for word in word_freqs}
        merges = {}
        merge_count = 0

        for i in range(num_merges):
            # Count frequency of all pairs
            pair_freqs = Counter()
            for word, tokens in vocab.items():
                freq = word_freqs[word]
                for j in range(len(tokens) - 1):
                    pair = (tokens[j], tokens[j + 1])
                    pair_freqs[pair] += freq

            if not pair_freqs:
                break

            # Find most frequent pair
            best_pair = max(pair_freqs, key=pair_freqs.get)

            # Add to merges
            merges[best_pair] = merge_count
            merge_count += 1

            # Merge this pair in all words
            new_vocab = {}
            for word, tokens in vocab.items():
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                        new_tokens.append(tokens[i] + tokens[i + 1])
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                new_vocab[word] = new_tokens

            vocab = new_vocab

        return merges

    def _build_final_vocab(self):
        """Build final vocabulary dictionary."""
        self.vocab = {}

        # Add special tokens first
        for idx, token in enumerate(self.special_tokens):
            self.vocab[token] = idx

        current_idx = len(self.special_tokens)

        # Add individual characters first
        all_chars = set()
        for pair in self.merges.keys():
            all_chars.add(pair[0])
            all_chars.add(pair[1])

        for char in sorted(all_chars):
            if char not in self.vocab:
                self.vocab[char] = current_idx
                current_idx += 1

        # Add merged tokens
        for pair, _ in self.merges.items():
            merged_token = pair[0] + pair[1]
            if merged_token not in self.vocab:
                self.vocab[merged_token] = current_idx
                current_idx += 1

        # Add space if not already there
        if ' ' not in self.vocab:
            self.vocab[' '] = current_idx
            current_idx += 1

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        if not self.trained:
            raise ValueError("Tokenizer not trained. Call train() first.")

        # Split into words
        words = text.split()
        token_ids = []

        for i, word in enumerate(words):
            # Tokenize each word
            word_tokens = self._tokenize_word(word)
            token_ids.extend(word_tokens)

            # Add space between words (except after last word)
            if i < len(words) - 1:
                token_ids.append(self.vocab.get(' ', self.vocab['<unk>']))

        return token_ids

    def _tokenize_word(self, word: str) -> List[int]:
        """Tokenize a single word using BPE merges."""
        if not word:
            return []

        # Start with individual characters
        tokens = list(word)

        # Keep applying merges until no more merges possible
        changed = True
        while changed and len(tokens) > 1:
            changed = False
            new_tokens = []
            i = 0

            while i < len(tokens):
                if i < len(tokens) - 1:
                    pair = (tokens[i], tokens[i + 1])
                    if pair in self.merges:
                        # Merge the pair
                        new_tokens.append(pair[0] + pair[1])
                        i += 2
                        changed = True
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

        # Convert to token IDs
        token_ids = []
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                # Try to handle unknown tokens by splitting further
                for char in token:
                    if char in self.vocab:
                        token_ids.append(self.vocab[char])
                    else:
                        token_ids.append(self.vocab['<unk>'])

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        if not self.trained:
            raise ValueError("Tokenizer not trained. Call train() first.")

        tokens = []
        for token_id in token_ids:
            if token_id in self.inverse_vocab:
                token = self.inverse_vocab[token_id]
                # Skip special tokens except space
                if token in self.special_tokens and token != ' ':
                    continue
                tokens.append(token)
            else:
                tokens.append('?')

        return ''.join(tokens)

    def save(self, save_dir: str):
        """Save tokenizer files."""
        os.makedirs(save_dir, exist_ok=True)

        # Save tokenizer config
        config = {
            "vocab_size": len(self.vocab),
            "special_tokens": self.special_tokens,
            "trained": self.trained
        }

        with open(os.path.join(save_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        # Save vocabulary
        with open(os.path.join(save_dir, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

        # Save merges
        merges_list = [f"{pair[0]} {pair[1]}" for pair in self.merges.keys()]
        with open(os.path.join(save_dir, "merges.txt"), "w", encoding="utf-8") as f:
            f.write('\n'.join(merges_list))

    def load(self, save_dir: str):
        """Load tokenizer from files."""
        # Load config
        with open(os.path.join(save_dir, "tokenizer_config.json"), "r", encoding="utf-8") as f:
            config = json.load(f)

        # Load vocabulary
        with open(os.path.join(save_dir, "vocab.json"), "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

        # Convert string keys to int for inverse vocab
        self.vocab = {k: int(v) for k, v in self.vocab.items()}
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

        # Load merges
        self.merges = {}
        if os.path.exists(os.path.join(save_dir, "merges.txt")):
            with open(os.path.join(save_dir, "merges.txt"), "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 2:
                            self.merges[(parts[0], parts[1])] = idx

        self.special_tokens = config["special_tokens"]
        self.trained = config.get("trained", True)

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text and return tokens (for debugging)."""
        token_ids = self.encode(text)
        return [self.inverse_vocab.get(token_id, "<unk>") for token_id in token_ids]