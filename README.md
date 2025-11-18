# BalNLP: Balochi Natural Language Processing Toolkit


BalNLP is a comprehensive Natural Language Processing toolkit specifically designed for the Balochi language. It provides utilities for text processing, normalization, cleaning, tokenization, and deduplication to support research and development in Balochi language technologies.

## 🌟 Features

### 📝 Text Processing
- Text Normalization – Remove diacritics, standardize orthographic variations
- Text Cleaning – Remove numbers, special characters, URLs, emails, emojis
- Stopword Removal – Remove common Balochi stopwords

### 🔤 Tokenization
- Word Tokenization – Split Balochi text into words with clitic handling
- Sentence Tokenization – Detect sentence boundaries with Balochi-specific rules
- BPE Tokenization – Byte Pair Encoding for LLM training
- SentencePiece Tokenization – Advanced subword tokenization

### 🧹 Deduplication
- Exact Deduplication – Remove byte-identical documents
- Near Deduplication – Remove semantically similar documents using Jaccard similarity
- Repetition Removal – Remove documents with repetitive patterns

### 🛠️ Utilities
- Unicode Normalization – Handle Balochi script variations
- File Utilities – Read/write Balochi text files with proper encoding
- Command Line Interface – Easy-to-use CLI for batch processing

---

## 🚀 Quick Start

### Installation

#### Method 1: Install from PyPI
```bash
pip install BalNLP
```

#### Method 2: Install from Source
```bash
# Clone the repository
git clone https://github.com/shehzadkhalid87/BalNLP
cd BalNLP

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt
```

### Basic Usage
```python
from balnlp import BalochiTextNormalizer, BalochiTextCleaner
from balnlp.bal_tokenizer.word_tokenizer import BalochiWordTokenizer
from balnlp.bal_tokenizer.sentence_tokenizer import BalochiSentenceTokenizer

# Sample Balochi text
text = "شمارا تھنا بلوچی وت وانگ ءُ دربرگی نہ اِنت بلکیں شمارا راج ءِ نودربراں ھم وانینگی اِنت۔"

# Initialize processors
normalizer = BalochiTextNormalizer()
cleaner = BalochiTextCleaner()
word_tokenizer = BalochiWordTokenizer()
sentence_tokenizer = BalochiSentenceTokenizer()

# Process text
normalized_text = normalizer.normalize(text, remove_diacritics=True)
cleaned_text = cleaner.clean_text(normalized_text, remove_numbers=True)
words = word_tokenizer.tokenize(cleaned_text)
sentences = sentence_tokenizer.tokenize(cleaned_text)

print("Words:", words)
print("Sentences:", sentences)
```

## 📚 Step-by-Step Guide

### Step 1: Basic Text Processing
```python
from balnlp import BalochiTextNormalizer, BalochiTextCleaner

text = "Your Balochi text here..."

# Normalize text (remove diacritics, standardize characters)
normalizer = BalochiTextNormalizer()
normalized = normalizer.normalize(text, remove_diacritics=True)

# Clean text (remove numbers, special characters)
cleaner = BalochiTextCleaner()
cleaned = cleaner.clean_text(normalized, remove_numbers=True)

print(f"Original: {text}")
print(f"Cleaned: {cleaned}")
```

### Step 2: Tokenization
```python
from balnlp.bal_tokenizer.word_tokenizer import BalochiWordTokenizer
from balnlp.bal_tokenizer.sentence_tokenizer import BalochiSentenceTokenizer

text = "من بلوچے آں. "

# Word tokenization
word_tokenizer = BalochiWordTokenizer()
words = word_tokenizer.tokenize(text)
print(f"Words: {words}")

# Sentence tokenization
sentence_tokenizer = BalochiSentenceTokenizer()
sentences = sentence_tokenizer.tokenize(text)
print(f"Sentences: {sentences}")
```

### Step 3: Deduplication
```python
from balnlp.dedup.exact_dedup import ExactDedup
from balnlp.dedup.near_dedup import NearDedup

documents = [
    "من بلوچے آں",
    "بلوچی زبان زندگ بات",
   "من بلوچے آں"
]

# Remove exact duplicates
exact_dedup = ExactDedup()
unique_docs = exact_dedup.remove_exact_duplicates(documents)
print(f"Exact dedup: {len(unique_docs)} documents")

# Remove near duplicates
near_dedup = NearDedup(threshold=0.4)
near_unique = near_dedup.remove_near_duplicates(documents)
print(f"Near dedup: {len(near_unique)} documents")
```

### Step 4: BPE Tokenization for LLM Training
```python
from balnlp import BalBPETokenizer

# Sample Balochi corpus
texts = [
    "من بلوچے آں",
    "بلوچی زبان زندگ بات"
]

# Train BPE tokenizer
tokenizer = BalBPETokenizer(vocab_size=5000)
tokenizer.train(texts, save_dir="balochi_tokenizer")

# Encode/decode text
encoded = tokenizer.encode("من بلوچے آں")
decoded = tokenizer.decode(encoded)

print(f"Encoded: {encoded}")
print(f"Decoded: {decoded}")
```

### Step 5: Processing Files
```python
from balnlp.utils.
from balnlp import BalochiTextNormalizer, BalBPETokenizer

# Read Balochi text file
lines = read_balochi_file("data/balochi_corpus.txt")

# Process each line
normalizer = BalochiTextNormalizer()
tokenizer = BalBPETokenizer()

processed_data = []
for line in lines:
    normalized = normalizer.normalize(line)
    tokens = tokenizer.tokenize(normalized)
    processed_data.append({
        'original': line,
        'normalized': normalized,
        'tokens': tokens
    })

# Save processed data
write_balochi_file("processed_corpus.json", processed_data)
```

## 🔧 Advanced Usage

### Custom Pipeline

```python
from balnlp import (
    BalochiTextNormalizer,
    BalochiTextCleaner,
    NearDedup,
    BalochiWordTokenizer,
    BalochiSentenceTokenizer
)


class BalochiTextPipeline:
    def __init__(self):
        self.normalizer = BalochiTextNormalizer()
        self.cleaner = BalochiTextCleaner()
        self.word_tokenizer = BalochiWordTokenizer()
        self.sentence_tokenizer = BalochiSentenceTokenizer()
        self.dedup = NearDedup(threshold=0.4)

    def process_corpus(self, texts):
        # Normalize and clean
        processed = [
            self.cleaner.clean_text(
                self.normalizer.normalize(text)
            )
            for text in texts
        ]

        # Remove duplicates
        unique_texts = self.dedup.remove_near_duplicates(processed)

        # Tokenize
        results = []
        for text in unique_texts:
            results.append({
                'text': text,
                'words': self.word_tokenizer.tokenize(text),
                'sentences': self.sentence_tokenizer.tokenize(text)
            })

        return results


# Usage
pipeline = BalochiTextPipeline()
texts = ["Your Balochi texts..."]
results = pipeline.process_corpus(texts)
```

### Training Custom Tokenizer
```python
from balnlp import BalBPETokenizer

# Load your Balochi corpus
with open('large_balochi_corpus.txt', 'r', encoding='utf-8') as f:
    texts = [line.strip() for line in f if line.strip()]

# Train with custom parameters
tokenizer = BalBPETokenizer(
    vocab_size=10000,
    special_tokens=["<pad>", "<unk>", "<s>", "</s>", "<mask>"]
)

tokenizer.train(texts, save_dir="custom_balochi_tokenizer")

# Use for LLM training
def prepare_training_data(texts, tokenizer, max_length=512):
    encoded_data = []
    for text in texts:
        tokens = tokenizer.encode(text)
        if len(tokens) <= max_length:
            encoded_data.append(tokens)
    return encoded_data

training_data = prepare_training_data(texts, tokenizer)
```

## 💻 Command Line Interface

### Process Single File
```bash
# Normalize text
balnlp normalize input.txt --output normalized.txt --remove-diacritics

# Clean text
balnlp clean input.txt --output cleaned.txt --remove-numbers

# Tokenize text
balnlp tokenize input.txt --output tokens.json --type word

# Remove duplicates
balnlp dedup input.txt --output unique.txt --type near --threshold 0.8
```

### Batch Processing
```bash
# Process entire directory
balnlp batch-process ./data/ --output ./processed/ --tasks normalize clean tokenize

# Train tokenizer on corpus
balnlp train-tokenizer ./corpus/ --output ./tokenizer/ --vocab-size 5000
```

### Available Commands
- `normalize` — Normalize Balochi text
  - Options: `--remove-diacritics`, `--output <file>`
- `clean` — Clean Balochi text
  - Options: `--remove-numbers`, `--keep-chars "<chars>"`
- `tokenize` — Tokenize text
  - Options: `--type {word|sentence}`, `--output <file>`
- `dedup` — Remove duplicates
  - Options: `--type {exact|near}`, `--threshold <float>`
- `train-tokenizer` — Train BPE tokenizer
  - Options: `--vocab-size <int>`, `--output <dir>`

## 🧪 Testing

### Run Tests
```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/test_tokenizer.py

# Run with coverage
pytest --cov=balnlp tests/
```

### Code Quality
```bash
# Format code
black balnlp/ tests/

# Sort imports
isort balnlp/ tests/

# Lint code
flake8 balnlp/ tests/

# Type checking
mypy balnlp/
```

### Pre-commit Hooks
```bash
# Install pre-commit hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```

## 📁 Project Structure
```text
BalNLP/
├── balnlp/                    # Main package
│   ├── bal_tokenizer/         # Word and sentence tokenizers
│   ├── dedup/                 # Deduplication modules
│   ├── preprocessing/         # Text normalization and cleaning
│   ├── tokenizers/            # BPE and SentencePiece tokenizers
│   ├── resources/             # Balochi stopwords and resources
│   └── utils/                 # Utility functions
├── tests/                     # Test suite
├── data/                      # Sample data and corpora
├── scripts/                   # Training and build scripts
└── docs/                      # Documentation
```

## 🤝 Contributing
We welcome contributions! Please see our Contributing Guide for details.

### Development Setup
```bash
# 1. Fork and clone the repository
git clone https://github.com/your-username/BalNLP
cd BalNLP

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install in development mode
pip install -e .[dev]

# 4. Run tests to verify setup
pytest

# 5. Make your changes and add tests

# 6. Ensure code quality
black balnlp tests
isort balnlp tests
flake8 balnlp tests

# 7. Submit a Pull Request
```

### Reporting Issues
Please report bugs and feature requests on the GitHub Issues page.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors
- Shehzad Khalid — Creator & Maintainer — shehzadkhalid87
- Email: shehzadkhalido4@gmail.com

## 🙏 Acknowledgments
- Balochi language community and contributors
- Researchers working on low-resource language technologies
- Open-source NLP community

## 📞 Support
If you need help or have questions:
- Check the documentation
- Open an issue
- Contact: shehzadkhalido4@gmail.com

## 🌐 Links
- GitHub: https://github.com/shehzadkhalid87/BalNLP
- PyPI: https://pypi.org/project/BalNLP/
- Documentation: https://balnlp.readthedocs.io/

BalNLP — Empowering Balochi Language Technology 🚀
