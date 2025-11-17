# BalNLP

**BalNLP** is a comprehensive Natural Language Processing toolkit for the Balochi language. It provides utilities for text preprocessing, normalization, cleaning, and tokenization at the word and sentence level, making it easier to work with Balochi text for research and development purposes.

---

## Features

- **Text Normalization** – Remove diacritics, standardize text, handle orthographic variations.
- **Text Cleaning** – Remove numbers, special characters, URLs, emails, emojis, and optionally keep specific characters.
- **Word Tokenization** – Split Balochi text into words, handling prefixes, suffixes, and clitics.
- **Sentence Tokenization** – Detect sentence boundaries including Balochi-specific punctuation and abbreviations.
- **Stopword Removal** – Remove common Balochi stopwords (planned/optional).

---

## Installation

You can install the package from PyPI (if published) or locally via source:

### From PyPI
```bash
pip install BalNlP
```
## From Source
```bash
git clone https://github.com/shehzadkhalid87/BalNLP
```
```bash
cd src

pip install -e .

pip install -r requirements-dev.txt
```

## Usage
## Basic Usage

from src.preprocessing.normalizer import BalochiTextNormalizer
from src.preprocessing.cleaner import BalochiTextCleaner

text = "شمارا تھنا بلوچی وت وانگ ءُ دربرگی نہ اِنت بلکیں شمارا راج ءِ نودربراں ھم وانینگی اِنت۔ وھدے کہ ما دْراہ راجی زُبان ءِ ھزمتکار ءُ اُستاد بہ ایں گُڑا بلوچی دیمروی کنت۔"

# Normalize text
normalizer = BalochiTextNormalizer()
normalized_text = normalizer.normalize(text, remove_diacritics=True)

# Clean text
cleaner = BalochiTextCleaner()
cleaned_text = cleaner.clean_text(normalized_text, remove_numbers=True)

# Word Tokenization
from src.bal_tokenizer import BalochiWordTokenizer

word_tokenizer = BalochiWordTokenizer()
tokens = word_tokenizer.tokenize(cleaned_text)
print(tokens)

# Full Pipeline Example
import json
from balochi_nlp.preprocessing.normalizer import BalochiTextNormalizer
from balochi_nlp.preprocessing.cleaner import BalochiTextCleaner
from balochi_nlp.tokenizers.word_tokenizer import BalochiWordTokenizer
from balochi_nlp.tokenizers.sentence_tokenizer import BalochiSentenceTokenizer

text = "data"

# Normalize
normalizer = BalochiTextNormalizer()
text_norm = normalizer.normalize(text, remove_diacritics=True)

# Clean
cleaner = BalochiTextCleaner()
text_clean = cleaner.clean_text(text_norm, remove_numbers=True)

# Tokenize
word_tokenizer = BalochiWordTokenizer()
words = word_tokenizer.tokenize(text_clean)

sentence_tokenizer = BalochiSentenceTokenizer()
sentences = sentence_tokenizer.tokenize(text_clean)

# Save output
output = {
    "normalized": text_norm,
    "cleaned": text_clean,
    "words": words,
    "sentences": sentences
}

with open("balochi_output.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

# Command Line Interface
python command.py <input_file> --task <task_name> [options]

# Example:
python command.py data/balochi_sample.txt --task normalize --remove-diacritics -o output.json

# Testing
pytest tests

# Code Quality
black src tests
isort src tests
flake8 src tests
mypy src

# Pre-commit Hooks
pre-commit install
pre-commit run --all-files

# Contributing
Fork the repository

Create a new branch: git checkout -b feature/your-feature

Make changes and add tests

```Ensure linting passes and run pytest

Commit your changes: git commit -m "Add your feature"

Push to branch: git push origin feature/your-feature

Create a Pull Reques```t

# License
This project is licensed under the MIT License. See the LICENSE file for details.

# Authors
Shehzad Khalid – Email: shehzadkhalid04@gmail.com