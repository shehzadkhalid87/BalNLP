import os
import json
from src import BalochiTextNormalizer, BalochiWordTokenizer, BalochiSentenceTokenizer, BalochiTextCleaner
# Assuming you have a BalochiStopwordRemover or TextCleaner
from src.preprocessing.stopwords import BalochiStopwordRemover

input_file = "/home/python-dev/BalNLP/Data"
output_file = "data/balochi_sample_output.json"

# Ensure output directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Load raw text
with open(input_file, "r", encoding="utf-8") as f:
    text = f.read()

print("\n=== RAW TEXT ===")
print(text[:200], "...")  # show first 200 chars

# Step 1: Clean text (remove unwanted chars, normalize spacing)
cleaner = BalochiTextCleaner()
text_cleaned = cleaner.clean_text(text)

# Step 2: Remove stopwords (optional)
stopword_remover = BalochiStopwordRemover()
text_no_stopwords = stopword_remover.remove_stopwords(text_cleaned)

# Step 3: Normalize text
normalizer = BalochiTextNormalizer()
text_norm = normalizer.normalize_text(text_no_stopwords, remove_diacritics=True)

print("\n=== NORMALIZED TEXT ===")
print(text_norm[:200], "...")

# Step 4: Word tokenize
word_tokenizer = BalochiWordTokenizer()
words = word_tokenizer.tokenize(text_norm)

# Step 5: Sentence tokenize
sentence_tokenizer = BalochiSentenceTokenizer()
sentences = sentence_tokenizer.tokenize(text_norm)

# Save results to JSON
output = {
    "raw": text,
    "cleaned": text_cleaned,
    "no_stopwords": text_no_stopwords,
    "normalized": text_norm,
    "words": words,
    "sentences": sentences
}

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\nSaved processed data to {output_file}")
