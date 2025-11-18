# scripts/build_corpus.py
import os
import json
from tqdm import tqdm

from balnlp.preprocessing.stopwords import BalochiStopwordRemover
from balnlp.preprocessing.text_cleaner import BalochiTextCleaner
from balnlp.preprocessing.text_normalizer import BalochiTextNormalizer


def build_corpus(input_dir: str, output_file: str, remove_stopwords=True):
    cleaner = BalochiTextCleaner()
    normalizer = BalochiTextNormalizer()
    stopwords = BalochiStopwordRemover() if remove_stopwords else None

    corpus = []
    files = [
        os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".txt")
    ]

    for file in tqdm(files, desc="Processing files"):
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()
        cleaned = cleaner.clean_text(text)
        normalized = normalizer.normalize_text(cleaned, remove_diacritics=True)
        tokens = normalized.split()  # list of tokens
        if remove_stopwords:
            tokens = stopwords.remove_stopwords_from_list(tokens)
        corpus.append(
            {"file": os.path.basename(file), "tokens": tokens, "text": " ".join(tokens)}
        )

    with open(output_file, "w", encoding="utf-8") as f:
        for entry in corpus:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Corpus saved to {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_file")
    parser.add_argument("--no-stopwords", action="store_true")
    args = parser.parse_args()

    build_corpus(
        args.input_dir, args.output_file, remove_stopwords=not args.no_stopwords
    )
