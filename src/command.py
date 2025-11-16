import argparse
import json
import sys
from typing import Dict, Optional, Union

from src import BalochiTextNormalizer, BalochiWordTokenizer, BalochiSentenceTokenizer, BalochiTextCleaner


def read_text_file(file_path: str) -> str:
    """Read text from a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def write_output(output: Dict[str, Union[str, int, list]], output_file: Optional[str] = None) -> None:
    """Write output to file or stdout."""
    output_json = json.dumps(output, ensure_ascii=False, indent=2)
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output_json)
    else:
        print(output_json)


def preprocess_text(
        text: str,
        remove_urls: bool = True,
        remove_emails: bool = True,
        remove_numbers: bool = True,
        remove_emojis: bool = True,
        remove_special: bool = True,
        keep_chars: Optional[list] = None,
        remove_diacritics: bool = True
) -> Dict[str, Union[str, list, int]]:
    """Full preprocessing pipeline: clean, normalize, tokenize."""

    # 1️⃣ Clean
    cleaner = BalochiTextCleaner()
    cleaned_text = cleaner.clean_text(
        text,
        remove_urls=remove_urls,
        remove_emails=remove_emails,
        remove_numbers=remove_numbers,
        remove_emojis=remove_emojis,
        preserve_special_chars=not remove_special,
        keep_chars=keep_chars,
    )

    # 2️⃣ Normalize
    normalizer = BalochiTextNormalizer()
    normalized_text = normalizer.normalize_text(cleaned_text, remove_diacritics=remove_diacritics)

    # 3️⃣ Word tokenize
    word_tokenizer = BalochiWordTokenizer()
    words = word_tokenizer.tokenize(normalized_text)

    # 4️⃣ Sentence tokenize
    sentence_tokenizer = BalochiSentenceTokenizer()
    sentences = sentence_tokenizer.tokenize(normalized_text)

    return {
        "original_length": len(text),
        "cleaned_length": len(cleaned_text),
        "normalized_length": len(normalized_text),
        "cleaned_text": cleaned_text,
        "normalized_text": normalized_text,
        "words": words,
        "word_count": len(words),
        "sentences": sentences,
        "sentence_count": len(sentences),
    }


def main():
    parser = argparse.ArgumentParser(description="Full Balochi NLP preprocessing pipeline")
    parser.add_argument("input_file", help="Input text file path")
    parser.add_argument("--output", "-o", help="Output JSON file path")
    parser.add_argument("--remove-urls", action="store_true", help="Remove URLs from text")
    parser.add_argument("--remove-emails", action="store_true", help="Remove emails")
    parser.add_argument("--remove-numbers", action="store_true", help="Remove numbers")
    parser.add_argument("--remove-emojis", action="store_true", help="Remove emojis")
    parser.add_argument("--remove-special", action="store_true", help="Remove special characters")
    parser.add_argument("--keep-chars", help="Comma-separated special characters to keep")
    parser.add_argument("--remove-diacritics", action="store_true", help="Remove diacritics during normalization")
    args = parser.parse_args()

    try:
        text = read_text_file(args.input_file)
    except Exception as e:
        print(f"Error reading input file: {e}", file=sys.stderr)
        sys.exit(1)

    keep_chars = args.keep_chars.split(",") if args.keep_chars else None

    try:
        output = preprocess_text(
            text,
            remove_urls=args.remove_urls,
            remove_emails=args.remove_emails,
            remove_numbers=args.remove_numbers,
            remove_emojis=args.remove_emojis,
            remove_special=args.remove_special,
            keep_chars=keep_chars,
            remove_diacritics=args.remove_diacritics,
        )
    except Exception as e:
        print(f"Error during preprocessing: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        write_output(output, args.output)
    except Exception as e:
        print(f"Error writing output: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
