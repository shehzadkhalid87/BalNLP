"""
Stopword removal functionality for Balochi text processing.

This module provides tools for handling stopwords in Balochi text, including
methods for loading stopwords from files, removing them from text, and managing
custom stopwords.
"""

import os
from pathlib import Path
from typing import List, Optional, Set


def load_stopwords(filepath: str) -> Set[str]:
    """
    Load stopwords from a text file.

    Args:
        filepath (str): Path to the stopwords file

    Returns:
        Set[str]: Set of stopwords

    The file should have one stopword per line. Lines starting with '#' are
    treated as comments and ignored.
    """
    stopwords = set()
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    # Remove inline comments if present
                    word = line.split("#")[0].strip()
                    if word:
                        stopwords.add(word)
        return stopwords
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Stopwords file not found at {filepath}. "
            "Please ensure the file exists."
        )


# Default stopwords path
_RESOURCES_DIR = Path(__file__).parent.parent / "resources"
_DEFAULT_STOPWORDS_PATH = _RESOURCES_DIR / "bal_stopwords" / "balochiStopwords.txt"
print(_RESOURCES_DIR)

# Load default stopwords
try:
    BALOCHI_STOPWORDS: Set[str] = load_stopwords(str(_DEFAULT_STOPWORDS_PATH))
except FileNotFoundError:
    BALOCHI_STOPWORDS = set()


class BalochiStopwordRemover:
    """Class for handling stopword removal in Balochi text."""

    def __init__(
        self,
        custom_stopwords: Optional[Set[str]] = None,
        stopwords_file: Optional[str] = None,
    ) -> None:
        """
        Initialize the stopword remover.

        Args:
            custom_stopwords (Optional[Set[str]]): Additional stopwords to include.
            stopwords_file (Optional[str]): Path to a custom stopwords file.
                If provided, these stopwords will be used instead of the default ones.
        """
        if stopwords_file:
            self.stopwords = load_stopwords(stopwords_file)
        else:
            self.stopwords = BALOCHI_STOPWORDS.copy()

        if custom_stopwords:
            self.stopwords.update(custom_stopwords)

    def is_stopword(self, word: str) -> bool:
        """Check if a word is a stopword."""
        return word.strip() in self.stopwords

    def remove_stopwords(self, text: str) -> str:
        """Remove stopwords from a text string."""
        words = text.split()
        filtered_words = [word for word in words if not self.is_stopword(word)]
        return " ".join(filtered_words)

    def remove_stopwords_from_list(self, words: List[str]) -> List[str]:
        """Remove stopwords from a list of words."""
        return [word for word in words if not self.is_stopword(word)]

    def add_stopwords(self, new_stopwords: Set[str]) -> None:
        """Add new stopwords to the existing set."""
        self.stopwords.update(new_stopwords)

    def remove_custom_stopwords(self, custom_stopwords: Set[str]) -> None:
        """Remove specific stopwords from the existing set."""
        self.stopwords.difference_update(custom_stopwords)

    def save_stopwords(self, filepath: str) -> None:
        """Save the current set of stopwords to a file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            for word in sorted(self.stopwords):
                f.write(f"{word}\n")
