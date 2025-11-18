from typing import Dict, List


class CharLevelDedup:
    """Remove documents with repetitive character patterns."""

    def __init__(self, repeat_threshold: float = 0.6, min_length: int = 10):
        self.repeat_threshold = repeat_threshold
        self.min_length = min_length

    def has_repetitive_pattern(self, text: str) -> bool:
        if len(text) < self.min_length:
            return False

        char_counts: Dict[str, int] = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1

        max_char_ratio = max(char_counts.values()) / len(text)
        return max_char_ratio > self.repeat_threshold

    def remove_repetitive_docs(self, documents: List[str]) -> List[str]:
        return [doc for doc in documents if not self.has_repetitive_pattern(doc)]


class WordRepetitionDedup:
    """Remove documents with repetitive word patterns."""

    def __init__(self, repeat_threshold: float = 0.5, min_words: int = 5):
        self.repeat_threshold = repeat_threshold
        self.min_words = min_words

    def has_repetitive_words(self, text: str) -> bool:
        words = text.split()
        if len(words) < self.min_words:
            return False

        word_counts: Dict[str, int] = {}  # ← FIX: Add type annotation
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

        unique_words = len(word_counts)
        total_words = len(words)
        repetition_ratio = 1 - (unique_words / total_words)

        # FIX: Ensure explicit boolean return
        return bool(repetition_ratio > self.repeat_threshold)

    def remove_repetitive_docs(self, documents: List[str]) -> List[str]:
        return [doc for doc in documents if not self.has_repetitive_words(doc)]


class RepetitionDedup:
    """Combined repetitive text cleaner for Balochi."""

    def __init__(
        self, char_repeat_threshold: float = 0.7, word_repeat_threshold: float = 0.6
    ):
        self.char_dedup = CharLevelDedup(char_repeat_threshold)
        self.word_dedup = WordRepetitionDedup(word_repeat_threshold)

    def remove_repetitive(self, documents: List[str]) -> List[str]:
        docs1 = self.char_dedup.remove_repetitive_docs(documents)
        docs2 = self.word_dedup.remove_repetitive_docs(docs1)
        return docs2
