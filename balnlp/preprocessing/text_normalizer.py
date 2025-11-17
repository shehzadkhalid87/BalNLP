from typing import List


class BalochiTextNormalizer:
    """Normalize Balochi text (chars, diacritics, spaces)."""

    def __init__(self):
        self.char_maps = {
            "ي": "ی", "ئ": "ی",
            "ك": "ک", "ق": "ک",
            "ة": "ہ",
            "س": "س", "ث": "س", "ص": "س",
            "ز": "ز", "ذ": "ز", "ض": "ز", "ظ": "ز",
            "ا": "ا", "ع": "ا",
            "پ": "پ", "ف": "پ",
            "ھ": "ھ", "ح": "ھ", "خ": "ھ",
            "ت": "ت", "ط": "ت",
            "گ": "گ", "غ": "گ"
        }
        self.diacritics = [
            "\u064b", "\u064c", "\u064d", "\u064e",
            "\u064f", "\u0650", "\u0651", "\u0652"
        ]

    def normalize_chars(self, text: str) -> str:
        for original, normalized in self.char_maps.items():
            text = text.replace(original, normalized)
        return text

    def remove_diacritics(self, text: str) -> str:
        for d in self.diacritics:
            text = text.replace(d, "")
        return text

    def normalize_spaces(self, text: str) -> str:
        text = " ".join(text.split())
        text = text.replace(" ،", "،").replace(" ؟", "؟").replace(" !", "!").replace(" .", ".")
        return text

    def normalize_text(self, text: str, remove_diacritics: bool = False) -> str:
        text = self.normalize_chars(text)
        if remove_diacritics:
            text = self.remove_diacritics(text)
        text = self.normalize_spaces(text)
        return text.strip()

    def normalize_corpus(self, corpus: List[str], remove_diacritics: bool = False) -> List[str]:
        return [self.normalize_text(text, remove_diacritics) for text in corpus]

