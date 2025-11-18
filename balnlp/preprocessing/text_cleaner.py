# src/preprocessing/cleaner.py
import re
from typing import Optional, List


class BalochiTextCleaner:
    def __init__(self):
        # Patterns
        self.url_pattern = r"https?://\S+|www\.\S+"
        self.email_pattern = r"\S+@\S+\.\S+"
        self.number_pattern = r"\d+"
        self.emoji_pattern = (
            r"[\U0001F600-\U0001F64F"
            r"\U0001F300-\U0001F5FF"
            r"\U0001F680-\U0001F6FF"
            r"\U0001F1E0-\U0001F1FF]"
        )
        self.latin_pattern = r"[A-Za-z]+"
        self.chinese_pattern = r"[\u4e00-\u9fff]+"
        self.devanagari_pattern = r"[\u0900-\u097F]+"
        self.extra_spaces = r"\s+"
        self.special_marks = ["ءُ", "ءَ", "ءِ"]

    def remove_urls(self, text):
        return re.sub(self.url_pattern, " ", text)

    def remove_emails(self, text):
        return re.sub(self.email_pattern, " ", text)

    def remove_numbers(self, text):
        return re.sub(self.number_pattern, " ", text)

    def remove_emojis(self, text):
        return re.sub(self.emoji_pattern, " ", text)

    def remove_non_balochi(self, text):
        text = re.sub(self.latin_pattern, " ", text)
        text = re.sub(self.chinese_pattern, " ", text)
        text = re.sub(self.devanagari_pattern, " ", text)
        return text

    def normalize_whitespace(self, text):
        return re.sub(self.extra_spaces, " ", text).strip()

    def process_special_chars(self, text):
        tokens = []
        for token in text.split():
            for mark in self.special_marks:
                if mark in token:
                    parts = token.split(mark)
                    for p in parts:
                        if p.strip():
                            tokens.append(p.strip())
                    tokens.append(mark)
                    break
            else:
                if "ء" in token:
                    parts = re.split(r"(ء)", token)
                    for p in parts:
                        if p.strip():
                            tokens.append(p.strip())
                else:
                    tokens.append(token)
        return " ".join(tokens)

    def clean_text(
            self,
            text: str,
            remove_urls: bool = True,
            remove_emails: bool = True,
            remove_numbers: bool = True,
            remove_emojis: bool = True,
            preserve_special_chars: bool = True,
            keep_chars: Optional[List[str]] = None,
    ) -> str:
        text = text.strip()
        if remove_urls: text = self.remove_urls(text)
        if remove_emails: text = self.remove_emails(text)
        if remove_emojis: text = self.remove_emojis(text)
        if preserve_special_chars:
            allowed = "".join(keep_chars) if keep_chars else "ءُءَءِ"
            text = re.sub(rf"[^\w\s{re.escape(allowed)}]", " ", text)
        else:
            text = re.sub(r"[^\w\s]", " ", text)
        text = self.remove_non_balochi(text)
        if remove_numbers: text = self.remove_numbers(text)
        text = self.normalize_whitespace(text)
        if preserve_special_chars: text = self.process_special_chars(text)
        return text
