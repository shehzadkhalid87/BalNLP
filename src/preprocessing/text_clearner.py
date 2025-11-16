import re


class BalochiTextCleaner:
    """
    Advanced text cleaner for the Balochi language.
    Handles URLs, emails, emojis, unwanted scripts,
    special Balochi characters, spacing, and normalization.
    """

    def __init__(self):
        # Patterns
        self.url_pattern = r"https?://\S+|www\.\S+"
        self.email_pattern = r"\S+@\S+\.\S+"
        self.number_pattern = r"\d+"

        # Unicode emoji ranges
        self.emoji_pattern = (
            r"[\U0001F600-\U0001F64F"
            r"\U0001F300-\U0001F5FF"
            r"\U0001F680-\U0001F6FF"
            r"\U0001F1E0-\U0001F1FF]"
        )

        # Non-Balochi Unicode ranges
        self.latin_pattern = r"[A-Za-z]+"
        self.chinese_pattern = r"[\u4e00-\u9fff]+"
        self.devanagari_pattern = r"[\u0900-\u097F]+"

        # Spacing
        self.extra_spaces = r"\s+"
        self.extra_newlines = r"\n\s*\n"

        # Balochi special combined forms
        self.special_marks = ["ءُ", "ءَ", "ءِ"]

    # -------------------------------
    # REMOVERS
    # -------------------------------

    def remove_urls(self, text: str) -> str:
        return re.sub(self.url_pattern, " ", text)

    def remove_emails(self, text: str) -> str:
        return re.sub(self.email_pattern, " ", text)

    def remove_numbers(self, text: str) -> str:
        return re.sub(self.number_pattern, " ", text)

    def remove_emojis(self, text: str) -> str:
        return re.sub(self.emoji_pattern, " ", text)

    def remove_non_balochi(self, text: str) -> str:
        """
        Remove Latin, Chinese, and Devanagari alphabets
        while keeping Arabic/Balochi script intact.
        """
        text = re.sub(self.latin_pattern, " ", text)
        text = re.sub(self.chinese_pattern, " ", text)
        text = re.sub(self.devanagari_pattern, " ", text)
        return text

    # -------------------------------
    # WHITESPACE NORMALIZATION
    # -------------------------------

    def normalize_whitespace(self, text: str) -> str:
        text = re.sub(self.extra_spaces, " ", text)
        text = text.replace("\t", " ")
        return text.strip()

    # -------------------------------
    # SPECIAL BALOCɦI HANDLING
    # -------------------------------

    def process_special_chars(self, text: str) -> str:
        """
        Split tokens correctly around: ءُ ، ءَ , ءِ
        while keeping these marks as standalone tokens.
        """

        tokens = []
        for token in text.split():

            # Specific ordered combos first
            for mark in self.special_marks:
                if mark in token:
                    parts = token.split(mark)
                    for p in parts:
                        if p.strip():
                            tokens.append(p.strip())
                    tokens.append(mark)
                    break
            else:
                # If contains plain ء
                if "ء" in token:
                    parts = re.split(r"(ء)", token)
                    for p in parts:
                        if p.strip():
                            tokens.append(p.strip())
                else:
                    tokens.append(token)

        return " ".join(tokens)

    # -------------------------------
    # MAIN PIPELINE
    # -------------------------------

    def clean_text(
        self,
        text: str,
        remove_numbers: bool = True,
        preserve_special_chars: bool = True,
    ) -> str:

        text = text.strip()

        # Stage 1: remove unwanted content
        text = self.remove_urls(text)
        text = self.remove_emails(text)
        text = self.remove_emojis(text)

        # Stage 2: safe punctuation filtering
        if preserve_special_chars:
            # Allow letters, digits, whitespace, and Balochi marks
            text = re.sub(r"[^\w\sءُءَءِ،؛.!؟-]", " ", text)
        else:
            text = re.sub(r"[^\w\s]", " ", text)

        # Stage 3: Remove non-Balochi scripts
        text = self.remove_non_balochi(text)

        # Stage 4: Remove numbers (optional)
        if remove_numbers:
            text = self.remove_numbers(text)

        # Stage 5: Normalize whitespace
        text = self.normalize_whitespace(text)

        # Stage 6: Final special-character splitting
        if preserve_special_chars:
            text = self.process_special_chars(text)

        return text.strip()

    # -------------------------------
    # FILE CLEANER
    # -------------------------------

    def clean_file(self, file_path: str, **kwargs) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            return self.clean_text(f.read(), **kwargs)
