import re
from typing import List, TypedDict


class TokenInfo(TypedDict):
    token: str
    root: str
    prefixes: List[str]
    suffixes: List[str]
    clitics: List[str]


class BalochiWordTokenizer:
    """A Balochi tokenizer that identifies prefixes, suffixes, and clitics."""

    def __init__(self):
        # Word boundaries
        self.word_boundaries = r'[\s,.!?؟،:;"\']+'

        # Common Balochi prefixes and suffixes
        self.prefixes = ["بے", "نا", "بی", "بد", "بر", "وا", "در", "ھر"]
        self.suffixes = ["اں", "ان", "ئے", "ئی", "یں", "انی", "ئیں", "باں", "بان", "ائی"]

        # Enclitics
        self.clitics = ["=ی", "=ن"]

    # ---------------------------------------------------------
    # Tokenization Core
    # ---------------------------------------------------------
    def tokenize(self, text: str, split_compounds: bool = True) -> List[str]:
        """Basic tokenizer with optional compound split."""
        text = text.strip()
        tokens = re.split(self.word_boundaries, text)
        tokens = [t for t in tokens if t]

        processed_tokens = []
        for token in tokens:
            if split_compounds and "\u200c" in token:
                # split zero-width non-joiner compounds
                parts = [t for t in token.split("\u200c") if t]
                processed_tokens.extend(parts)
            else:
                processed_tokens.append(token)

        return processed_tokens

    # ---------------------------------------------------------
    # Affix / Clitic splitting
    # ---------------------------------------------------------
    def split_affixes_and_clitics(self, token: str) -> TokenInfo:
        prefixes_found = []
        suffixes_found = []
        clitics_found = []

        root = token

        # ---- Extract clitics (end markers) ----
        for clitic in self.clitics:
            if root.endswith(clitic):
                clitics_found.append(clitic)
                root = root[:-len(clitic)]

        # ---- Extract MULTIPLE prefixes ----
        prefix_changed = True
        while prefix_changed:
            prefix_changed = False
            for prefix in self.prefixes:
                if root.startswith(prefix):
                    prefixes_found.append(prefix)
                    root = root[len(prefix):]
                    prefix_changed = True

        # ---- Extract MULTIPLE suffixes ----
        suffix_changed = True
        while suffix_changed:
            suffix_changed = False
            for suffix in self.suffixes:
                if root.endswith(suffix):
                    suffixes_found.append(suffix)
                    root = root[:-len(suffix)]
                    suffix_changed = True

        # Edge case: avoid empty roots
        if root == "":
            root = token

        return TokenInfo(
            token=token,
            root=root,
            prefixes=prefixes_found,
            suffixes=suffixes_found,
            clitics=clitics_found
        )

    # ---------------------------------------------------------
    # Full Tokenization Pipeline
    # ---------------------------------------------------------
    def tokenize_with_affixes(self, text: str) -> List[TokenInfo]:
        tokens = self.tokenize(text)
        return [self.split_affixes_and_clitics(t) for t in tokens]

