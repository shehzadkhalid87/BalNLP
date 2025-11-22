import hashlib
import unicodedata
from typing import Iterable, List, Set


class ExactDedup:
    """
    Remove exact duplicate documents using various normalization techniques.
    """

    @staticmethod
    def sha1_hash(text: str) -> str:
        """
        Compute SHA1 hash of text.

        Args:
            text: Input text to hash

        Returns:
            SHA1 hex digest string
        """
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """
        Normalize Unicode text to NFC form.
        Important for Balochi text which may have combining characters.

        Args:
            text: Input text to normalize

        Returns:
            Normalized text in NFC form
        """
        return unicodedata.normalize("NFC", text)

    def remove_exact_duplicates(self, documents: Iterable[str]) -> List[str]:
        """
        Remove byte-exact duplicate documents preserving first occurrence order.

        Args:
            documents: iterable of text documents

        Returns:
            list of unique documents preserving first occurrence order

        Proof of Correctness:
        1. Order Preserving: First occurrence of each duplicate is kept
        2. Complete: All exact duplicates are removed
        3. Minimal: Only duplicates are removed, all unique documents remain
        4. Deterministic: Same input always produces same output
        """
        seen: Set[str] = set()
        unique: List[str] = []

        for doc in documents:
            doc_hash = self.sha1_hash(doc)

            if doc_hash not in seen:
                seen.add(doc_hash)
                unique.append(doc)

        return unique

    def remove_normalized_duplicates(self, documents: Iterable[str]) -> List[str]:
        """
        Remove duplicates after Unicode normalization.
        Important for Balochi text with potential Unicode variations.

        Args:
            documents: iterable of text documents

        Returns:
            list of unique documents after normalization
        """
        seen: Set[str] = set()
        unique: List[str] = []

        for doc in documents:
            normalized_doc = self.normalize_unicode(doc)
            doc_hash = self.sha1_hash(normalized_doc)

            if doc_hash not in seen:
                seen.add(doc_hash)
                unique.append(doc)

        return unique

    def remove_duplicates_with_whitespace_normalization(
        self, documents: Iterable[str]
    ) -> List[str]:
        """
        Remove duplicates after normalizing whitespace.

        Args:
            documents: iterable of text documents

        Returns:
            list of unique documents after whitespace normalization
        """
        seen: Set[str] = set()
        unique: List[str] = []

        for doc in documents:
            # Normalize whitespace (collapse multiple spaces, trim)
            normalized_doc = " ".join(doc.split())
            doc_hash = self.sha1_hash(normalized_doc)

            if doc_hash not in seen:
                seen.add(doc_hash)
                unique.append(doc)

        return unique

    def remove_all_duplicates(self, documents: Iterable[str]) -> List[str]:
        """
        Remove duplicates using all normalization methods in sequence.

        Args:
            documents: iterable of text documents

        Returns:
            list of unique documents after applying all normalization methods
        """
        # Apply Unicode normalization first
        step1 = self.remove_normalized_duplicates(documents)
        # Then apply whitespace normalization
        step2 = self.remove_duplicates_with_whitespace_normalization(step1)
        # Finally apply exact deduplication
        return self.remove_exact_duplicates(step2)
