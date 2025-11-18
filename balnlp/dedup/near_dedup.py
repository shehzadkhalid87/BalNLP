class NearDedup:
    """Remove near duplicates using Jaccard similarity over shingles."""

    def __init__(self, shingle_size=3, threshold=0.8, mode="word"):
        """
        Args:
            shingle_size: Size of shingles (n-grams)
            threshold: Jaccard similarity threshold (0.0 to 1.0)
            mode: 'word' for word-level shingles, 'char' for character-level shingles
        """
        self.shingle_size = shingle_size
        self.threshold = threshold
        self.mode = mode

    def shingles(self, text: str):
        """Generate shingles based on selected mode."""
        if self.mode == "word":
            return self._word_shingles(text)
        else:  # character level
            return self._char_shingles(text)

    def _word_shingles(self, text: str):
        """Generate word-level shingles."""
        words = text.split()
        if len(words) < self.shingle_size:
            return set()
        return set(
            tuple(words[i : i + self.shingle_size])
            for i in range(len(words) - self.shingle_size + 1)
        )

    def _char_shingles(self, text: str):
        """Generate character-level shingles."""
        text = text.strip()
        if len(text) < self.shingle_size:
            return set()
        return set(
            text[i : i + self.shingle_size]
            for i in range(len(text) - self.shingle_size + 1)
        )

    def jaccard(self, set1, set2):
        """Calculate Jaccard similarity between two sets."""
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    def remove_near_duplicates(self, documents):
        """Remove near-duplicate documents."""
        if not documents:
            return []

        unique = []
        seen_shingles = []

        for doc in documents:
            if not doc or not doc.strip():
                continue

            sh = self.shingles(doc)

            is_dup = False
            for old_shingles in seen_shingles:
                sim = self.jaccard(sh, old_shingles)
                if sim >= self.threshold:
                    is_dup = True
                    break

            if not is_dup:
                seen_shingles.append(sh)
                unique.append(doc)

        return unique
