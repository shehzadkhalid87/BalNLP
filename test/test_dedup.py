from balnlp.dedup import ExactDedup

docs = [
    "من بلوچے آں",
    "من ءَ بلوچی دوست بیت",
    "بلوچی زبان زندگ بات",
    "من ءَ بلوچی دوست بیت",
    " ",
    " ",
]

test = ExactDedup()
# Remove exact duplicates
unique_docs = test.remove_exact_duplicates(docs)
print(f"Original: {len(docs)}, Unique: {len(unique_docs)}")
# Output: Original: 5, Unique: 3

# Remove duplicates with Unicode normalization
normalized_unique = test.remove_normalized_duplicates(docs)
print(f"Normalized: {len(normalized_unique)}, Unique: {len(normalized_unique)}")

# Remove duplicates with whitespace normalization
whitespace_normalized = test.remove_duplicates_with_whitespace_normalization(docs)
print(whitespace_normalized)
