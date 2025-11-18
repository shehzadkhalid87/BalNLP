from balnlp.dedup.near_dedup import NearDedup

docs = [
    "من بلوچے آں",  # doc0
    "من بلوچے ءِ آں",  # doc1
    "من ءَ بلوچی دوست بیت",  # doc2
    "بلوچی زبان زندگ بات",  # doc3
    "من ءَ بلوچی دوست بیت",  # doc4 (exact duplicate of doc2)
]

nd = NearDedup(threshold=0.4)
unique = nd.remove_near_duplicates(docs)
print(unique)
