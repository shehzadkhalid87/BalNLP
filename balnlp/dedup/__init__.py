from .exact_dedup import ExactDedup
from .near_dedup import NearDedup
from .repetition_dedup import RepetitionDedup, CharLevelDedup, WordRepetitionDedup

__all__ = [
    "ExactDedup",
    "NearDedup", 
    "RepetitionDedup",
    "CharLevelDedup",
    "WordRepetitionDedup",
]