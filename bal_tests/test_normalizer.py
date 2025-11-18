"""Tests for BalochiTextNormalizer."""

import pytest
from balnlp.preprocessing.text_normalizer import BalochiTextNormalizer


@pytest.fixture
def normalizer():
    return BalochiTextNormalizer()


def test_char_mapping(normalizer):
    """Test that characters are mapped correctly."""
    text = "ي ئ ك ق ث ص س ذ ض ظ ع ح خ ط غ"
    result = normalizer.normalize_chars(text)
    # Mapping checks
    assert "ی" in result
    assert "ک" in result
    assert "س" in result  # ث, ص -> س
    assert "ز" in result  # ذ, ض, ظ -> ز
    assert "ا" in result  # ع -> ا
    assert "ھ" in result  # ح, خ -> ھ
    assert "ت" in result  # ط -> ت
    assert "گ" in result  # غ -> گ

    # Fun error: check that we don't accidentally map a letter twice
    # For example: "س" itself should remain "س" and not duplicated
    assert result.count("س") == 3  # ث, ص, original س -> 3 s's


def test_remove_diacritics(normalizer):
    """Test that diacritics are removed."""
    text = "كِتَابُ"
    result = normalizer.remove_diacritics(text)
    for d in normalizer.diacritics:
        assert d not in result
    # Original letters should remain
    assert "ك" in result


def test_normalize_spaces(normalizer):
    """Test spacing and punctuation fixes."""
    text = "سلام  ، دنيا  ؟  "
    result = normalizer.normalize_spaces(text)
    assert "  " not in result
    assert " ،" not in result
    assert " ؟" not in result
    assert " !" not in result
    assert " ." not in result


def test_full_normalization(normalizer):
    """Test the full pipeline."""
    text = "ي كِتَابُ  ،  دنيا ؟"
    result = normalizer.normalize_text(text, remove_diacritics=True)
    assert "ی" in result
    assert "ک" in result
    assert "ا" in result  # ع or original ا
    for d in normalizer.diacritics:
        assert d not in result
    assert "  " not in result
    assert result.endswith("؟")


def test_normalize_corpus(normalizer):
    """Test normalizing a list of texts."""
    corpus = ["ي كِتَابُ", "سلام  ، دنيا ؟"]
    normalized = normalizer.normalize_corpus(corpus, remove_diacritics=True)
    assert len(normalized) == 2
    for text in normalized:
        for d in normalizer.diacritics:
            assert d not in text
        assert "  " not in text  # no double spaces
