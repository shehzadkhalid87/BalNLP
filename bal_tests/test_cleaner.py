"""Tests for BalochiTextCleaner."""

import pytest

from balnlp.preprocessing.text_clearner import BalochiTextCleaner


@pytest.fixture
def cleaner():
    return BalochiTextCleaner()

def test_basic_cleaning(cleaner):
    text = "منی نام احمد اِنت۔ https://example.com user@email.com 123 ABC"
    cleaned = cleaner.clean_text(text)
    assert "https://example.com" not in cleaned
    assert "user@email.com" not in cleaned
    assert "123" not in cleaned
    assert "ABC" not in cleaned
    assert "منی نام احمد" in cleaned

def test_special_char_handling(cleaner):
    test_cases = [
        ("دشتءِ", ["دشت", "ءِ"]),
        ("کتابءَ", ["کتاب", "ءَ"]),
        ("گسءُ", ["گس", "ءُ"])
    ]
    for input_text, expected_parts in test_cases:
        cleaned = cleaner.clean_text(input_text)
        for part in expected_parts:
            assert part in cleaned

def test_whitespace_normalization(cleaner):
    text = "منی    نام     احمد    اِنت۔"
    cleaned = cleaner.clean_text(text)
    assert "    " not in cleaned
    assert cleaned.count(" ") == 3  # Should have 3 spaces

def test_punctuation_removal(cleaner):
    text = "منی نام احمد اِنت۔ من بلوچستان ءَ زندگ کنان۔"
    cleaned = cleaner.clean_text(text)
    assert "۔" not in cleaned
    assert "منی نام احمد" in cleaned
    assert "من بلوچستان" in cleaned

def test_number_removal_option(cleaner):
    text = "منی نام احمد 123 اِنت۔"
    cleaned_with_removal = cleaner.clean_text(text, remove_numbers=True)
    assert "123" not in cleaned_with_removal

    cleaned_without_removal = cleaner.clean_text(text, remove_numbers=False)
    assert "123" in cleaned_without_removal

def test_special_chars_preservation_option(cleaner):
    text = "دشتءِ کتابءَ گسءُ"

    cleaned_with = cleaner.clean_text(text, preserve_special_chars=True)
    for char in ["ءِ", "ءَ", "ءُ"]:
        assert char in cleaned_with

    cleaned_without = cleaner.clean_text(text, preserve_special_chars=False)
    for char in ["ءِ", "ءَ", "ءُ"]:
        assert char not in cleaned_without
