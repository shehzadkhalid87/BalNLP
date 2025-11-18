"""Tests for BalochiStopwordRemover."""

import os
import tempfile
from pathlib import Path

import pytest

from balnlp.preprocessing.stopwords import BalochiStopwordRemover


@pytest.fixture
def default_remover():
    return BalochiStopwordRemover()


def test_default_stopwords(default_remover):
    remover = default_remover
    text = "من کتاب وانان"
    filtered = remover.remove_stopwords(text)
    if "من" in remover.stopwords:
        assert "من" not in filtered
    assert "کتاب" in filtered


def test_custom_stopwords():
    custom_stopwords = {"کتاب"}
    remover = BalochiStopwordRemover()
    remover.stopwords = set(custom_stopwords)
    text = "من کتاب وانان"
    filtered = remover.remove_stopwords(text)
    assert "کتاب" not in filtered
    assert "من" in filtered
    assert "وانان" in filtered


def test_stopwords_from_list(default_remover):
    remover = default_remover
    tokens = ["من", "کتاب", "وانان"]
    filtered = remover.remove_stopwords_from_list(tokens)
    if "من" in remover.stopwords:
        assert "من" not in filtered
    assert "کتاب" in filtered


def test_add_stopwords(default_remover):
    remover = default_remover
    remover.add_stopwords({"کتاب", "وانان"})
    text = "من کتاب وانان"
    filtered = remover.remove_stopwords(text)
    assert "کتاب" not in filtered
    assert "وانان" not in filtered


def test_remove_custom_stopwords(default_remover):
    remover = default_remover
    text = "من کتاب وانان"
    if "من" in remover.stopwords:
        assert "من" not in remover.remove_stopwords(text)
    remover.remove_custom_stopwords({"من"})
    filtered2 = remover.remove_stopwords(text)
    assert "من" in filtered2


def test_load_stopwords_from_file():
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as f:
        f.write("کتاب\nوانان\n")
        temp_path = f.name
    try:
        remover = BalochiStopwordRemover(stopwords_file=temp_path)
        text = "من کتاب وانان"
        filtered = remover.remove_stopwords(text)
        assert "کتاب" not in filtered
        assert "وانان" not in filtered
        assert "من" in filtered
    finally:
        os.unlink(temp_path)


def test_save_stopwords(default_remover):
    remover = default_remover
    with tempfile.TemporaryDirectory() as temp_dir:
        save_path = Path(temp_dir) / "stopwords.txt"
        remover.save_stopwords(str(save_path))
        assert save_path.exists()
        with open(save_path, "r", encoding="utf-8") as f:
            content = f.read()
            for word in remover.stopwords:
                assert word in content


def test_empty_text(default_remover):
    remover = default_remover
    assert remover.remove_stopwords("") == ""
    assert remover.remove_stopwords_from_list([]) == []


def test_no_stopwords_in_text(default_remover):
    remover = default_remover
    text = "کتاب قلم"
    filtered = remover.remove_stopwords(text)
    assert filtered == text
