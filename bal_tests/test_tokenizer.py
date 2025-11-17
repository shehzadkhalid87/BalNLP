import pytest

from balnlp.bal_tokenizer.sentence_tokenizer import BalochiSentenceTokenizer
from balnlp.bal_tokenizer.Word_tokenizer import BalochiWordTokenizer



@pytest.fixture
def word_tokenizer():
    return BalochiWordTokenizer()


@pytest.fixture
def sentence_tokenizer():
    return BalochiSentenceTokenizer()


# -------------------------------------------------------
# WORD TOKENIZER TESTS
# -------------------------------------------------------

def test_basic_word_tokenization(word_tokenizer):
    text = "منی نام احمد اِنت"
    tokens = word_tokenizer.tokenize(text)
    assert tokens == ["منی", "نام", "احمد", "اِنت"]


def test_punctuation_handling(word_tokenizer):
    text = "منی نام احمد اِنت۔ من بلوچستان ءَ زندگ کنان۔"
    tokens = word_tokenizer.tokenize(text)

    assert "۔" not in tokens
    assert {"منی", "نام", "احمد"}.issubset(set(tokens))


def test_special_char_tokenization(word_tokenizer):
    test_cases = [
        ("دشتءِ", ["دشتءِ"]),
        ("کتابءَ", ["کتابءَ"]),
        ("گسءُ", ["گسءُ"]),
    ]

    for input_text, expected_tokens in test_cases:
        assert word_tokenizer.tokenize(input_text) == expected_tokens


def test_compound_word_handling(word_tokenizer):
    text = "کتاب\u200cخانہ"
    tokens = word_tokenizer.tokenize(text, split_compounds=True)
    assert tokens == ["کتاب", "خانہ"]


def test_empty_string_handling(word_tokenizer):
    assert word_tokenizer.tokenize("") == []
    assert word_tokenizer.tokenize("   ") == []


def test_multiple_spaces_handling(word_tokenizer):
    text = "منی    نام     احمد    اِنت"
    assert word_tokenizer.tokenize(text) == ["منی", "نام", "احمد", "اِنت"]


def test_tokenize_with_affixes(word_tokenizer):
    text = "بےوفا نامراد"
    tokens = word_tokenizer.tokenize_with_affixes(text)

    assert tokens[0]["token"] == "بےوفا"
    assert "بے" in tokens[0]["prefixes"]

    assert tokens[1]["token"] == "نامراد"
    assert "نا" in tokens[1]["prefixes"]


# -------------------------------------------------------
# SENTENCE TOKENIZER TESTS
# -------------------------------------------------------

def test_basic_sentence_tokenization(sentence_tokenizer):
    text = "منی نام احمد اِنت۔ من بلوچستان ءَ زندگ کنان۔"
    sentences = sentence_tokenizer.tokenize(text)

    # Because '۔' is included in sentence endings → result should be 2 sentences
    assert len(sentences) == 2
    assert sentences[0].startswith("منی نام احمد اِنت")
    assert sentences[1].startswith("من بلوچستان")


def test_multiple_punctuation_handling(sentence_tokenizer):
    text = "منی نام احمد اِنت؟ من بلوچستان ءَ زندگ کنان! کجا رئوی۔"
    sentences = sentence_tokenizer.tokenize(text)

    assert len(sentences) == 3
    assert sentences[0].endswith("؟")
    assert sentences[1].endswith("!")
    assert sentences[2].endswith("۔")


def test_sentence_boundary_with_special_chars(sentence_tokenizer):
    text = "کتابءَ بوان۔ درسءِ یاد کن۔"
    sentences = sentence_tokenizer.tokenize(text)

    # Two '۔' → two sentences expected
    assert len(sentences) == 2
    assert sentences[0].startswith("کتابءَ بوان")
    assert sentences[1].startswith("درسءِ یاد کن")
