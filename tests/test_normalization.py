from integrity_engine.preprocessing.normalization import normalize_text,comparison_text

def test_normalizes_unicode_and_whitespace():
    assert normalize_text("Ａ  B\n\n\tC") == "A B\nC"

def test_comparison_keeps_numbers_and_words():
    assert comparison_text("Plan v2.1!") == "plan v2 1"
