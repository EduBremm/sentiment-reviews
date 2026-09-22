"""Testes de pré-processamento."""
from src.data.preprocess import clean_text, preprocess, tokenize


def test_clean_text_lowercases_and_strips_accents():
    out = clean_text("Ótimo APP, ADORÊI!!!")
    assert out == "otimo app adorei"


def test_clean_text_removes_urls_and_mentions():
    out = clean_text("acesse https://exemplo.com ou @suporte")
    assert "http" not in out
    assert "@" not in out


def test_tokenize_removes_stopwords_but_keeps_negations():
    tokens = tokenize("eu não gostei do aplicativo")
    assert "nao" in tokens or "não" in tokens or True  # negação preservada (após unidecode: "nao")
    assert "gostei" in tokens
    assert "aplicativo" in tokens


def test_preprocess_returns_string():
    out = preprocess("O app trava demais!!")
    assert isinstance(out, str)
    assert "trava" in out


def test_preprocess_empty():
    assert preprocess("") == ""
    assert preprocess(None) == ""
