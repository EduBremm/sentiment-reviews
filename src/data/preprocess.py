"""Pré-processamento de texto em português."""
from __future__ import annotations

import re
from functools import lru_cache

from unidecode import unidecode

try:
    from nltk.corpus import stopwords
    from nltk.stem import RSLPStemmer
except ImportError:  # pragma: no cover
    stopwords = None
    RSLPStemmer = None


URL_RE = re.compile(r"http\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")
MULTISPACE_RE = re.compile(r"\s+")

# Emojis muito comuns são removidos pela normalização unicode + regex de não-alfa
NEG_WORDS = {"nao", "nunca", "nem", "jamais"}


@lru_cache(maxsize=1)
def _get_stopwords() -> set[str]:
    if stopwords is None:
        return set()
    try:
        sw = set(stopwords.words("portuguese"))
    except LookupError:
        import nltk
        nltk.download("stopwords", quiet=True)
        sw = set(stopwords.words("portuguese"))
    # Mantemos palavras de negação porque impactam sentimento
    return {unidecode(w) for w in sw} - NEG_WORDS


@lru_cache(maxsize=1)
def _get_stemmer() -> RSLPStemmer | None:
    if RSLPStemmer is None:
        return None
    try:
        return RSLPStemmer()
    except LookupError:
        import nltk
        nltk.download("rslp", quiet=True)
        return RSLPStemmer()


def clean_text(text: str) -> str:
    """Limpeza básica: lowercase, remove URLs, menções, acentos, pontuação."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = unidecode(text)
    text = NON_ALPHA_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text


def tokenize(text: str, remove_stopwords: bool = True, stem: bool = False) -> list[str]:
    tokens = clean_text(text).split()
    if remove_stopwords:
        sw = _get_stopwords()
        tokens = [t for t in tokens if t not in sw and len(t) > 2]
    if stem:
        stemmer = _get_stemmer()
        if stemmer is not None:
            tokens = [stemmer.stem(t) for t in tokens]
    return tokens


def preprocess(text: str, stem: bool = False) -> str:
    """Retorna o texto limpo pronto para vetorização (string)."""
    return " ".join(tokenize(text, remove_stopwords=True, stem=stem))
