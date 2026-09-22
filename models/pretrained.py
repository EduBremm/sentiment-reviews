"""Modelo pré-treinado de NLP para análise de sentimento em português.

Usa o léxico SentiLex-PT adaptado via regras e o TextBlob com tradução leve,
sem necessidade de baixar modelos pesados (BERT/transformers).

Estratégia: VADER-style com léxico em português — palavras positivas/negativas
ponderadas + modificadores de intensidade + negação.

Uso:
    python -m src.models.pretrained
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.config import MODELS_DIR
from src.data.preprocess import clean_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

LABELS = ["NEGATIVO", "NEUTRO", "POSITIVO"]

# ──────────────────────────────────────────────
# Léxico de sentimento em português (SentiLex-PT simplificado)
# Palavras positivas → score +1, negativas → score -1
# ──────────────────────────────────────────────
POSITIVE_WORDS = {
    "otimo", "excelente", "perfeito", "adorei", "amei", "incrivel", "maravilhoso",
    "fantastico", "bom", "boa", "bons", "boas", "melhor", "rapido", "rapida",
    "facil", "pratico", "util", "funciona", "funcionou", "recomendo", "gostei",
    "satisfeito", "satisfeita", "top", "show", "demais", "sensacional", "nota",
    "aprovado", "aprovei", "eficiente", "estavel", "seguro", "confiavel",
    "intuitivo", "simples", "completo", "completa", "agradavel", "gratuito",
    "gratis", "leve", "economico", "inovador", "moderno", "prazeroso",
}

NEGATIVE_WORDS = {
    "pessimo", "horrivel", "terrivel", "ruim", "lento", "lenta", "trava", "trava",
    "travando", "bug", "bugs", "erro", "erros", "falha", "falhas", "problema",
    "problemas", "odio", "odeio", "nao funciona", "nao abre", "nao consigo",
    "propaganda", "propagandas", "anuncio", "anuncios", "instavel", "cai",
    "caindo", "desinstalar", "desinstalei", "desinstalo", "fraude", "golpe",
    "inseguro", "lixo", "inutil", "horrendo", "patético", "vergonha",
    "decepcao", "decepcionado", "decepcionante", "piorou", "pior", "quebrou",
    "banido", "banida", "suspensa", "suspendida", "bloqueada", "bloqueado",
    "dificil", "complicado", "confuso", "lerdo", "pesado", "gasta", "drena",
    "consome", "demora", "demorado", "engana", "enganoso", "abusivo",
}

NEGATION_WORDS = {"nao", "nunca", "jamais", "nem", "tampouco"}
INTENSIFIERS = {"muito", "demais", "super", "ultra", "extremamente", "bastante", "cada vez mais"}


def _score_text(text: str) -> float:
    """Retorna score entre -1 (muito negativo) e +1 (muito positivo)."""
    tokens = clean_text(text).split()
    score = 0.0
    i = 0
    while i < len(tokens):
        token = tokens[i]
        # Verifica bigrama (ex: "nao funciona")
        bigram = f"{token} {tokens[i+1]}" if i + 1 < len(tokens) else ""

        negated = (i > 0 and tokens[i - 1] in NEGATION_WORDS)
        intensified = (i > 0 and tokens[i - 1] in INTENSIFIERS)
        multiplier = 1.5 if intensified else 1.0

        if bigram in NEGATIVE_WORDS:
            delta = -1.5 * multiplier
            score += -delta if negated else delta
            i += 2
            continue
        elif token in POSITIVE_WORDS:
            delta = 1.0 * multiplier
            score += -delta if negated else delta
        elif token in NEGATIVE_WORDS:
            delta = -1.0 * multiplier
            score += -delta if negated else delta
        elif token in NEGATION_WORDS:
            pass  # capturado no próximo token
        i += 1

    # Normaliza pelo tamanho do texto
    n = max(len(tokens), 1)
    return max(-1.0, min(1.0, score / (n ** 0.5)))


def predict_pretrained(text: str) -> dict:
    """Prediz sentimento via léxico pré-treinado."""
    score = _score_text(text)
    if score > 0.05:
        sentiment = "POSITIVO"
    elif score < -0.05:
        sentiment = "NEGATIVO"
    else:
        sentiment = "NEUTRO"

    # Converte score para probabilidades aproximadas
    pos = max(0.0, score)
    neg = max(0.0, -score)
    neu = 1.0 - pos - neg
    total = pos + neg + neu
    probs = {
        "POSITIVO": round(pos / total, 3),
        "NEUTRO": round(neu / total, 3),
        "NEGATIVO": round(neg / total, 3),
    }
    return {
        "text": text,
        "sentiment": sentiment,
        "confidence": round(abs(score), 3),
        "probabilities": probs,
        "method": "pretrained_lexicon_pt",
    }


def evaluate_pretrained(df: pd.DataFrame) -> dict:
    """Avalia o modelo pré-treinado contra os rótulos reais."""
    df = df.dropna(subset=["text", "sentiment"]).copy()
    df = df[df["sentiment"].isin(LABELS)]

    logger.info("Avaliando modelo pré-treinado em %s reviews...", len(df))
    y_true = df["sentiment"].values
    y_pred = np.array([predict_pretrained(t)["sentiment"] for t in df["text"].values])

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0)
    report = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS).tolist()

    logger.info("pretrained_lexicon_pt | acc=%.4f  f1_macro=%.4f", acc, f1)

    result = {
        "name": "pretrained_lexicon_pt",
        "accuracy": acc,
        "f1_macro": f1,
        "roc_auc_ovr": None,
        "report": report,
        "confusion": cm,
    }

    out = MODELS_DIR / "pretrained_metrics.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    logger.info("Métricas salvas em %s", out)
    return result


def main() -> None:
    from src.models.train import load_data_from_db, load_data_from_csv
    from src.config import RAW_DIR

    try:
        df = load_data_from_db()
    except Exception as exc:
        logger.warning("Falha no banco (%s). Usando CSV...", exc)
        csvs = sorted(RAW_DIR.glob("reviews_*.csv"))
        if not csvs:
            raise RuntimeError("Sem dados. Rode src.data.collect antes.") from exc
        df = load_data_from_csv(csvs[-1])

    result = evaluate_pretrained(df)
    print("\n=== Resultado — Modelo Pré-Treinado (Léxico PT) ===")
    print(f"Accuracy : {result['accuracy']:.4f}")
    print(f"F1 Macro : {result['f1_macro']:.4f}")
    print("\nReport:")
    print(pd.DataFrame(result["report"]).T.round(3))


if __name__ == "__main__":
    main()
