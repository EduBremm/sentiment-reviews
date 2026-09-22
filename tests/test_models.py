"""Testes leves do pipeline de treino (usa fixture pequena)."""
import pandas as pd
import pytest

from src.models.train import LABELS, train_all


@pytest.fixture
def toy_df():
    positivos = ["adorei o app é maravilhoso", "excelente muito bom recomendo", "perfeito funciona bem"] * 20
    neutros = ["mais ou menos serve", "razoavel poderia melhorar", "nada demais"] * 20
    negativos = ["péssimo trava toda hora", "horrível não abre desinstalei", "muito ruim propaganda demais"] * 20
    rows = (
        [(t, "POSITIVO") for t in positivos]
        + [(t, "NEUTRO") for t in neutros]
        + [(t, "NEGATIVO") for t in negativos]
    )
    return pd.DataFrame(rows, columns=["text", "sentiment"])


def test_train_produces_best_model(toy_df, tmp_path):
    result = train_all(toy_df, out_dir=tmp_path)
    assert result["best_model"] in {"naive_bayes", "logistic_regression", "svm_linear"}
    assert (tmp_path / "best_model.pkl").exists()
    assert (tmp_path / "metrics.json").exists()
    for r in result["results"]:
        assert r["name"] in {"naive_bayes", "logistic_regression", "svm_linear"}
        assert 0 <= r["accuracy"] <= 1
        assert 0 <= r["f1_macro"] <= 1


def test_labels_constant():
    assert set(LABELS) == {"POSITIVO", "NEUTRO", "NEGATIVO"}
