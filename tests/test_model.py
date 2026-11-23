from pathlib import Path
from typing import Any

import pytest

from pii_br import Detector, Entity
from pii_br.model import NerModel
from pii_br.words import words

LOCAL_MODEL = Path(__file__).parents[1] / "outputs" / "mini"


def fake_model(outputs: list[dict[str, Any]]) -> NerModel:
    model = NerModel.__new__(NerModel)
    model._pipe = lambda _text: outputs  # type: ignore[assignment]  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
    return model


def test_predict_trims_filters_and_joins() -> None:
    text = "Sou  Maria Souza , moro na Rua A, 12."
    raw = [
        {"entity_group": "NOME", "start": 3, "end": 10, "score": 0.9},  # " Maria" com espaços
        {"entity_group": "NOME", "start": 11, "end": 16, "score": 0.8},  # "Souza"
        {"entity_group": "ENDERECO", "start": 27, "end": 32, "score": 0.7},
        {"entity_group": "ENDERECO", "start": 34, "end": 36, "score": 0.95},
        {"entity_group": "MISC", "start": 0, "end": 3, "score": 0.99},  # rótulo desconhecido
        {"entity_group": "CPF", "start": 16, "end": 17, "score": 0.5},  # só espaço
    ]
    spans = fake_model(raw).predict(text)
    assert [(s.label, s.text, s.source) for s in spans] == [
        (Entity.NOME, "Maria Souza", "modelo"),
        (Entity.ENDERECO, "Rua A, 12", "modelo"),
    ]
    assert spans[0].score == 0.8  # junção fica com o menor score


def test_predict_empty_text() -> None:
    assert (
        fake_model([{"entity_group": "NOME", "start": 0, "end": 1, "score": 1}]).predict("  ") == []
    )


@pytest.mark.skipif(not (LOCAL_MODEL / "config.json").exists(), reason="modelo local ausente")
def test_local_model_end_to_end() -> None:
    pytest.importorskip("torch")
    model = NerModel(str(LOCAL_MODEL), quantize=True, threads=2)
    text = ("Prezados, segue o cadastro: Maria Aparecida Souza, CPF 123.456.789-09. " * 20).strip()
    spans = Detector(model).detect(text)
    assert len(text) > 1000  # passa da janela de 256 tokens
    assert sum(s.label is Entity.CPF for s in spans) == 20
    assert all(text[s.start : s.end] == s.text for s in spans)


def test_words_helper() -> None:
    assert words("CPF: 123.456.789-09!") == ["CPF", ":", "123.456.789-09", "!"]
