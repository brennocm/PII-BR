import random
import re

import pytest

from pii_br import Detector, Entity, Masker, Span, mask, merge, recognize
from pii_br import validators as v


def model_span(text: str, value: str, label: Entity, score: float = 0.9) -> Span:
    start = text.index(value)
    return Span(start, start + len(value), label, value, "modelo", score)


class FakeModel:
    def __init__(self, spans: list[Span]) -> None:
        self.spans = spans

    def predict(self, text: str) -> list[Span]:
        return self.spans


# ------------------------------------------------------------------------ fusão

TEXT = "Sou Ana Lima, CPF 123.456.789-09, RG 12.345.678-9, tel (11) 98765-4321."


def test_same_label_becomes_ambos_with_regex_bounds() -> None:
    regex = recognize(TEXT)
    model = [model_span(TEXT, "CPF 123.456.789-09", Entity.CPF, 0.7)]
    cpf = next(s for s in merge(regex, model) if s.label is Entity.CPF)
    assert cpf.source == "ambos" and cpf.text == "123.456.789-09" and cpf.score == 0.99


def test_model_only_entities_are_added() -> None:
    model = [
        model_span(TEXT, "Ana Lima", Entity.NOME),
        model_span(TEXT, "12.345.678-9", Entity.RG),
    ]
    labels = [s.label for s in merge(recognize(TEXT), model)]
    assert labels == [Entity.NOME, Entity.CPF, Entity.RG, Entity.TELEFONE]


def test_valid_regex_beats_conflicting_model_label() -> None:
    model = [model_span(TEXT, "123.456.789-09", Entity.RG)]
    [cpf] = [s for s in merge(recognize(TEXT), model) if s.start == TEXT.index("123.456")]
    assert cpf.label is Entity.CPF


def test_model_wins_for_context_entities() -> None:
    text = "RG 1234567890 emitido em SP"
    regex = recognize(text)  # 10 dígitos com DDD 12 válido parecem telefone para a regex
    assert [s.label for s in regex] == [Entity.TELEFONE]
    merged = merge(regex, [model_span(text, "1234567890", Entity.RG)])
    assert [s.label for s in merged] == [Entity.RG]


def test_longer_span_wins_otherwise() -> None:
    regex = [Span(4, 13, Entity.CEP, "01310-100", "regex", 0.85, None)]
    model = [Span(4, 9, Entity.TELEFONE, "01310", "modelo", 0.9)]
    assert merge(regex, model) == regex


def test_detector_threshold_keeps_invalid_alerts() -> None:
    text = "Sou Ana, CPF 123.456.789-00"
    detector = Detector(FakeModel([model_span(text, "Ana", Entity.NOME, 0.3)]), threshold=0.5)
    spans = detector.detect(text)
    assert [(s.label, s.valid) for s in spans] == [(Entity.CPF, False)]
    assert [s.label for s in Detector(threshold=0.5).detect(text, [Entity.NOME])] == []


# ----------------------------------------------------------------- mascaramento

SAMPLE = "Ana Lima, CPF 123.456.789-09, e-mail ana.lima@example.com, tel (11) 98765-4321."


def spans_for(text: str) -> list[Span]:
    return Detector(FakeModel([model_span(text, "Ana Lima", Entity.NOME)])).detect(text)


def test_redact() -> None:
    assert mask(SAMPLE, spans_for(SAMPLE)) == "[NOME], CPF [CPF], e-mail [EMAIL], tel [TELEFONE]."


def test_partial() -> None:
    out = mask(SAMPLE, spans_for(SAMPLE), "partial")
    assert out == "[NOME], CPF ***.456.789-**, e-mail a***@example.com, tel (**) *****-4321."


@pytest.mark.parametrize(
    ("label", "value", "expected"),
    [
        (Entity.CNPJ, "12.ABC.345/01DE-35", "**.ABC.345/****-**"),
        (Entity.CEP, "01310-100", "01310-***"),
        (Entity.RG, "12.345.678-9", "**.***.**8-9"),
        (Entity.DATA_NASC, "03/07/1990", "**/**/1990"),
        (Entity.ENDERECO, "Rua A, 1", "[ENDERECO]"),
    ],
)
def test_partial_per_entity(label: Entity, value: str, expected: str) -> None:
    assert Masker().replacement(label, value, "partial") == expected


def test_pseudonym_is_consistent_and_normalized() -> None:
    text = "CPF 123.456.789-09, de novo 12345678909, outro 529.982.247-25. Ana e ANA."
    a, b = text.index("Ana"), text.index("ANA")
    spans = [
        *recognize(text),
        Span(a, a + 3, Entity.NOME, "Ana", "modelo", 0.9),
        Span(b, b + 3, Entity.NOME, "ANA", "modelo", 0.9),
    ]
    out = Masker().mask(text, spans, "pseudonym")
    assert out == "CPF <CPF_1>, de novo <CPF_1>, outro <CPF_2>. <NOME_1> e <NOME_1>."


def test_pseudonym_depends_on_secret_salt() -> None:
    a, b = Masker(salt=b"a" * 32), Masker(salt=b"b" * 32)
    va = a.replacement(Entity.CPF, "123.456.789-09", "pseudonym")
    vb = b.replacement(Entity.CPF, "123.456.789-09", "pseudonym")
    assert va == vb == "<CPF_1>"  # rótulo igual, mas o mapa interno usa HMAC com sal
    assert a._pseudonyms.keys() != b._pseudonyms.keys()  # pyright: ignore[reportPrivateUsage]


def test_synthetic_values_are_valid_and_keep_format() -> None:
    m = Masker(rng=random.Random(1))
    cpf = m.replacement(Entity.CPF, "123.456.789-09", "synthetic")
    raw_cpf = m.replacement(Entity.CPF, "12345678909", "synthetic")
    cnpj = m.replacement(Entity.CNPJ, "12.ABC.345/01DE-35", "synthetic")
    phone = m.replacement(Entity.TELEFONE, "(11) 98765-4321", "synthetic")
    date = m.replacement(Entity.DATA_NASC, "03-07-90", "synthetic")
    assert v.is_valid_cpf(cpf) and re.fullmatch(r"\d{3}\.\d{3}\.\d{3}-\d{2}", cpf)
    assert v.is_valid_cpf(raw_cpf) and raw_cpf.isdigit()
    assert v.is_valid_cnpj(cnpj) and any(c.isalpha() for c in cnpj)
    assert v.is_valid_phone(phone, require_ddd=True) and re.fullmatch(
        r"\(\d\d\) \d{5}-\d{4}", phone
    )
    assert re.fullmatch(r"\d\d-\d\d-\d\d", date)
    for label in Entity:
        assert m.replacement(label, "x 12", "synthetic")  # nunca quebra


def test_mask_full_synthetic_text_has_no_original_values() -> None:
    out = mask(SAMPLE, spans_for(SAMPLE), "synthetic")
    for original in ("Ana Lima", "123.456.789-09", "ana.lima@example.com", "98765-4321"):
        assert original not in out
