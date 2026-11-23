import time

import pytest

from pii_br import Entity, recognize
from pii_br import validators as v


def labels(text: str) -> list[tuple[str, str, bool | None]]:
    return [(s.label.value, s.text, s.valid) for s in recognize(text)]


def test_spec_example() -> None:
    text = (
        "Oi, sou Maria Aparecida Souza, meu CPF é 123.456.789-09 e moro na "
        "Rua das Flores, 123, ap. 42, CEP 01310-100."
    )
    assert labels(text) == [("CPF", "123.456.789-09", True), ("CEP", "01310-100", True)]


def test_offsets_point_to_original_text() -> None:
    text = "Contato: ana@example.com, tel (11) 98765-4321, CNPJ 12.345.678/0001-95."
    spans = recognize(text)
    assert [s.label for s in spans] == [Entity.EMAIL, Entity.TELEFONE, Entity.CNPJ]
    for s in spans:
        assert text[s.start : s.end] == s.text
        assert s.source == "regex"


def test_invalid_cpf_is_flagged_not_dropped() -> None:
    [span] = recognize("CPF 123.456.789-00")
    assert span.label is Entity.CPF and span.valid is False and span.score < 0.9


@pytest.mark.parametrize(
    "value", ["12.ABC.345/01DE-35", "12.abc.345/01de-35", "12ABC34501DE35", "12ABC345/01DE-35"]
)
def test_alphanumeric_cnpj(value: str) -> None:
    assert labels(f"Empresa de CNPJ {value} ativa") == [("CNPJ", value, True)]


def test_alphanumeric_cnpj_bare_needs_valid_digit() -> None:
    assert recognize("código 12ABC34501DE36 do lote") == []


@pytest.mark.parametrize(
    "phone",
    [
        "(11) 98765-4321",
        "+55 11 98765-4321",
        "+55 (11) 98765-4321",
        "11 98765-4321",
        "(21) 2345-6789",
        "(11)987654321",
        "98765-4321",
        "5511987654321",
    ],
)
def test_phone_formats(phone: str) -> None:
    assert labels(f"me chama no {phone} amanhã") == [("TELEFONE", phone, True)]


def test_landline_without_ddd_needs_context() -> None:
    assert labels("ligue 3456-7890") == [("TELEFONE", "3456-7890", True)]
    assert recognize("pedido 3456-7890") == []


@pytest.mark.parametrize("text", ["SAC 0800 123 4567", "central 4004-0001", "0800-123-4567"])
def test_service_numbers_are_not_pii(text: str) -> None:
    assert [s for s in recognize(text) if s.label is Entity.TELEFONE] == []


def test_eleven_digits_follow_context() -> None:
    number = "11987654321"  # celular válido; também testamos o roteamento como CPF
    assert labels(f"whats {number}") == [("TELEFONE", number, True)]
    cpf_span = recognize(f"CPF: {number}")[0]
    assert cpf_span.label is Entity.CPF
    assert cpf_span.valid is v.is_valid_cpf(number)


def test_eleven_digits_without_context() -> None:
    assert labels("12345678909") == [("CPF", "12345678909", True)]
    assert labels("11987654321") == [("TELEFONE", "11987654321", True)]
    assert recognize("12345678900") == []  # nem CPF válido nem celular


def test_fourteen_digits() -> None:
    assert labels("12345678000195") == [("CNPJ", "12345678000195", True)]
    assert recognize("pedido 12345678000194") == []
    assert labels("CNPJ 12345678000194") == [("CNPJ", "12345678000194", False)]


def test_cep_variants() -> None:
    assert labels("CEP 01.310-100") == [("CEP", "01.310-100", True)]
    assert labels("cep: 01310100") == [("CEP", "01310100", True)]
    assert recognize("total 01310100") == []


def test_hard_negatives() -> None:
    text = "Pedido 2024000123, protocolo 98765432100123, boleto 12345 e data 03/07/1990."
    assert recognize(text) == []


def test_does_not_match_inside_longer_numbers() -> None:
    assert recognize("1123.456.789-09") == []
    assert recognize("serial 123.456.789-0912") == []


def test_email_variants() -> None:
    text = "Escreva para MARIA.SOUZA@EXAMPLE.COM.BR ou joao+ti@example.com."
    assert labels(text) == [
        ("EMAIL", "MARIA.SOUZA@EXAMPLE.COM.BR", True),
        ("EMAIL", "joao+ti@example.com", True),
    ]


def test_fullwidth_digits_keep_offsets() -> None:
    wide = "123.456.789-09".translate({ord(d): 0xFF10 + int(d) for d in "0123456789"})
    text = f"CPF {wide} ok"
    [span] = recognize(text)
    assert span.label is Entity.CPF and span.valid is True
    assert span.text == wide and span.text != "123.456.789-09"


def test_entity_filter() -> None:
    text = "CPF 123.456.789-09, e-mail ana@example.com"
    assert [s.label for s in recognize(text, entities=[Entity.EMAIL])] == [Entity.EMAIL]


def test_multiline_and_noise() -> None:
    text = "oi!! 😊\nmeu cpf eh 123 456 789 09\nzap (11)98765-4321"
    assert [s.label for s in recognize(text)] == [Entity.CPF, Entity.TELEFONE]


@pytest.mark.parametrize(
    "text", ["1/" * 20_000, "1.2-3/" * 8_000, "@" * 40_000, ("a" * 60 + "@") * 600]
)
def test_email_search_is_linear(text: str) -> None:
    started = time.perf_counter()
    recognize(text)
    assert time.perf_counter() - started < 2  # antes: 28 s para "1/" * 20_000


@pytest.mark.parametrize(
    ("text", "emails"),
    [
        ("a@example.com, b.c+d@sub.example.com.br.", ["a@example.com", "b.c+d@sub.example.com.br"]),
        ("ana@example.com;bia@example.com", ["ana@example.com", "bia@example.com"]),
        ("x@a@example.com", ["a@example.com"]),
        ("joao@@example.com", []),
        ("a" * 70 + "@example.com", []),  # parte local acima do limite da RFC
        # chave=valor de log: o "=" é válido na RFC, mas a chave não faz parte do e-mail
        ("email=ana@example.com cep=70040010", ["ana@example.com", "70040010"]),
        ('to="bia@example.com"', ["bia@example.com"]),
    ],
)
def test_email_edges(text: str, emails: list[str]) -> None:
    assert [s.text for s in recognize(text)] == emails
