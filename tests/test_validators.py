import random
from datetime import date

import pytest

from pii_br import validators as v

# ----------------------------------------------------------------------- CPF


@pytest.mark.parametrize(
    "value", ["123.456.789-09", "12345678909", "987.654.321-00", "123 456 789 09", "529.982.247-25"]
)
def test_cpf_valid(value: str) -> None:
    assert v.is_valid_cpf(value)


@pytest.mark.parametrize(
    "value",
    [
        "123.456.789-00",  # dígito errado
        "111.111.111-11",  # sequência repetida (fecha o dígito, mas é inválida)
        "000.000.000-00",
        "1234567890",  # curto
        "123456789090",  # longo
        "123.456.78a-09",
        "",
    ],
)
def test_cpf_invalid(value: str) -> None:
    assert not v.is_valid_cpf(value)


def test_cpf_check_digits() -> None:
    assert v.cpf_check_digits("123456789") == "09"
    with pytest.raises(ValueError):
        v.cpf_check_digits("12345678")


def test_generate_cpf_valid_and_invalid() -> None:
    rng = random.Random(1)
    for _ in range(300):
        good = v.generate_cpf(rng)
        bad = v.generate_cpf(rng, valid=False, formatted=False)
        assert v.is_valid_cpf(good) and len(good) == 14
        assert not v.is_valid_cpf(bad) and len(bad) == 11


# ---------------------------------------------------------------------- CNPJ

OFFICIAL_ALNUM_EXAMPLE = "12.ABC.345/01DE-35"  # exemplo publicado pela Receita Federal


@pytest.mark.parametrize(
    "value",
    [
        "12.345.678/0001-95",
        "12345678000195",
        OFFICIAL_ALNUM_EXAMPLE,
        "12ABC34501DE35",
        "12.abc.345/01de-35",  # entrada em minúsculas é normalizada
    ],
)
def test_cnpj_valid(value: str) -> None:
    assert v.is_valid_cnpj(value)


@pytest.mark.parametrize(
    "value",
    [
        "12.345.678/0001-94",
        "12.ABC.345/01DE-36",
        "12ABC34501DEA5",  # dígito verificador nunca é letra
        "00.000.000/0000-00",
        "12.ABC.345/01D!-35",
        "1234567800019",
    ],
)
def test_cnpj_invalid(value: str) -> None:
    assert not v.is_valid_cnpj(value)


def test_cnpj_check_digits_official_example() -> None:
    assert v.cnpj_check_digits("12ABC34501DE") == "35"
    assert v.cnpj_check_digits("123456780001") == "95"
    with pytest.raises(ValueError):
        v.cnpj_check_digits("12ABC34501D")


def test_generate_cnpj() -> None:
    rng = random.Random(2)
    for _ in range(300):
        numeric = v.generate_cnpj(rng)
        alnum = v.generate_cnpj(rng, alphanumeric=True, formatted=False)
        bad = v.generate_cnpj(rng, alphanumeric=True, valid=False)
        assert v.is_valid_cnpj(numeric) and numeric.replace(".", "").replace("/", "")[:12].isdigit()
        assert v.is_valid_cnpj(alnum) and any(c.isalpha() for c in alnum[:12])
        assert not v.is_valid_cnpj(bad)


def test_format_cnpj() -> None:
    assert v.format_cnpj("12abc34501de35") == OFFICIAL_ALNUM_EXAMPLE


# ------------------------------------------------------------------ Telefone


@pytest.mark.parametrize(
    "value",
    [
        "(11) 98765-4321",
        "+55 11 98765-4321",
        "5511987654321",
        "011 98765-4321",
        "11987654321",
        "(21) 2345-6789",
        "98765-4321",
        "3456-7890",
    ],
)
def test_phone_valid(value: str) -> None:
    assert v.is_valid_phone(value)


@pytest.mark.parametrize(
    "value",
    [
        "(20) 98765-4321",  # DDD inexistente
        "(11) 88765-4321",  # 9 dígitos sem começar com 9
        "(11) 1234-5678",  # fixo começa com 2-5
        "0800 123 4567",
        "4004-0001",
        "123",
    ],
)
def test_phone_invalid(value: str) -> None:
    assert not v.is_valid_phone(value)


def test_phone_require_ddd() -> None:
    assert not v.is_valid_phone("98765-4321", require_ddd=True)
    assert v.is_valid_phone("(11) 98765-4321", require_ddd=True)


def test_all_ddds_known() -> None:
    assert len(v.VALID_DDDS) == 67


# --------------------------------------------------------------- CEP, e-mail


def test_cep() -> None:
    assert v.is_valid_cep("01310-100")
    assert v.is_valid_cep("01.310-100")
    assert v.is_valid_cep("99999999")
    assert not v.is_valid_cep("00000-000")
    assert not v.is_valid_cep("00999-999")
    assert not v.is_valid_cep("1310-100")


def test_email() -> None:
    assert v.is_valid_email("maria.souza@example.com")
    assert v.is_valid_email("MARIA+tag@sub.example.com.br")
    assert not v.is_valid_email("maria@example")
    assert not v.is_valid_email("maria..souza@example.com")
    assert not v.is_valid_email("@example.com")


# --------------------------------------------------------------------- Datas

TODAY = date(2026, 11, 23)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("03/07/1990", date(1990, 7, 3)),
        ("3-7-1990", date(1990, 7, 3)),
        ("03.07.90", date(1990, 7, 3)),
        ("03/07/05", date(2005, 7, 3)),
        ("3 de julho de 1990", date(1990, 7, 3)),
        ("1º de março de 1985", date(1985, 3, 1)),
        ("3 de jul. de 1990", date(1990, 7, 3)),
        ("31/02/1990", None),
        ("3 de julhoo de 1990", None),
        ("ontem", None),
    ],
)
def test_parse_date(value: str, expected: date | None) -> None:
    assert v.parse_date(value, today=TODAY) == expected


def test_birth_date() -> None:
    assert v.is_valid_birth_date("03/07/1990", today=TODAY)
    assert not v.is_valid_birth_date("03/07/2030", today=TODAY)
    assert not v.is_valid_birth_date("03/07/1850", today=TODAY)
