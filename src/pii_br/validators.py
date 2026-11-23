"""Validação determinística e geração de identificadores brasileiros.

Só usa a biblioteca padrão. Os geradores recebem um `random.Random` explícito para que o
dataset seja reprodutível por seed; não servem para nada criptográfico.

Referências das regras:
- CPF: módulo 11, pesos 10→2 e 11→2; resto 10 vira 0.
- CNPJ: módulo 11, pesos 5..2,9..2 e 6..2,9..2; resto < 2 vira 0.
- CNPJ alfanumérico (IN RFB nº 2.229/2024): as 12 primeiras posições aceitam [0-9A-Z],
  cada caractere vale ord(c) - 48; os dois dígitos verificadores continuam numéricos.
  Exemplo oficial da Receita: 12.ABC.345/01DE-35.
"""

from __future__ import annotations

import random
import re
from datetime import date

__all__ = [
    "VALID_DDDS",
    "cnpj_check_digits",
    "cpf_check_digits",
    "format_cnpj",
    "format_cpf",
    "generate_cnpj",
    "generate_cpf",
    "is_service_number",
    "is_valid_birth_date",
    "is_valid_cep",
    "is_valid_cnpj",
    "is_valid_cpf",
    "is_valid_email",
    "is_valid_phone",
    "normalize_cnpj",
    "only_digits",
    "parse_date",
    "phone_digits",
]

_SEPARATORS = re.compile(r"[.\-/\s]")
_ALNUM = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# DDDs em uso segundo o plano de numeração da Anatel (67 códigos).
VALID_DDDS: frozenset[str] = frozenset(
    {
        *("11", "12", "13", "14", "15", "16", "17", "18", "19"),  # SP
        *("21", "22", "24", "27", "28"),  # RJ, ES
        *("31", "32", "33", "34", "35", "37", "38"),  # MG
        *("41", "42", "43", "44", "45", "46", "47", "48", "49"),  # PR, SC
        *("51", "53", "54", "55"),  # RS
        *("61", "62", "63", "64", "65", "66", "67", "68", "69"),  # Centro-Oeste, TO, AC, RO
        *("71", "73", "74", "75", "77", "79"),  # BA, SE
        *("81", "82", "83", "84", "85", "86", "87", "88", "89"),  # Nordeste
        *("91", "92", "93", "94", "95", "96", "97", "98", "99"),  # Norte, MA
    }
)


def only_digits(value: str) -> str:
    return "".join(c for c in value if c.isascii() and c.isdigit())


# --------------------------------------------------------------------------- CPF


def cpf_check_digits(base: str) -> str:
    """Os 2 dígitos verificadores para os 9 primeiros dígitos de um CPF."""
    if len(base) != 9 or not base.isascii() or not base.isdigit():
        raise ValueError("a base do CPF tem 9 dígitos")
    digits = [int(c) for c in base]
    for _ in range(2):
        total = sum(d * w for d, w in zip(digits, range(len(digits) + 1, 1, -1), strict=True))
        remainder = total * 10 % 11
        digits.append(0 if remainder == 10 else remainder)
    return f"{digits[9]}{digits[10]}"


def is_valid_cpf(value: str) -> bool:
    """CPF com ou sem pontuação. Sequências repetidas (111.111.111-11) são inválidas."""
    clean = _SEPARATORS.sub("", value)
    if len(clean) != 11 or not clean.isascii() or not clean.isdigit():
        return False
    if len(set(clean)) == 1:
        return False
    return cpf_check_digits(clean[:9]) == clean[9:]


def format_cpf(digits: str) -> str:
    return f"{digits[:3]}.{digits[3:6]}.{digits[6:9]}-{digits[9:]}"


def generate_cpf(rng: random.Random, *, formatted: bool = True, valid: bool = True) -> str:
    """CPF aleatório. Com `valid=False`, os dígitos verificadores saem errados de propósito."""
    while True:
        base = "".join(rng.choice("0123456789") for _ in range(9))
        if len(set(base)) > 1:
            break
    dv = cpf_check_digits(base)
    if not valid:
        dv = _wrong_check_digits(rng, dv)
    digits = base + dv
    return format_cpf(digits) if formatted else digits


# -------------------------------------------------------------------------- CNPJ

_CNPJ_WEIGHTS = (6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2)


def _cnpj_char_value(char: str) -> int:
    return ord(char) - 48


def normalize_cnpj(value: str) -> str:
    """Remove separadores e passa para maiúsculas (o formato alfanumérico usa A-Z)."""
    return _SEPARATORS.sub("", value).upper()


def cnpj_check_digits(base: str) -> str:
    """Os 2 dígitos verificadores para as 12 primeiras posições (numéricas ou alfanuméricas)."""
    base = base.upper()
    if len(base) != 12 or any(c not in _ALNUM for c in base):
        raise ValueError("a base do CNPJ tem 12 caracteres [0-9A-Z]")
    values = [_cnpj_char_value(c) for c in base]
    for _ in range(2):
        weights = _CNPJ_WEIGHTS[-len(values) :]
        remainder = sum(v * w for v, w in zip(values, weights, strict=True)) % 11
        values.append(0 if remainder < 2 else 11 - remainder)
    return f"{values[12]}{values[13]}"


def is_valid_cnpj(value: str) -> bool:
    """CNPJ numérico ou alfanumérico, com ou sem pontuação, em qualquer caixa."""
    clean = normalize_cnpj(value)
    if len(clean) != 14:
        return False
    base, dv = clean[:12], clean[12:]
    if any(c not in _ALNUM for c in base) or not (dv.isascii() and dv.isdigit()):
        return False
    if len(set(clean)) == 1:
        return False
    return cnpj_check_digits(base) == dv


def format_cnpj(value: str) -> str:
    v = normalize_cnpj(value)
    return f"{v[:2]}.{v[2:5]}.{v[5:8]}/{v[8:12]}-{v[12:]}"


def generate_cnpj(
    rng: random.Random,
    *,
    alphanumeric: bool = False,
    formatted: bool = True,
    valid: bool = True,
) -> str:
    """CNPJ aleatório. No formato alfanumérico, a raiz e a ordem têm ao menos uma letra."""
    pool = _ALNUM if alphanumeric else "0123456789"
    while True:
        root = "".join(rng.choice(pool) for _ in range(8))
        branch = "0001" if rng.random() < 0.7 else "".join(rng.choice(pool) for _ in range(4))
        base = root + branch
        has_letter = any(c.isalpha() for c in base)
        if len(set(base)) > 1 and has_letter == alphanumeric:
            break
    dv = cnpj_check_digits(base)
    if not valid:
        dv = _wrong_check_digits(rng, dv)
    value = base + dv
    return format_cnpj(value) if formatted else value


def _wrong_check_digits(rng: random.Random, correct: str) -> str:
    while True:
        candidate = f"{rng.randrange(100):02d}"
        if candidate != correct:
            return candidate


# ---------------------------------------------------------------------- Telefone

_SERVICE_PREFIXES = ("0800", "0300", "0500", "0900")
_CAPITAL_PREFIXES = frozenset({"3003", "3004", "4002", "4003", "4004", "4020"})


def phone_digits(value: str) -> str:
    """Dígitos do número sem o código do país (55) e sem o zero de discagem."""
    digits = only_digits(value)
    if len(digits) in (12, 13) and digits.startswith("55"):
        digits = digits[2:]
    if len(digits) in (11, 12) and digits.startswith("0") and not digits.startswith("0800"):
        digits = digits[1:]
    return digits


def is_service_number(value: str) -> bool:
    """0800, 0300, 4004-xxxx e similares: números de empresa, não de pessoa."""
    digits = only_digits(value)
    if digits.startswith(_SERVICE_PREFIXES):
        return True
    return len(digits) == 8 and digits[:4] in _CAPITAL_PREFIXES


def is_valid_phone(value: str, *, require_ddd: bool = False) -> bool:
    """Fixo (8 dígitos, começa com 2-5) ou celular (9 dígitos, começa com 9), com DDD válido."""
    if is_service_number(value):
        return False
    digits = phone_digits(value)
    if len(digits) in (10, 11):
        ddd, subscriber = digits[:2], digits[2:]
        if ddd not in VALID_DDDS:
            return False
    elif len(digits) in (8, 9) and not require_ddd:
        subscriber = digits
    else:
        return False
    if len(subscriber) == 9:
        return subscriber[0] == "9"
    return subscriber[0] in "2345"


# --------------------------------------------------------------- CEP e e-mail

_EMAIL = re.compile(
    r"[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*"
    r"@(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?\.)+[A-Za-z]{2,}"
)


def is_valid_cep(value: str) -> bool:
    """8 dígitos na faixa usada pelos Correios (01000-000 a 99999-999)."""
    clean = _SEPARATORS.sub("", value)
    return len(clean) == 8 and clean.isascii() and clean.isdigit() and int(clean) >= 1_000_000


def is_valid_email(value: str) -> bool:
    return len(value) <= 254 and _EMAIL.fullmatch(value) is not None


# ------------------------------------------------------------------------ Datas

_MONTHS = {
    "janeiro": 1, "jan": 1, "fevereiro": 2, "fev": 2, "marco": 3, "março": 3, "mar": 3,
    "abril": 4, "abr": 4, "maio": 5, "mai": 5, "junho": 6, "jun": 6, "julho": 7, "jul": 7,
    "agosto": 8, "ago": 8, "setembro": 9, "set": 9, "outubro": 10, "out": 10,
    "novembro": 11, "nov": 11, "dezembro": 12, "dez": 12,
}  # fmt: skip
_NUMERIC_DATE = re.compile(r"(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4}|\d{2})")
_WRITTEN_DATE = re.compile(r"(\d{1,2})º?\s+de\s+([a-zç]+)\.?\s+de\s+(\d{4})", re.IGNORECASE)


def parse_date(value: str, *, today: date | None = None) -> date | None:
    """Lê dd/mm/aaaa (também com - ou .), dd/mm/aa e "3 de julho de 1990"."""
    value = value.strip()
    today = today or date.today()
    if m := _NUMERIC_DATE.fullmatch(value):
        day, month, year = int(m[1]), int(m[2]), int(m[3])
        if len(m[3]) == 2:
            year += 2000 if year <= today.year % 100 else 1900
    elif m := _WRITTEN_DATE.fullmatch(value):
        month_number = _MONTHS.get(m[2].lower())
        if month_number is None:
            return None
        day, month, year = int(m[1]), month_number, int(m[3])
    else:
        return None
    try:
        return date(year, month, day)
    except ValueError:
        return None


def is_valid_birth_date(value: str, *, today: date | None = None) -> bool:
    """Data válida, no passado e com no máximo 130 anos."""
    today = today or date.today()
    parsed = parse_date(value, today=today)
    return parsed is not None and parsed <= today and today.year - parsed.year <= 130
