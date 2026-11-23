"""Estratégias de mascaramento aplicadas aos spans detectados.

| Estratégia     | "CPF 123.456.789-09" vira | Uso                                   |
| -------------- | ------------------------- | ------------------------------------- |
| `redact`       | `CPF [CPF]`               | Padrão, mais seguro                   |
| `partial`      | `CPF ***.456.789-**`      | Telas de atendimento                  |
| `pseudonym`    | `CPF <CPF_1>`             | Mesmo valor → mesmo rótulo            |
| `synthetic`    | `CPF 987.654.321-00`      | Dados de teste realistas (valor válido) |

O pseudônimo usa HMAC-SHA256 com um sal secreto: o espaço de CPFs é pequeno o bastante
para força bruta, então um hash sem sal seria reversível.
"""

from __future__ import annotations

import hashlib
import hmac
import random
import re
import secrets
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

from pii_br import validators as v
from pii_br.types import Entity, Span

__all__ = ["STRATEGIES", "Masker", "Strategy", "mask"]

Strategy = Literal["redact", "partial", "pseudonym", "synthetic"]
STRATEGIES: tuple[Strategy, ...] = ("redact", "partial", "pseudonym", "synthetic")

_SYNTHETIC_NAMES = (
    "Ana Lima", "Bruno Costa", "Carla Dias", "Diego Rocha", "Elisa Prado", "Fábio Nunes",
    "Gabriela Reis", "Hugo Martins", "Íris Campos", "João Freitas", "Kátia Moura", "Luís Teles",
)  # fmt: skip
_SYNTHETIC_STREETS = ("Rua Exemplo", "Avenida Modelo", "Travessa Fictícia", "Alameda Teste")


@dataclass
class Masker:
    """Mascara textos de uma sessão. O sal e o gerador vivem enquanto a sessão durar, para
    que o mesmo valor receba o mesmo pseudônimo em todos os textos (ex.: linhas de um CSV)."""

    salt: bytes = field(default_factory=lambda: secrets.token_bytes(32))
    rng: random.Random = field(default_factory=lambda: random.Random(secrets.randbits(64)))
    _pseudonyms: dict[tuple[Entity, str], str] = field(default_factory=lambda: {})
    _counters: dict[Entity, int] = field(default_factory=lambda: {})

    def mask(self, text: str, spans: Iterable[Span], strategy: Strategy = "redact") -> str:
        ordered = sorted(spans, key=lambda s: s.start)
        # Substituições calculadas na ordem do texto (pseudônimos numerados por aparição)
        # e aplicadas de trás para frente (os offsets continuam valendo).
        news = [self.replacement(s.label, text[s.start : s.end], strategy) for s in ordered]
        out = text
        for span, new in reversed(list(zip(ordered, news, strict=True))):
            out = out[: span.start] + new + out[span.end :]
        return out

    def replacement(self, label: Entity, value: str, strategy: Strategy) -> str:
        if strategy == "redact":
            return f"[{label.value}]"
        if strategy == "partial":
            return _partial(label, value)
        if strategy == "pseudonym":
            return self._pseudonym(label, value)
        # Sorteia de novo se sair o próprio valor original (ex.: um nome da lista fixa).
        for _ in range(20):
            new = self._synthetic(label, value)
            if _normalize(label, new) != _normalize(label, value):
                return new
        return f"[{label.value}]"

    # ----------------------------------------------------------------- pseudônimo
    def _pseudonym(self, label: Entity, value: str) -> str:
        digest = hmac.new(
            self.salt, f"{label.value}:{_normalize(label, value)}".encode(), hashlib.sha256
        )
        key = (label, digest.hexdigest())
        if key not in self._pseudonyms:
            self._counters[label] = self._counters.get(label, 0) + 1
            self._pseudonyms[key] = f"<{label.value}_{self._counters[label]}>"
        return self._pseudonyms[key]

    # ------------------------------------------------------------------ sintético
    def _synthetic(self, label: Entity, value: str) -> str:
        rng = self.rng
        if label is Entity.CPF:
            formatted = "." in value or "-" in value
            return v.generate_cpf(rng, formatted=formatted)
        if label is Entity.CNPJ:
            alnum = any(c.isalpha() for c in value)
            return v.generate_cnpj(rng, alphanumeric=alnum, formatted="/" in value)
        if label is Entity.TELEFONE:
            return _synthetic_phone(rng, value)
        if label is Entity.EMAIL:
            name = rng.choice(_SYNTHETIC_NAMES).split()[0].lower()
            return f"{_ascii(name)}.{rng.randint(10, 99)}@example.com"
        if label is Entity.NOME:
            return rng.choice(_SYNTHETIC_NAMES)
        if label is Entity.ENDERECO:
            return f"{rng.choice(_SYNTHETIC_STREETS)}, {rng.randint(1, 999)}"
        if label is Entity.DATA_NASC:
            return _synthetic_date(rng, value)
        return _replace_digits(value, rng)  # CEP, RG


def mask(text: str, spans: Iterable[Span], strategy: Strategy = "redact") -> str:
    """Atalho para um texto isolado (sal e gerador novos a cada chamada)."""
    return Masker().mask(text, spans, strategy)


# --------------------------------------------------------------------- helpers


def _normalize(label: Entity, value: str) -> str:
    if label in (Entity.CPF, Entity.CEP):
        return v.only_digits(value)
    if label is Entity.CNPJ:
        return v.normalize_cnpj(value)
    if label is Entity.TELEFONE:
        return v.phone_digits(value)
    if label is Entity.RG:
        return re.sub(r"[^0-9A-Za-z]", "", value).upper()
    return " ".join(_ascii(value).casefold().split())


def _ascii(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def _mask_positions(value: str, keep: range) -> str:
    """Troca por `*` os caracteres alfanuméricos fora de `keep` (posição contada só entre
    os alfanuméricos), preservando a pontuação."""
    out: list[str] = []
    i = 0
    for c in value:
        if c.isalnum():
            out.append(c if i in keep else "*")
            i += 1
        else:
            out.append(c)
    return "".join(out)


def _partial(label: Entity, value: str) -> str:
    n = sum(c.isalnum() for c in value)
    if label is Entity.CPF:
        return _mask_positions(value, range(3, 9))
    if label is Entity.CNPJ:
        return _mask_positions(value, range(2, 8))
    if label is Entity.TELEFONE:
        return _mask_positions(value, range(n - 4, n))
    if label is Entity.CEP:
        return _mask_positions(value, range(0, 5))
    if label is Entity.RG:
        return _mask_positions(value, range(n - 2, n))
    if label is Entity.EMAIL:
        local, _, domain = value.partition("@")
        return f"{local[:1]}***@{domain}"
    if label is Entity.DATA_NASC:
        m = re.search(r"\d{4}", value)
        return f"**/**/{m[0]}" if m else "[DATA_NASC]"
    return f"[{label.value}]"  # NOME e ENDERECO não têm parte "segura"


def _synthetic_phone(rng: random.Random, value: str) -> str:
    digits = v.only_digits(value)
    new = [rng.choice("0123456789") for _ in digits]
    body_start = 0
    if len(digits) in (12, 13) and digits.startswith("55"):
        new[0], new[1] = "5", "5"
        body_start = 2
    if len(digits) - body_start in (10, 11):
        ddd = rng.choice(sorted(v.VALID_DDDS))
        new[body_start], new[body_start + 1] = ddd[0], ddd[1]
        body_start += 2
    if len(digits) - body_start == 9:
        new[body_start] = "9"
    elif len(digits) - body_start == 8:
        new[body_start] = rng.choice("2345")
    it = iter(new)
    return "".join(next(it) if c.isdigit() else c for c in value)


def _synthetic_date(rng: random.Random, value: str) -> str:
    """Data aleatória no mesmo formato numérico (dd/mm/aaaa, dd-mm-aa...) quando possível."""
    year, month, day = rng.randint(1950, 2005), rng.randint(1, 12), rng.randint(1, 28)
    m = re.fullmatch(r"\s*\d{1,2}([/.\-])\d{1,2}[/.\-](\d{2}|\d{4})\s*", value)
    if m is None:
        return f"{day:02d}/{month:02d}/{year}"
    year_text = f"{year % 100:02d}" if len(m[2]) == 2 else str(year)
    return f"{day:02d}{m[1]}{month:02d}{m[1]}{year_text}"


def _replace_digits(value: str, rng: random.Random) -> str:
    return "".join(rng.choice("0123456789") if c.isdigit() else c for c in value)
