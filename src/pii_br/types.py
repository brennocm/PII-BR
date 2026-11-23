"""Tipos compartilhados: rótulos de entidade e spans detectados."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

Source = Literal["regex", "modelo", "ambos"]


class Entity(StrEnum):
    """As 9 entidades do MVP. O valor é o rótulo usado no dataset (BIO) e na saída."""

    NOME = "NOME"
    CPF = "CPF"
    CNPJ = "CNPJ"
    RG = "RG"
    TELEFONE = "TELEFONE"
    EMAIL = "EMAIL"
    ENDERECO = "ENDERECO"
    CEP = "CEP"
    DATA_NASC = "DATA_NASC"


@dataclass(frozen=True, slots=True)
class Span:
    """Trecho de PII em offsets de caractere sobre o texto original (fim exclusivo).

    `valid` é None quando a entidade não tem validação determinística, True quando o
    validador confirmou o valor e False para formato certo com validação falha
    (ex.: CPF com dígito verificador errado), que vira alerta e é mascarado por padrão.
    """

    start: int
    end: int
    label: Entity
    text: str
    source: Source
    score: float
    valid: bool | None = None

    def overlaps(self, other: Span) -> bool:
        return self.start < other.end and other.start < self.end

    def __len__(self) -> int:
        return self.end - self.start
