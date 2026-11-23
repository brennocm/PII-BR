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


# Rótulos BIO do dataset e do modelo, na ordem dos ids (O = 0). Mudar a ordem invalida
# modelos já treinados.
BIO_LABELS: tuple[str, ...] = ("O", *(f"{p}-{e.value}" for e in Entity for p in ("B", "I")))


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
