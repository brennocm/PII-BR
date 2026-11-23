"""Pipeline híbrido: recognizers determinísticos + modelo NER + fusão."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import replace
from typing import Protocol

from pii_br.merge import merge
from pii_br.recognizers import recognize
from pii_br.types import Entity, Span

__all__ = ["Detector", "SpanModel"]


class SpanModel(Protocol):
    def predict(self, text: str) -> list[Span]: ...


class Detector:
    """Detecta PII com regex + validação e, se houver modelo, NER.

    `threshold` vale só para o modelo: os spans dele abaixo do limiar são descartados
    *antes* da fusão, para que um palpite fraco nunca tome o lugar de um achado da regex e
    depois suma no filtro (o que deixaria o dado original sem máscara). Os spans da regex
    já são validados e não passam por limiar.
    """

    def __init__(self, model: SpanModel | None = None, *, threshold: float = 0.5) -> None:
        self.model = model
        self.threshold = threshold

    def detect(self, text: str, entities: Iterable[Entity] | None = None) -> list[Span]:
        wanted = frozenset(Entity) if entities is None else frozenset(entities)
        regex_spans = recognize(text, wanted)
        model_spans = [
            s
            for s in (self.model.predict(text) if self.model else [])
            if s.label in wanted and s.score >= self.threshold
        ]
        return _complete_dates(text, merge(regex_spans, model_spans))


# Data inteira (numérica ou por extenso) em volta de um pedaço marcado como DATA_NASC.
_DATE = re.compile(
    r"(?<!\d)\d{1,2}[/.\-]\d{1,2}[/.\-](?:\d{4}|\d{2})(?!\d)"
    r"|(?<!\d)\d{1,2}º?\s+de\s+[a-zç]+\.?\s+de\s+\d{4}(?!\d)",
    re.IGNORECASE,
)


def _complete_dates(text: str, spans: list[Span]) -> list[Span]:
    """Estende um `DATA_NASC` parcial do modelo (ex.: só o dia de 14/06/1992) até a data
    inteira, para não deixar mês e ano à mostra. Só estende sobre texto ainda não marcado."""
    dates = [s for s in spans if s.label is Entity.DATA_NASC]
    if not dates:
        return spans
    out = list(spans)
    for m in _DATE.finditer(text):
        inside = [s for s in dates if s.start < m.end() and m.start() < s.end]
        if not inside:
            continue
        start, end = (
            min(m.start(), *(s.start for s in inside)),
            max(m.end(), *(s.end for s in inside)),
        )
        others = [s for s in out if s not in inside and s.start < end and start < s.end]
        if others:
            continue
        grown = replace(inside[0], start=start, end=end, text=text[start:end])
        out = [s for s in out if s not in inside] + [grown]
    return sorted(out, key=lambda s: s.start)
