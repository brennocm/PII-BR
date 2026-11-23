"""Pipeline híbrido: recognizers determinísticos + modelo NER + fusão."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from pii_br.merge import merge
from pii_br.recognizers import recognize
from pii_br.types import Entity, Span

__all__ = ["Detector", "SpanModel"]


class SpanModel(Protocol):
    def predict(self, text: str) -> list[Span]: ...


class Detector:
    """Detecta PII com regex + validação e, se houver modelo, NER.

    `threshold` filtra os spans de baixa confiança. Alertas de documento inválido
    (`valid=False`) nunca são filtrados: em privacidade, mascarar a mais é o erro barato.
    """

    def __init__(self, model: SpanModel | None = None, *, threshold: float = 0.5) -> None:
        self.model = model
        self.threshold = threshold

    def detect(self, text: str, entities: Iterable[Entity] | None = None) -> list[Span]:
        wanted = frozenset(Entity) if entities is None else frozenset(entities)
        regex_spans = recognize(text)
        model_spans = self.model.predict(text) if self.model else []
        spans = merge(regex_spans, model_spans)
        return [
            s
            for s in spans
            if s.label in wanted and (s.score >= self.threshold or s.valid is False)
        ]
