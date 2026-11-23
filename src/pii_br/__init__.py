"""Detecção de PII em português do Brasil: validadores, recognizers, fusão e mascaramento."""

from pii_br.recognizers import DETERMINISTIC_ENTITIES, recognize
from pii_br.types import Entity, Span

__all__ = ["DETERMINISTIC_ENTITIES", "Entity", "Span", "recognize"]
__version__ = "0.1.0"
