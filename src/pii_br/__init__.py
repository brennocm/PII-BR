"""Detecção de PII em português do Brasil: validadores, recognizers, fusão e mascaramento."""

from pii_br.detector import Detector
from pii_br.masking import STRATEGIES, Masker, Strategy, mask
from pii_br.merge import merge
from pii_br.recognizers import DETERMINISTIC_ENTITIES, recognize
from pii_br.types import BIO_LABELS, Entity, Span

__all__ = [
    "BIO_LABELS",
    "DETERMINISTIC_ENTITIES",
    "STRATEGIES",
    "Detector",
    "Entity",
    "Masker",
    "Span",
    "Strategy",
    "mask",
    "merge",
    "recognize",
]
__version__ = "0.1.0"
