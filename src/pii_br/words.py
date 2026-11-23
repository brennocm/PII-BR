"""Tokenização por palavras usada no dataset (BIO) e na avaliação.

Regra: uma palavra é uma sequência de letras/dígitos que pode conter `.`, `-`, `/`, `@`,
`+` e apóstrofo *entre* caracteres de palavra (então `123.456.789-09`,
`maria.souza@example.com` e `d'água` são um token só); qualquer outro caractere que não
seja espaço vira um token próprio (`(`, `)`, `,`, `:`, emojis).
"""

from __future__ import annotations

import re

__all__ = ["WORD_PATTERN", "word_spans", "words"]

WORD_PATTERN = re.compile(r"\w+(?:[.\-/@+'\u2019]\w+)*|[^\w\s]")


def word_spans(text: str) -> list[tuple[int, int]]:
    """Offsets (início, fim) de cada palavra em `text`."""
    return [(m.start(), m.end()) for m in WORD_PATTERN.finditer(text)]


def words(text: str) -> list[str]:
    return WORD_PATTERN.findall(text)
