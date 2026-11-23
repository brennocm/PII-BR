"""Recognizers determinísticos: regex + validadores + palavras de contexto.

Cobrem CPF, CNPJ (numérico e alfanumérico), telefone, e-mail e CEP. Nome, endereço, RG e
data de nascimento dependem de contexto semântico e ficam com o modelo.

Regras para números sem pontuação (onde mora a ambiguidade):
- 11 dígitos: CPF ou celular com DDD. Decide a palavra de contexto mais próxima ("CPF",
  "tel", "whats"); sem contexto, vira CPF se o dígito verificador fecha e TELEFONE se o
  número é um celular válido.
- 14 dígitos: CNPJ se o dígito fecha, ou alerta de CNPJ inválido se houver contexto.
- 8 dígitos: CEP ou telefone fixo, só com palavra de contexto.

Formato certo com dígito verificador errado (ex.: 123.456.789-00) sai com `valid=False`:
é um alerta de "possível documento inválido", e o mascaramento o trata como PII.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Iterable, Iterator

from pii_br import validators as v
from pii_br.types import Entity, Span

__all__ = ["DETERMINISTIC_ENTITIES", "recognize"]

DETERMINISTIC_ENTITIES: frozenset[Entity] = frozenset(
    {Entity.CPF, Entity.CNPJ, Entity.TELEFONE, Entity.EMAIL, Entity.CEP}
)

SCORE_VALID = 0.99
SCORE_CONTEXT = 0.85
SCORE_INVALID = 0.6

# Fronteiras: o número não pode continuar um token maior (letra, dígito, "1.2" ou "1-2").
_START = r"(?<![\w@])(?<!\d[./-])"
_END = r"(?![\w@])(?![./-]\d)"

_CPF_FORMATTED = re.compile(
    _START + r"(?:\d{3}\.\d{3}\.\d{3}\s?-\s?\d{2}|\d{3}\s\d{3}\s\d{3}[\s-]\d{2}|\d{9}-\d{2})" + _END
)
_CNPJ_FORMATTED = re.compile(
    _START + r"[0-9A-Z]{2}\.?[0-9A-Z]{3}\.?[0-9A-Z]{3}\s?/\s?[0-9A-Z]{4}\s?-?\s?\d{2}" + _END,
    re.IGNORECASE,
)
_CNPJ_ALNUM_BARE = re.compile(_START + r"[0-9A-Z]{12}\d{2}" + _END, re.IGNORECASE)
_DIGIT_RUN = re.compile(_START + r"\+?\d{8,14}" + _END)
_PHONE_FORMATTED = re.compile(
    _START
    + r"(?:\+\s?55\s?(?:-\s?)?)?"
    + r"(?:\(\s?0?\d{2}\s?\)\s?|0?\d{2}[\s.-]\s?)?"
    + r"(?:9[\s.]?)?\d{4}[\s.-]?\d{4}"
    + _END
)
_CEP_FORMATTED = re.compile(_START + r"(?:\d{5}-\d{3}|\d{2}\.\d{3}-\d{3})" + _END)
_EMAIL = re.compile(
    r"(?<![\w.+-])[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*"
    r"@(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?\.)+[A-Za-z]{2,}(?![\w-])"
)

_EMAIL_LOCAL_MAX = 64  # RFC 5321
_EMAIL_DOMAIN_MAX = 255

_CONTEXT_WORDS: dict[Entity, re.Pattern[str]] = {
    Entity.CPF: re.compile(r"\bc\.?p\.?f\b"),
    Entity.CNPJ: re.compile(r"\bcnpj\b"),
    Entity.TELEFONE: re.compile(
        r"\b(?:tel\w*|fone|celular|cel|whats\w*|zap|wpp|contato|liga\w*|ligue|fixo)\b"
    ),
    Entity.CEP: re.compile(r"\bcep\b"),
}
_CONTEXT_BEFORE = 30
_CONTEXT_AFTER = 15

# Dígitos de largura total (comuns em texto copiado de PDF/planilha) viram ASCII.
# A troca é 1 por 1, então os offsets continuam valendo para o texto original.
_FULLWIDTH = str.maketrans({chr(0xFF10 + i): str(i) for i in range(10)})


def recognize(text: str, entities: Iterable[Entity] | None = None) -> list[Span]:
    """Spans determinísticos em `text`, sem sobreposição, ordenados por posição."""
    wanted = DETERMINISTIC_ENTITIES if entities is None else frozenset(entities)
    norm = text.translate(_FULLWIDTH)
    finders: list[Callable[[str], Iterator[Span]]] = [
        _find_email,
        _find_cpf_formatted,
        _find_cnpj_formatted,
        _find_cnpj_alnum_bare,
        _find_digit_runs,
        _find_phone_formatted,
        _find_cep_formatted,
    ]
    candidates = [s for find in finders for s in find(norm) if s.label in wanted]
    spans = _resolve_overlaps(candidates)
    return [_with_original_text(s, text) for s in spans]


# ---------------------------------------------------------------------- finders


def _find_email(text: str) -> Iterator[Span]:
    # A regex só roda numa janela em volta de cada "@". Rodada no texto inteiro, ela tenta
    # de novo a partir de cada posição e fica quadrática em textos sem "@" (ReDoS).
    # `pos` não corta o texto, então o lookbehind ainda enxerga o que vem antes da janela.
    last_end = 0
    for at in re.finditer("@", text):
        if at.start() < last_end:
            continue  # "@" dentro de um e-mail já encontrado
        lo = max(last_end, at.start() - _EMAIL_LOCAL_MAX - 1)
        hi = min(len(text), at.end() + _EMAIL_DOMAIN_MAX)
        m = _EMAIL.search(text, lo, hi)
        while m is not None and m.end() <= at.start():
            m = _EMAIL.search(text, m.end(), hi)
        if m is not None and m.start() <= at.start() and v.is_valid_email(m[0]):
            last_end = m.end()
            yield _span(m, Entity.EMAIL, SCORE_VALID, valid=True)


def _find_cpf_formatted(text: str) -> Iterator[Span]:
    for m in _CPF_FORMATTED.finditer(text):
        ok = v.is_valid_cpf(m[0])
        yield _span(m, Entity.CPF, SCORE_VALID if ok else SCORE_INVALID, valid=ok)


def _find_cnpj_formatted(text: str) -> Iterator[Span]:
    for m in _CNPJ_FORMATTED.finditer(text):
        if not any(c.isdigit() for c in m[0][:-2]):
            continue  # só letras na base: é texto, não documento
        ok = v.is_valid_cnpj(m[0])
        yield _span(m, Entity.CNPJ, SCORE_VALID if ok else SCORE_INVALID, valid=ok)


def _find_cnpj_alnum_bare(text: str) -> Iterator[Span]:
    for m in _CNPJ_ALNUM_BARE.finditer(text):
        base = m[0][:12]
        if base.isdigit() or not any(c.isdigit() for c in base):
            continue  # numérico puro fica com _find_digit_runs; só letras é palavra
        if v.is_valid_cnpj(m[0]):
            yield _span(m, Entity.CNPJ, SCORE_VALID, valid=True)


def _find_digit_runs(text: str) -> Iterator[Span]:
    for m in _DIGIT_RUN.finditer(text):
        digits = v.only_digits(m[0])
        context = _nearest_context(text, m.start(), m.end())
        if span := _classify_digit_run(m, digits, context):
            yield span


def _classify_digit_run(m: re.Match[str], digits: str, context: Entity | None) -> Span | None:
    n = len(digits)
    phone_ok = v.is_valid_phone(digits, require_ddd=n >= 10)
    if n == 14:
        if v.is_valid_cnpj(digits):
            return _span(m, Entity.CNPJ, SCORE_VALID, valid=True)
        if context is Entity.CNPJ:
            return _span(m, Entity.CNPJ, SCORE_INVALID, valid=False)
        return None
    if n == 11:
        cpf_ok = v.is_valid_cpf(digits)
        if context is Entity.CPF:
            return _span(m, Entity.CPF, SCORE_VALID if cpf_ok else SCORE_INVALID, valid=cpf_ok)
        if context is Entity.TELEFONE and phone_ok:
            return _span(m, Entity.TELEFONE, SCORE_VALID, valid=True)
        if cpf_ok:
            return _span(m, Entity.CPF, SCORE_CONTEXT, valid=True)
        if phone_ok:
            return _span(m, Entity.TELEFONE, SCORE_CONTEXT, valid=True)
        return None
    if n in (10, 12, 13) and phone_ok:
        return _span(m, Entity.TELEFONE, SCORE_VALID, valid=True)
    if n == 8 and context is Entity.CEP and v.is_valid_cep(digits):
        return _span(m, Entity.CEP, SCORE_CONTEXT, valid=True)
    if n in (8, 9) and context is Entity.TELEFONE and phone_ok:
        return _span(m, Entity.TELEFONE, SCORE_CONTEXT, valid=True)
    return None


def _find_phone_formatted(text: str) -> Iterator[Span]:
    for m in _PHONE_FORMATTED.finditer(text):
        raw = m[0]
        if raw.lstrip("+").isdigit():
            continue  # número corrido: já tratado por _find_digit_runs
        digits = v.phone_digits(raw)
        if v.is_service_number(raw) or v.is_service_number(digits):
            continue
        if len(digits) >= 10:
            if v.is_valid_phone(raw, require_ddd=True):
                yield _span(m, Entity.TELEFONE, SCORE_VALID, valid=True)
        elif v.is_valid_phone(raw):
            mobile = len(digits) == 9
            has_context = _nearest_context(text, m.start(), m.end()) is Entity.TELEFONE
            if mobile or has_context:
                yield _span(m, Entity.TELEFONE, SCORE_CONTEXT, valid=True)


def _find_cep_formatted(text: str) -> Iterator[Span]:
    for m in _CEP_FORMATTED.finditer(text):
        ok = v.is_valid_cep(m[0])
        has_context = _nearest_context(text, m.start(), m.end()) is Entity.CEP
        if ok or has_context:
            score = SCORE_VALID if ok and has_context else SCORE_CONTEXT if ok else SCORE_INVALID
            yield _span(m, Entity.CEP, score, valid=ok)


# ---------------------------------------------------------------------- helpers


def _span(m: re.Match[str], label: Entity, score: float, *, valid: bool | None) -> Span:
    return Span(m.start(), m.end(), label, m[0], "regex", score, valid)


def _with_original_text(span: Span, text: str) -> Span:
    original = text[span.start : span.end]
    if original == span.text:
        return span
    return Span(span.start, span.end, span.label, original, span.source, span.score, span.valid)


def _fold(text: str) -> str:
    """Minúsculas sem acento, preservando o comprimento (um caractere por caractere)."""
    out: list[str] = []
    for ch in text.lower():
        base = unicodedata.normalize("NFKD", ch)[:1]
        out.append(base if base else ch)
    return "".join(out)


def _nearest_context(text: str, start: int, end: int) -> Entity | None:
    """A entidade cuja palavra de contexto está mais perto do trecho [start, end)."""
    before = _fold(text[max(0, start - _CONTEXT_BEFORE) : start])
    after = _fold(text[end : end + _CONTEXT_AFTER])
    best: tuple[int, Entity] | None = None
    for entity, pattern in _CONTEXT_WORDS.items():
        for hit in pattern.finditer(before):
            distance = len(before) - hit.end()
            if best is None or distance < best[0]:
                best = (distance, entity)
        if later := pattern.search(after):
            distance = later.start() + _CONTEXT_BEFORE  # contexto depois pesa menos que antes
            if best is None or distance < best[0]:
                best = (distance, entity)
    return best[1] if best else None


def _resolve_overlaps(candidates: list[Span]) -> list[Span]:
    """Mantém o melhor de cada grupo sobreposto: maior score, depois o span mais longo."""
    kept: list[Span] = []
    for span in sorted(candidates, key=lambda s: (-s.score, -len(s), s.start)):
        if not any(span.overlaps(k) for k in kept):
            kept.append(span)
    return sorted(kept, key=lambda s: s.start)
