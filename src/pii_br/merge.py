"""Fusão dos spans da regex com os do modelo.

Regras, aplicadas a cada span do modelo contra os spans da regex que ele sobrepõe:
1. Mesmo rótulo: fica um span só, com fonte `ambos`, as bordas da regex (mais precisas para
   formatos fixos) e o maior score.
2. Regex de CPF/CNPJ com dígito verificador válido vence (DDD ou faixa de CEP válidos são
   sinais fracos demais para isso).
3. O modelo vence em `NOME`, `ENDERECO` e `RG`, que dependem de contexto.
4. Nos demais casos fica o span mais longo; no empate, a regex.

Invariante de privacidade: a fusão nunca descobre um caractere que algum span cobria. O
modelo só substitui um span da regex que ele cobre por inteiro; se cobre só uma parte, a
regex fica e o modelo entra só com o que sobra fora dela (ex.: o modelo marcando `-100` de
`CEP 01310-100` como endereço não pode deixar `01310` à mostra).
"""

from __future__ import annotations

from dataclasses import replace

from pii_br.types import Entity, Span

__all__ = ["MODEL_FIRST", "merge"]

MODEL_FIRST: frozenset[Entity] = frozenset({Entity.NOME, Entity.ENDERECO, Entity.RG})
CHECKSUM: frozenset[Entity] = frozenset({Entity.CPF, Entity.CNPJ})
_TRIM = " \t\n,;:-./()"


def merge(regex_spans: list[Span], model_spans: list[Span]) -> list[Span]:
    """Une as duas listas em spans sem sobreposição, ordenados por posição."""
    result = list(regex_spans)
    for m in sorted(model_spans, key=lambda s: (-s.score, -len(s), s.start)):
        rivals = [r for r in result if r.overlaps(m)]
        keep = [r for r in rivals if _regex_wins(r, m) or not _covers(m, r)]
        result = [r for r in result if r not in rivals or r in keep]
        if not keep:
            result.append(m)
            continue
        for r in keep:
            if r.label is m.label and r.source == "regex":
                result[result.index(r)] = replace(r, source="ambos", score=max(r.score, m.score))
        result.extend(_outside(m, keep))
    return sorted(result, key=lambda s: s.start)


def _covers(outer: Span, inner: Span) -> bool:
    return outer.start <= inner.start and inner.end <= outer.end


def _outside(m: Span, blockers: list[Span]) -> list[Span]:
    """Os pedaços de `m` fora dos `blockers`, sem bordas de espaço ou pontuação."""
    pieces: list[Span] = []
    cursor = m.start
    end_marker = Span(m.end, m.end, m.label, "", "modelo", 0)
    for b in [*sorted(blockers, key=lambda s: s.start), end_marker]:
        start, end = cursor, min(b.start, m.end)
        cursor = max(cursor, b.end)
        if start >= end:
            continue
        raw = m.text[start - m.start : end - m.start]
        left = len(raw) - len(raw.lstrip(_TRIM))
        text = raw.strip(_TRIM)
        if any(c.isalnum() for c in text):
            pieces.append(replace(m, start=start + left, end=start + left + len(text), text=text))
    return pieces


def _regex_wins(r: Span, m: Span) -> bool:
    if r.label is m.label:
        return True
    if r.valid is True and r.label in CHECKSUM and r.source != "modelo":
        return True
    if m.label in MODEL_FIRST:
        return False
    return len(r) >= len(m)
