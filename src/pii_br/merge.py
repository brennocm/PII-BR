"""Fusão dos spans da regex com os do modelo.

Regras, na ordem em que são aplicadas a cada par sobreposto (regex contra modelo):
1. Mesmo rótulo: fica um span só, com fonte `ambos`, as bordas da regex (mais precisas para
   formatos fixos) e o maior score.
2. Regex de CPF/CNPJ com dígito verificador válido vence (DDD ou faixa de CEP válidos são
   sinais fracos demais para isso).
3. O modelo vence em `NOME`, `ENDERECO` e `RG`, que dependem de contexto.
4. Nos demais casos fica o span mais longo; no empate, a regex.
"""

from __future__ import annotations

from dataclasses import replace

from pii_br.types import Entity, Span

__all__ = ["MODEL_FIRST", "merge"]

MODEL_FIRST: frozenset[Entity] = frozenset({Entity.NOME, Entity.ENDERECO, Entity.RG})
CHECKSUM: frozenset[Entity] = frozenset({Entity.CPF, Entity.CNPJ})


def merge(regex_spans: list[Span], model_spans: list[Span]) -> list[Span]:
    """Une as duas listas em spans sem sobreposição, ordenados por posição."""
    result = list(regex_spans)
    for m in sorted(model_spans, key=lambda s: (-s.score, -len(s), s.start)):
        rivals = [r for r in result if r.overlaps(m)]
        if not rivals:
            result.append(m)
            continue
        if all(_regex_wins(r, m) for r in rivals):
            for r in rivals:
                if r.label is m.label and r.source == "regex":
                    result[result.index(r)] = replace(
                        r, source="ambos", score=max(r.score, m.score)
                    )
            continue
        if any(_regex_wins(r, m) and r.label is not m.label for r in rivals):
            continue  # m perde para ao menos um rival forte; nada muda
        result = [r for r in result if r not in rivals]
        result.append(m)
    return sorted(result, key=lambda s: s.start)


def _regex_wins(r: Span, m: Span) -> bool:
    if r.label is m.label:
        return True
    if r.valid is True and r.label in CHECKSUM and r.source != "modelo":
        return True
    if m.label in MODEL_FIRST:
        return False
    return len(r) >= len(m)
