"""Modelo NER (BERTimbau fine-tuned) como fonte de spans. Requer o extra `pii-br[model]`.

O texto inteiro passa pelo pipeline do `transformers` em janelas de 256 tokens com
sobreposição de 64 (`stride`), então textos longos não são truncados. Por padrão o modelo é
quantizado em int8 dinâmico, que em CPU corta a latência para cerca de 1/3 (ver SPEC.md).
"""

# torch e transformers não têm tipos completos; as regras de "Unknown" do pyright estrito
# ficam desligadas só neste arquivo, que é a única ponte com essas bibliotecas.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportDeprecated=false

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, cast

from pii_br.types import Entity, Span

if TYPE_CHECKING:
    from transformers import Pipeline

__all__ = ["NerModel"]

WINDOW = 256
STRIDE = 64


class NerModel:
    """Carrega um modelo de token classification do Hub (ou de uma pasta local)."""

    def __init__(
        self,
        model_id: str,
        *,
        quantize: bool = True,
        threads: int | None = None,
        revision: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

        if threads:
            torch.set_num_threads(threads)
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        tokenizer.model_max_length = WINDOW
        model = AutoModelForTokenClassification.from_pretrained(model_id, revision=revision)
        model.eval()
        if quantize:
            # A API eager de quantização está depreciada no torch, mas segue funcional e é a mais
            # simples para CPU; o aviso é silenciado para não poluir o log do Space.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = torch.ao.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )
        self.model_id = model_id
        self._pipe: Pipeline = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="first",
            stride=STRIDE,
            device=-1,
        )

    def predict(self, text: str) -> list[Span]:
        if not text.strip():
            return []
        raw = cast(list[dict[str, Any]], self._pipe(text))
        spans: list[Span] = []
        for item in raw:
            label = str(item["entity_group"])
            if label not in Entity.__members__:
                continue
            start, end = int(item["start"]), int(item["end"])
            while start < end and text[start].isspace():
                start += 1
            while end > start and text[end - 1].isspace():
                end -= 1
            if start < end:
                spans.append(
                    Span(start, end, Entity(label), text[start:end], "modelo", float(item["score"]))
                )
        return _join_adjacent(spans, text)


def _joinable(label: Entity, gap: str) -> bool:
    """Pedaços vizinhos do mesmo rótulo que o pipeline às vezes separa. Vírgula só une
    endereço ("Rua A, 12"): entre nomes ("Ana, Bruno") ela separa duas pessoas."""
    if len(gap) > 2:
        return False
    allowed = " ,-" if label is Entity.ENDERECO else " -"
    return not gap.strip(allowed)


def _join_adjacent(spans: list[Span], text: str) -> list[Span]:
    """Une pedaços vizinhos do mesmo rótulo (ver `_joinable`)."""
    out: list[Span] = []
    for span in sorted(spans, key=lambda s: s.start):
        if out:
            prev = out[-1]
            gap = text[prev.end : span.start]
            if prev.label is span.label and _joinable(span.label, gap):
                score = min(prev.score, span.score)
                out[-1] = Span(
                    prev.start, span.end, prev.label, text[prev.start : span.end], "modelo", score
                )
                continue
        out.append(span)
    return out
