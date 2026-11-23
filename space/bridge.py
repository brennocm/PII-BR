"""Ponte entre a página (JavaScript) e o pacote `pii_br`, rodando no Pyodide, no navegador.

O texto do usuário nunca sai do navegador: a página tokeniza com `encode`, roda o modelo
ONNX no `onnxruntime-web` e devolve os logits para `analyze`, que decodifica, funde com as
regras determinísticas e mascara. Sem modelo (ainda baixando ou indisponível), `analyze`
recebe `logits=None` e usa só as regras.

As funções trocam JSON (strings) com o JavaScript para não depender de conversões de tipo
do Pyodide.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from pii_br import Detector, Entity, Masker, Span, Strategy
from pii_br.ner import Encoding, WordPiece, decode

MAX_CHARS = 20_000
N_LABELS = 19  # len(BIO_LABELS)
STRATEGIES: tuple[Strategy, ...] = ("redact", "partial", "pseudonym", "synthetic")


class _Session:
    tokenizer: WordPiece | None = None
    masker: Masker | None = None  # um sal por aba, criado na primeira análise
    last: Encoding | None = None


_session = _Session()


def load_tokenizer(tokenizer_json: str) -> None:
    _session.tokenizer = WordPiece.from_tokenizer_json(tokenizer_json)


def encode(text: str) -> str:
    """Janelas de ids do texto (já cortado em `MAX_CHARS`) como JSON: `[[ids...], ...]`."""
    if _session.tokenizer is None:
        raise RuntimeError("tokenizer não carregado")
    _session.last = _session.tokenizer.encode(text[:MAX_CHARS])
    return json.dumps([ids for _, ids in _session.last.windows])


class _Fixed:
    """Modelo cujos spans já foram calculados (o `Detector` só precisa de `predict`)."""

    def __init__(self, spans: list[Span]) -> None:
        self.spans = spans

    def predict(self, text: str) -> list[Span]:
        return self.spans


def _reshape(flat: Sequence[float], n_tokens: int) -> list[list[float]]:
    return [list(flat[i * N_LABELS : (i + 1) * N_LABELS]) for i in range(n_tokens)]


def analyze(request_json: str) -> str:
    """Detecta e mascara. O pedido traz `text`, `strategy`, `entities`, `threshold` e
    `logits`: a saída achatada do modelo para cada janela de `encode(text)`, na mesma ordem,
    ou `null` para usar só as regras."""
    request: dict[str, Any] = json.loads(request_json)
    raw_text: str = request["text"]
    strategy: str = request["strategy"]
    if strategy not in STRATEGIES:
        raise ValueError(f"estratégia desconhecida: {strategy}")
    cut = len(raw_text) > MAX_CHARS
    text = raw_text[:MAX_CHARS]
    wanted = [Entity(e) for e in request["entities"]]
    logits: list[list[float]] | None = request.get("logits")
    model = None
    if logits is not None:
        encoding = _session.last
        if encoding is None or encoding.text != text:
            raise RuntimeError("logits de outro texto: chame encode antes")
        windows = [
            _reshape(flat, len(ids))
            for flat, (_, ids) in zip(logits, encoding.windows, strict=True)
        ]
        model = _Fixed(decode(encoding, windows))
    spans = Detector(model, threshold=float(request["threshold"])).detect(text, wanted)
    _session.masker = _session.masker or Masker()
    masked = _session.masker.mask(text, spans, strategy)
    result: dict[str, Any] = {
        "masked": masked,
        "cut": cut,
        "chars": len(text),
        "spans": [
            {
                "start": s.start,
                "end": s.end,
                "label": s.label.value,
                "text": s.text,
                "source": s.source,
                "score": round(s.score, 3),
                "valid": s.valid,
            }
            for s in spans
        ],
    }
    return json.dumps(result, ensure_ascii=False)
