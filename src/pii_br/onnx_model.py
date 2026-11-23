"""Modelo NER em ONNX Runtime (CPU). Requer o extra `pii-br[onnx]`.

É o mesmo caminho da demo no navegador, que usa `onnxruntime-web`: tokenização, janelas e
decodificação vêm de `pii_br.ner` (Python puro); só a inferência muda de runtime. Em Python
serve para conferir o modelo exportado (`training/export_onnx.py`) e para a avaliação.
"""

# onnxruntime não tem tipos completos.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeStubs=false

from __future__ import annotations

from pathlib import Path
from typing import Any

from pii_br.ner import WordPiece, predict
from pii_br.types import Span

__all__ = ["OnnxModel"]


class OnnxModel:
    def __init__(self, onnx_path: str | Path, tokenizer_json: str | Path) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.log_severity_level = 3
        self._session = ort.InferenceSession(
            str(onnx_path), options, providers=["CPUExecutionProvider"]
        )
        self.tokenizer = WordPiece.from_tokenizer_json(
            Path(tokenizer_json).read_text(encoding="utf-8")
        )

    def logits(self, ids: list[int]) -> list[list[float]]:
        import numpy as np

        feed = {
            "input_ids": np.array([ids], dtype=np.int64),
            "attention_mask": np.ones((1, len(ids)), dtype=np.int64),
            "token_type_ids": np.zeros((1, len(ids)), dtype=np.int64),
        }
        out: Any = self._session.run(["logits"], feed)[0]
        return out[0].tolist()

    def predict(self, text: str) -> list[Span]:
        return predict(self.tokenizer, self.logits, text)
