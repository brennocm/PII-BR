"""Modelo NER (BERTimbau fine-tuned) em PyTorch como fonte de spans. Requer `pii-br[model]`.

Tokenização, janelas de 256 tokens com sobreposição de 64 e decodificação vêm de
`pii_br.ner`, os mesmos da versão ONNX (`onnx_model.py`) e da demo no navegador; aqui só a
inferência é em torch. Por padrão o modelo é quantizado em int8 dinâmico, que em CPU corta a
latência para menos da metade (ver SPEC.md), exceto a saída do FFN, que perde a precisão em
int8 (ver `quantizable`).
"""

# torch e transformers não têm tipos completos; as regras de "Unknown" do pyright estrito
# ficam desligadas só neste arquivo, que é a única ponte com essas bibliotecas.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportDeprecated=false

from __future__ import annotations

import re
import warnings
from typing import Any

from pii_br.ner import WordPiece, predict
from pii_br.types import Span

__all__ = ["NerModel", "quantizable"]

# Saída do FFN de cada camada (`encoder.layer.N.output.dense`). A entrada dela é a ativação do
# GELU (3.072 dimensões) com outliers; o int8 dinâmico usa uma escala por tensor e destrói a
# precisão: no modelo publicado, nomes somem e o CPF sai partido com score 0,2. Fica em fp32.
_FFN_OUT = re.compile(r"\.layer\.\d+\.output\.dense$")


def quantizable(name: str) -> bool:
    """Se a camada `Linear` chamada `name` pode ir para int8 dinâmico (ver `_FFN_OUT`)."""
    return _FFN_OUT.search(name) is None


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
        from torch.ao.quantization.qconfig import default_dynamic_qconfig
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        if threads:
            torch.set_num_threads(threads)
        fast = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.tokenizer = WordPiece.from_tokenizer_json(fast.backend_tokenizer.to_str())
        model = AutoModelForTokenClassification.from_pretrained(model_id, revision=revision)
        model.eval()
        if quantize:
            # A API eager de quantização está depreciada no torch, mas segue funcional e é a mais
            # simples para CPU; o aviso é silenciado para não poluir o log do Space.
            layers = {
                name: default_dynamic_qconfig
                for name, module in model.named_modules()
                if isinstance(module, torch.nn.Linear) and quantizable(name)
            }
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = torch.ao.quantization.quantize_dynamic(model, layers)
        self.model_id = model_id
        self._model: Any = model

    def logits(self, ids: list[int]) -> list[list[float]]:
        import torch

        with torch.inference_mode():
            t = torch.tensor([ids])
            out = self._model(
                input_ids=t, attention_mask=torch.ones_like(t), token_type_ids=torch.zeros_like(t)
            )
        return out.logits[0].tolist()

    def predict(self, text: str) -> list[Span]:
        return predict(self.tokenizer, self.logits, text)
