"""Exporta o modelo para ONNX int8 (demo no navegador) e confere contra o fp32.

Uso:
    uv run python -m training.export_onnx --model brennocm/PII-BR --data data
    uv run python -m training.export_onnx --model brennocm/PII-BR --data data --push

Gera `model.onnx` (fp32) e `model_int8.onnx` em `--out`, roda os dois no `test_real` e no
`test_hard` pelo mesmo caminho do navegador (`pii_br.onnx_model`) e falha se o int8 perder
mais de `--max-loss` ponto de F1. Com `--push`, envia `onnx/model_int8.onnx` ao repositório do
modelo (precisa de login com escrita no Hub).

Quantização: MatMul e Gather (embeddings) em int8 dinâmico, menos a saída do FFN
(`layer.N/output/dense`), que fica em fp32 pelo mesmo motivo de `pii_br.model.quantizable`.
"""

# torch, transformers, onnx e huggingface_hub não têm tipos completos.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path
from typing import Any

from pii_br import Detector
from pii_br.onnx_model import OnnxModel
from training.evaluate import as_gold, load_split, score_split, summarize

FFN_OUT = re.compile(r"/layer\.\d+/output/dense/MatMul$")
INPUTS = ("input_ids", "attention_mask", "token_type_ids")


def export(model_id: str, revision: str | None, out: Path) -> Path:
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    (out / "tokenizer.json").write_text(tokenizer.backend_tokenizer.to_str(), "utf-8")
    model = AutoModelForTokenClassification.from_pretrained(model_id, revision=revision).eval()
    ids = torch.ones((1, 16), dtype=torch.long)
    path = out / "model.onnx"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model,
            (ids, torch.ones_like(ids), torch.zeros_like(ids)),
            str(path),
            input_names=list(INPUTS),
            output_names=["logits"],
            dynamic_axes={k: {0: "batch", 1: "seq"} for k in (*INPUTS, "logits")},
            opset_version=17,
            dynamo=False,
        )
    return path


def quantize(fp32: Path) -> Path:
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    nodes = [n.name for n in onnx.load(str(fp32)).graph.node]
    keep = [n for n in nodes if FFN_OUT.search(n)]
    if len(keep) != 12:  # BERT base tem 12 camadas
        raise SystemExit(f"esperava 12 nós de saída do FFN, achei {len(keep)}")
    path = fp32.with_name("model_int8.onnx")
    quantize_dynamic(
        fp32,
        path,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gather"],
        nodes_to_exclude=keep,
    )
    return path


def hybrid_f1(model: OnnxModel, rows: list[dict[str, Any]]) -> float:
    detector = Detector(model)
    result = score_split(rows, lambda text: as_gold(detector.detect(text)))
    return float(summarize(result.overlap)["micro"]["f1"])


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--model", required=True, help="repo do Hub ou pasta local")
    parser.add_argument("--revision", help="commit do modelo (padrão: o mais recente)")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("outputs/onnx"))
    parser.add_argument("--max-loss", type=float, default=0.005, help="perda máxima de F1")
    parser.add_argument("--push", action="store_true", help="envia o int8 ao repo do modelo")
    args = parser.parse_args()

    fp32 = export(args.model, args.revision, args.out)
    int8 = quantize(fp32)
    tokenizer = args.out / "tokenizer.json"
    for split in ("test_hard", "test_real"):
        rows = load_split(args.data, split)
        if not rows:
            continue
        ref = hybrid_f1(OnnxModel(fp32, tokenizer), rows)
        got = hybrid_f1(OnnxModel(int8, tokenizer), rows)
        print(f"{split:10} híbrido F1 fp32={ref:.3f} int8={got:.3f}")
        if ref - got > args.max_loss:
            raise SystemExit(f"int8 perdeu {ref - got:.3f} de F1 em {split}; não publico.")
    size = int8.stat().st_size / 1e6
    print(f"{int8} ({size:.0f} MB)")

    if args.push:
        from huggingface_hub import HfApi

        info = HfApi().upload_file(
            path_or_fileobj=str(int8),
            path_in_repo="onnx/model_int8.onnx",
            repo_id=args.model,
            commit_message="ONNX int8 para a demo no navegador",
        )
        print(f"Publicado: {info}")


if __name__ == "__main__":
    main()
