"""Preenche cards/model.md com os resultados da avaliação → outputs/model_card/README.md."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[1]
TEMPLATE = ROOT / "cards" / "model.md"
OUT = ROOT / "outputs" / "model_card" / "README.md"


def _seqeval_section(model: str) -> str:
    """Tabela do F1 estrito (seqeval) do notebook, se o arquivo estiver acessível."""
    path = Path(model) / "metrics_seqeval.json"
    if not path.exists():
        try:
            from huggingface_hub import hf_hub_download

            path = Path(hf_hub_download(model, "metrics_seqeval.json"))
        except Exception:  # sem rede, repo privado ou arquivo ausente: o card sai sem a tabela
            return ""
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    lines = [
        "**Só o modelo, F1 estrito por token (seqeval, IOB2):**",
        "",
        "| Split | Precisão | Recall | F1 |",
        "| --- | --- | --- | --- |",
    ]
    for split, report in data["splits"].items():
        micro = report["micro avg"]
        lines.append(
            f"| `{split}` | {micro['precision']:.3f} | {micro['recall']:.3f} "
            f"| {micro['f1-score']:.3f} |".replace(".", ",")
        )
    return "\n".join(lines)


def write_card(results: dict[str, Any], report: str, errors: str, errors_key: str) -> Path:
    model = str(results["model"])
    user = os.environ.get("HF_USER") or (model.split("/")[0] if "/" in model else "<usuario>")
    repo = f"{user}/PII-BR"
    lat = results.get("latency", {})
    values = {
        "{{MODEL_REPO}}": model if "/" in model and not Path(model).exists() else repo,
        "{{DATASET_REPO}}": repo,
        "{{SPACE_REPO}}": repo,
        "{{GITHUB_URL}}": os.environ.get("GITHUB_URL", f"https://github.com/{user}/PII-BR"),
        "{{THRESHOLD}}": str(results["threshold"]).replace(".", ","),
        "{{RESULTS}}": report,
        "{{SEQEVAL}}": _seqeval_section(model),
        "{{LATENCY}}": f"{lat.get('median_s', float('nan')):.2f}".replace(".", ","),
        "{{LATENCY_CHARS}}": str(lat.get("chars", "?")),
        "{{ERRORS}}": errors,
        "{{ERRORS_SPLIT}}": errors_key.split("/")[-1],
    }
    card = TEMPLATE.read_text(encoding="utf-8")
    for key, value in values.items():
        card = card.replace(key, value)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(card, encoding="utf-8")
    print(f"model card: {OUT}")
    return OUT
