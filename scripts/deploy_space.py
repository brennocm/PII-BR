"""Monta a demo estática (página + pacote pii_br em zip) e publica no Hugging Face Spaces.

Uso:
    uv run --group hub python scripts/deploy_space.py --repo brennocm/PII-BR --dry-run
    uv run --group hub python scripts/deploy_space.py --repo brennocm/PII-BR --out outputs/space
    HF_TOKEN=... uv run --group hub python scripts/deploy_space.py --repo brennocm/PII-BR

O Space é do tipo `static` (grátis): o navegador roda o pacote no Pyodide e o modelo ONNX
no `onnxruntime-web`. `--dry-run` só monta a pasta e imprime o caminho; para ver a página,
sirva a pasta com `python -m http.server -d <pasta>` (ou `mise run web`).

O modelo vem de `--model` no commit `MODEL_REVISION` (ou `--model-revision` /
`PII_BR_MODEL_REVISION`): fixar um commit impede que um push no repositório do modelo
mude o que a demo roda.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).parents[1]
SPACE_FILES = ("index.html", "style.css", "app.js", "README.md")
# Arquivos do Space Gradio antigo que o deploy estático remove do repositório do Space.
STALE = ["app.py", "requirements.txt", "pii_br/**"]
# Commit do repositório do modelo com `onnx/model_int8.onnx`; troque ao publicar um modelo novo.
MODEL_REVISION = "65df4ddcfb770f61551e16272d522c66aa4188d9"


def build(target: Path, model: str, revision: str) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    for name in SPACE_FILES:
        shutil.copy2(ROOT / "space" / name, target / name)
    config = {"model": model, "revision": revision}
    (target / "config.json").write_text(json.dumps(config, indent=2) + "\n", "utf-8")
    package = ROOT / "src" / "pii_br"
    with zipfile.ZipFile(target / "pii_br.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(package.rglob("*.py")):
            zf.write(path, Path("pii_br") / path.relative_to(package))
        zf.write(ROOT / "space" / "bridge.py", "bridge.py")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--repo", required=True, help="Space, ex.: brennocm/PII-BR")
    parser.add_argument("--model", default="brennocm/PII-BR", help="repositório do modelo")
    parser.add_argument(
        "--model-revision",
        default=os.environ.get("PII_BR_MODEL_REVISION") or MODEL_REVISION,
        help="commit do modelo (padrão: PII_BR_MODEL_REVISION ou MODEL_REVISION)",
    )
    parser.add_argument("--out", type=Path, help="pasta de saída (padrão: temporária)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out = args.out or Path(tempfile.mkdtemp(prefix="pii-br-space-"))
    folder = build(out, args.model, args.model_revision)
    if args.dry_run:
        print(folder)
        return

    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Defina HF_TOKEN (token com escrita no Space).")
    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type="space", space_sdk="static", exist_ok=True)
    sha = os.environ.get("GITHUB_SHA", "local")[:7]
    api.upload_folder(
        folder_path=str(folder),
        repo_id=args.repo,
        repo_type="space",
        commit_message=f"Deploy {sha}",
        delete_patterns=STALE,
    )
    print(f"Space atualizado: https://huggingface.co/spaces/{args.repo}")


if __name__ == "__main__":
    main()
