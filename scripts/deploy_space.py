"""Monta a pasta do Space (app + cópia do pacote pii_br) e publica no Hugging Face.

Uso:
    uv run --group hub python scripts/deploy_space.py --repo brennocm/PII-BR --dry-run
    HF_TOKEN=... uv run --group hub python scripts/deploy_space.py --repo brennocm/PII-BR

`--dry-run` só monta a pasta e imprime o caminho (dá para rodar `python app.py` nela).
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).parents[1]
SPACE_FILES = ("app.py", "requirements.txt", "README.md")


def build(target: Path) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    for name in SPACE_FILES:
        shutil.copy2(ROOT / "space" / name, target / name)
    shutil.copytree(
        ROOT / "src" / "pii_br",
        target / "pii_br",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        dirs_exist_ok=True,
    )
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--repo", required=True, help="brennocm/PII-BR")
    parser.add_argument("--model", help="repo do modelo, se diferente de brennocm/PII-BR")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    folder = build(Path(tempfile.mkdtemp(prefix="pii-br-space-")))
    if args.dry_run:
        print(folder)
        return

    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Defina HF_TOKEN (token com escrita no Space).")
    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type="space", space_sdk="gradio", exist_ok=True)
    if args.model:
        api.add_space_variable(args.repo, "PII_BR_MODEL", args.model)
    sha = os.environ.get("GITHUB_SHA", "local")[:7]
    api.upload_folder(
        folder_path=str(folder),
        repo_id=args.repo,
        repo_type="space",
        commit_message=f"Deploy {sha}",
        delete_patterns=["pii_br/**"],  # remove módulos que deixaram de existir
    )
    print(f"Space atualizado: https://huggingface.co/spaces/{args.repo}")


if __name__ == "__main__":
    main()
