"""Monta o DatasetDict a partir de `data/` e publica no Hugging Face Hub com o card.

Uso:
    uv run --group hub python -m data_gen.push_dataset --repo <usuario>/PII-BR --dry-run
    HF_TOKEN=... uv run --group hub python -m data_gen.push_dataset --repo <usuario>/PII-BR

`--dry-run` monta tudo e grava o card em `data/README.md`, sem publicar. O token vem da
variável HF_TOKEN ou do login do `hf auth login`; nunca é passado na linha de comando.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value

from pii_br.types import BIO_LABELS, Entity

CARD_TEMPLATE = Path(__file__).parents[1] / "cards" / "dataset.md"
SPLIT_NOTES = {
    "train": "80% dos templates",
    "validation": "mesmos templates do train, valores novos",
    "test": "mesmos templates do train, valores novos",
    "test_hard": "20% dos templates, nunca vistos no treino",
    "test_real": "frases escritas à mão, fora dos templates",
}

FEATURES = Features(
    {
        "id": Value("string"),
        "text": Value("string"),
        "tokens": Sequence(Value("string")),
        "ner_tags": Sequence(ClassLabel(names=list(BIO_LABELS))),
        "spans": [{"start": Value("int32"), "end": Value("int32"), "label": Value("string")}],
        "domain": Value("string"),
        "template_id": Value("string"),
        "noisy": Value("bool"),
        "has_invalid_dv": Value("bool"),
    }
)


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build(data_dir: Path) -> DatasetDict:
    dataset = DatasetDict()
    for path in sorted(data_dir.glob("*.jsonl")):
        dataset[path.stem] = Dataset.from_list(load_rows(path), features=FEATURES)
    return dataset


def render_card(
    stats: dict[str, Any], repo: str, model_repo: str, space_repo: str, github: str
) -> str:
    splits: dict[str, Any] = stats["splits"]
    order = [s for s in SPLIT_NOTES if s in splits]
    split_table = "\n".join(
        f"| `{s}` | {splits[s]['sentences']:_} | {SPLIT_NOTES[s]} |".replace("_", ".")
        for s in order
    )
    entities = [e.value for e in Entity]
    header = "| Split | " + " | ".join(f"`{e}`" for e in entities) + " | sem PII |"
    sep = "| --- " * (len(entities) + 2) + "|"
    rows = [
        f"| `{s}` | "
        + " | ".join(str(splits[s]["entities"].get(e, 0)) for e in entities)
        + f" | {splits[s]['no_pii']} |"
        for s in order
    ]
    card = CARD_TEMPLATE.read_text(encoding="utf-8")
    replacements = {
        "{{SPLIT_TABLE}}": split_table,
        "{{ENTITY_TABLE}}": "\n".join([header, sep, *rows]),
        "{{LABELS}}": ", ".join(f"`{label}`" for label in stats["labels"]),
        "{{TEMPLATES_TOTAL}}": str(stats["templates_total"]),
        "{{FAKER_VERSION}}": f"versão {stats['faker_version']}",
        "{{SEED}}": str(stats["seed"]),
        "{{DATASET_REPO}}": repo,
        "{{MODEL_REPO}}": model_repo,
        "{{SPACE_REPO}}": space_repo,
        "{{GITHUB_URL}}": github,
    }
    for key, value in replacements.items():
        card = card.replace(key, value)
    return card


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--repo", required=True, help="<usuario>/PII-BR")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--github", default="https://github.com/<usuario>/PII-BR")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    stats = json.loads((args.data / "stats.json").read_text(encoding="utf-8"))
    if "test_real" not in stats["splits"]:
        raise SystemExit("test_real ausente: rode o gerador com data_gen/test_real.tsv presente.")
    dataset = build(args.data)
    card = render_card(stats, args.repo, args.repo, args.repo, args.github)
    (args.data / "README.md").write_text(card, encoding="utf-8")
    print(dataset)
    if args.dry_run:
        print(f"dry-run: card gravado em {args.data / 'README.md'}; nada publicado.")
        return

    from huggingface_hub import HfApi  # import tardio: só quem publica precisa

    dataset.push_to_hub(args.repo, private=False, commit_message="Publica dataset PII-BR")
    HfApi().upload_file(
        path_or_fileobj=str(args.data / "README.md"),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="dataset",
        commit_message="Atualiza dataset card",
    )
    print(f"Publicado: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
