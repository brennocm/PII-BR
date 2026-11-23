"""Gera o dataset sintético: templates + Faker pt_BR → texto, tokens, tags BIO e spans.

Uso:
    uv run python -m data_gen.generate --out data            # tamanho da spec
    uv run python -m data_gen.generate --out data --small    # 1/10, para testes rápidos

Saída: um JSONL por split (train, validation, test, test_hard e, se existir o arquivo
revisado, test_real), `stats.json` e `split_templates.json`.

A anotação é feita por construção: o texto é montado em segmentos (literal ou entidade), e
cada segmento é tokenizado separadamente, então todo limite de entidade cai num limite
de token. O ruído é aplicado aos segmentos antes de montar o texto, então os offsets
nunca precisam ser recalculados.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import string
from collections import Counter
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import yaml

from data_gen.fillers import EMOJIS, ENTITY_SLOTS, PLAIN_SLOTS, Filler, strip_accents
from pii_br import validators as v
from pii_br.types import BIO_LABELS, Entity
from pii_br.words import word_spans

TEMPLATES_DIR = Path(__file__).parent / "templates"
TEST_REAL_FILE = Path(__file__).parent / "test_real.tsv"
SLOT = re.compile(r"\{([A-Z_]+?)(\d*)\}")
LABEL_ID = {label: i for i, label in enumerate(BIO_LABELS)}

SIZES = {"train": 30_000, "validation": 3_000, "test": 3_000, "test_hard": 2_000}
NO_PII_RATE = 0.15
NOISE_RATE = 0.30
HARD_FRACTION = 0.20


@dataclass(frozen=True)
class Template:
    id: str
    domain: str
    text: str

    @property
    def slots(self) -> list[str]:
        return [m[1] for m in SLOT.finditer(self.text)]

    @property
    def has_pii(self) -> bool:
        return any(s in ENTITY_SLOTS for s in self.slots)


Segment = tuple[str, str | None]  # (texto, entidade ou None)


def load_templates(directory: Path = TEMPLATES_DIR) -> list[Template]:
    templates: list[Template] = []
    for path in sorted(directory.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        domain = str(data["domain"])
        for i, text in enumerate(data["templates"]):
            unknown = {m[1] for m in SLOT.finditer(text)} - (
                ENTITY_SLOTS.keys() | PLAIN_SLOTS.keys()
            )
            if unknown:
                raise ValueError(f"{path.name}#{i}: slots desconhecidos {sorted(unknown)}")
            templates.append(Template(f"{domain}-{i:03d}", domain, str(text)))
    return templates


def split_templates(templates: list[Template], seed: int) -> tuple[list[str], list[str]]:
    """80% dos templates para treino/val/test e 20% para test_hard, estratificado por
    domínio e por ter ou não PII (para o test_hard também ter frases sem PII)."""
    rng = random.Random(seed)
    seen: list[str] = []
    hard: list[str] = []
    groups: dict[tuple[str, bool], list[str]] = {}
    for t in templates:
        groups.setdefault((t.domain, t.has_pii), []).append(t.id)
    for key in sorted(groups):
        ids = sorted(groups[key])
        rng.shuffle(ids)
        n_hard = max(1, round(len(ids) * HARD_FRACTION))
        hard += ids[:n_hard]
        seen += ids[n_hard:]
    return sorted(seen), sorted(hard)


# ----------------------------------------------------------------- renderização


def render(template: Template, filler: Filler, *, noisy: bool) -> list[Segment]:
    """Preenche o template e devolve os segmentos (literal ou entidade), já com ruído."""
    pieces: list[tuple[str, str | None, str | None]] = []  # (literal, slot, chave)
    last = 0
    for m in SLOT.finditer(template.text):
        if m.start() > last:
            pieces.append((template.text[last : m.start()], None, None))
        pieces.append(("", m[1], m[0]))
        last = m.end()
    if last < len(template.text):
        pieces.append((template.text[last:], None, None))

    # NOME primeiro, para o e-mail poder derivar do nome da mesma frase.
    values: dict[str, str] = {}
    # Ordem de aparição (determinística; um set mudaria a cada processo por causa do hash).
    keys = list(dict.fromkeys(k for _, _, k in pieces if k))
    keys.sort(key=lambda k: not k.startswith("{NOME"))
    for key in keys:
        slot = key.strip("{}").rstrip("0123456789")
        if slot == "EMAIL":
            name_hint = values.get("{NOME}")
            values[key] = filler.email(name_hint)
        elif slot in ENTITY_SLOTS:
            values[key] = ENTITY_SLOTS[slot](filler)
        else:
            values[key] = PLAIN_SLOTS[slot](filler)

    segments: list[Segment] = []
    for literal, slot, key in pieces:
        if slot is None or key is None:
            segments.append((literal, None))
        else:
            segments.append((values[key], slot if slot in ENTITY_SLOTS else None))
    if noisy:
        segments = add_noise(segments, filler)
    return segments


# ------------------------------------------------------------------------ ruído

_FREE_TEXT = {"NOME", "ENDERECO"}


def add_noise(segments: list[Segment], filler: Filler) -> list[Segment]:
    lower = filler.chance(0.4)
    no_accents = filler.chance(0.4)
    typos = filler.chance(0.35)
    emoji = filler.chance(0.25)
    newlines = filler.chance(0.15)
    shout = not lower and filler.chance(0.05)
    out: list[Segment] = []
    for text, label in segments:
        if label is None:
            if typos:
                text = _typos(text, filler)
            if newlines:
                text = re.sub(r"([.,!?]) ", lambda m: m[1] + "\n", text)
        if label is None or label in _FREE_TEXT or label == "EMAIL":
            if lower:
                text = text.lower()
            if shout and label != "EMAIL":
                text = text.upper()
        if no_accents and (label is None or label in _FREE_TEXT):
            text = strip_accents(text)
        out.append((text, label))
    if emoji and out:
        last_text, last_label = out[-1]
        if last_label is None:
            out[-1] = (last_text.rstrip(".") + " " + filler.pick(EMOJIS), None)
        else:
            out.append((" " + filler.pick(EMOJIS), None))
    return out


def _typos(text: str, filler: Filler) -> str:
    def corrupt(m: re.Match[str]) -> str:
        word = m[0]
        if len(word) < 4 or not filler.chance(0.08):
            return word
        i = filler.rng.randrange(1, len(word) - 1)
        kind = filler.rng.random()
        if kind < 0.35:  # troca vizinhos
            return word[: i - 1] + word[i] + word[i - 1] + word[i + 1 :]
        if kind < 0.65:  # apaga
            return word[:i] + word[i + 1 :]
        if kind < 0.85:  # duplica
            return word[:i] + word[i] + word[i:]
        return word[:i] + filler.pick(string.ascii_lowercase) + word[i + 1 :]  # troca letra

    return re.sub(r"[^\W\d_]+", corrupt, text)


# --------------------------------------------------------------------- exemplo


def to_example(
    idx: str, segments: list[Segment], domain: str, template_id: str, noisy: bool
) -> dict[str, Any]:
    """Monta texto, tokens, tags BIO e spans. Cada segmento é tokenizado à parte, então
    nenhum token atravessa o limite de uma entidade."""
    text = ""
    tokens: list[str] = []
    tags: list[int] = []
    spans: list[dict[str, Any]] = []
    for value, label in segments:
        offset = len(text)
        if label is not None and value:
            spans.append({"start": offset, "end": offset + len(value), "label": label})
        for i, (start, end) in enumerate(word_spans(value)):
            tokens.append(value[start:end])
            tag = "O" if label is None else f"{'B' if i == 0 else 'I'}-{label}"
            tags.append(LABEL_ID[tag])
        text += value
    invalid = any(
        (s["label"] == "CPF" and not v.is_valid_cpf(text[s["start"] : s["end"]]))
        or (s["label"] == "CNPJ" and not v.is_valid_cnpj(text[s["start"] : s["end"]]))
        for s in spans
    )
    return {
        "id": idx,
        "text": text,
        "tokens": tokens,
        "ner_tags": tags,
        "spans": spans,
        "domain": domain,
        "template_id": template_id,
        "noisy": noisy,
        "has_invalid_dv": invalid,
    }


def generate_split(name: str, templates: list[Template], n: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(f"{seed}-{name}")
    filler = Filler(rng)
    with_pii = [t for t in templates if t.has_pii]
    without_pii = [t for t in templates if not t.has_pii]
    rows: list[dict[str, Any]] = []
    for i in range(n):
        pool = without_pii if without_pii and rng.random() < NO_PII_RATE else with_pii
        template = rng.choice(pool)
        noisy = rng.random() < NOISE_RATE
        segments = render(template, filler, noisy=noisy)
        rows.append(to_example(f"{name}-{i:05d}", segments, template.domain, template.id, noisy))
    return rows


# -------------------------------------------------------------------- test_real

_MARKUP = re.compile(r"\[([^\]]+)\]\(([A-Z_]+)\)")


def parse_markup(line: str) -> list[Segment]:
    """`Sou [Ana Souza](NOME), CPF [123.456.789-09](CPF).` → segmentos. `\\n` vira quebra."""
    segments: list[Segment] = []
    last = 0
    for m in _MARKUP.finditer(line):
        segments.append((line[last : m.start()].replace("\\n", "\n"), None))
        if m[2] not in Entity.__members__:
            raise ValueError(f"rótulo desconhecido {m[2]!r} em: {line}")
        segments.append((m[1], m[2]))
        last = m.end()
    segments.append((line[last:].replace("\\n", "\n"), None))
    return [seg for seg in segments if seg[0]]


def load_test_real(path: Path = TEST_REAL_FILE) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not raw.strip() or raw.startswith("#"):
            continue
        domain, line = raw.split("\t", 1)
        rows.append(to_example(f"test_real-{n:04d}", parse_markup(line), domain, "real", False))
    return rows


# ------------------------------------------------------------------------ main


def stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(s["label"] for r in rows for s in r["spans"])
    return {
        "sentences": len(rows),
        "entities": dict(sorted(labels.items())),
        "no_pii": sum(not r["spans"] for r in rows),
        "noisy": sum(r["noisy"] for r in rows),
        "invalid_dv": sum(r["has_invalid_dv"] for r in rows),
        "domains": dict(sorted(Counter(r["domain"] for r in rows).items())),
        "templates": len({r["template_id"] for r in rows}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--small", action="store_true", help="1/10 do tamanho, para testes")
    args = parser.parse_args()

    templates = load_templates()
    by_id = {t.id: t for t in templates}
    seen_ids, hard_ids = split_templates(templates, args.seed)
    seen = [by_id[i] for i in seen_ids]
    hard = [by_id[i] for i in hard_ids]

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "split_templates.json").write_text(
        json.dumps({"seen": seen_ids, "hard": hard_ids}, indent=1), encoding="utf-8"
    )
    summary: dict[str, Any] = {
        "seed": args.seed,
        "faker_version": version("faker"),
        "templates_total": len(templates),
        "templates_seen": len(seen),
        "templates_hard": len(hard),
        "labels": list(BIO_LABELS),
        "splits": {},
    }
    splits = {
        name: generate_split(
            name,
            hard if name == "test_hard" else seen,
            size // 10 if args.small else size,
            args.seed,
        )
        for name, size in SIZES.items()
    }
    if TEST_REAL_FILE.exists():
        splits["test_real"] = load_test_real()
    for name, rows in splits.items():
        with (args.out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary["splits"][name] = stats(rows)
        print(f"{name}: {len(rows)} frases")
    (args.out / "stats.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
