import hashlib
import json
import os
import random
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from data_gen import generate as g
from data_gen.fillers import Filler
from pii_br import recognize
from pii_br import validators as v
from pii_br.types import BIO_LABELS
from pii_br.words import word_spans

TEMPLATES = g.load_templates()


def test_templates_meet_spec() -> None:
    domains = Counter(t.domain for t in TEMPLATES)
    assert len(TEMPLATES) >= 400
    assert len(domains) == 8 and min(domains.values()) >= 40
    assert all(any(not t.has_pii for t in TEMPLATES if t.domain == d) for d in domains)
    assert len({t.text for t in TEMPLATES}) == len(TEMPLATES)  # sem duplicatas


def test_split_has_no_template_leakage() -> None:
    seen, hard = g.split_templates(TEMPLATES, seed=42)
    assert not set(seen) & set(hard)
    assert set(seen) | set(hard) == {t.id for t in TEMPLATES}
    assert 0.17 <= len(hard) / len(TEMPLATES) <= 0.23
    assert g.split_templates(TEMPLATES, seed=42) == (seen, hard)


@pytest.fixture(scope="module")
def rows() -> list[dict[str, Any]]:
    return g.generate_split("train", TEMPLATES, 1500, seed=7)


def test_generation_is_reproducible(rows: list[dict[str, Any]]) -> None:
    again = g.generate_split("train", TEMPLATES, 1500, seed=7)
    digest = [hashlib.sha256(json.dumps(r).encode()).hexdigest() for r in (rows, again)]
    assert digest[0] == digest[1]


def test_generation_is_reproducible_across_processes(tmp_path: Path) -> None:
    """O hash de strings muda entre processos; a saída não pode mudar junto."""
    digests = []
    for hash_seed in ("1", "2"):
        out = tmp_path / hash_seed
        subprocess.run(
            [sys.executable, "-m", "data_gen.generate", "--out", str(out), "--small"],
            check=True,
            capture_output=True,
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
        )
        digests.append(hashlib.sha256((out / "test_hard.jsonl").read_bytes()).hexdigest())
    assert digests[0] == digests[1]


def test_tags_match_spans_and_tokens(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        text: str = row["text"]
        tokens: list[str] = row["tokens"]
        tags: list[int] = row["ner_tags"]
        spans: list[dict[str, Any]] = row["spans"]
        assert len(tokens) == len(tags)
        assert "".join(tokens) == re.sub(r"\s", "", text)
        begins = [BIO_LABELS[t][2:] for t in tags if BIO_LABELS[t].startswith("B-")]
        assert begins == [s["label"] for s in spans]
        for s in spans:
            value = text[s["start"] : s["end"]]
            assert value and value == value.strip()


def test_proportions(rows: list[dict[str, Any]]) -> None:
    n = len(rows)
    no_pii = sum(not r["spans"] for r in rows) / n
    noisy = sum(bool(r["noisy"]) for r in rows) / n
    assert 0.11 <= no_pii <= 0.19
    assert 0.25 <= noisy <= 0.35
    docs = [
        (s["label"], str(r["text"])[s["start"] : s["end"]])
        for r in rows
        for s in r["spans"]
        if s["label"] in ("CPF", "CNPJ")
    ]
    invalid = sum(
        not (v.is_valid_cpf(t) if label == "CPF" else v.is_valid_cnpj(t)) for label, t in docs
    )
    assert 0.05 <= invalid / len(docs) <= 0.15
    alnum = [t for label, t in docs if label == "CNPJ" and any(c.isalpha() for c in t)]
    assert alnum  # o formato novo aparece no dataset


def test_hard_negatives_are_not_documents() -> None:
    filler = Filler(random.Random(3))
    for _ in range(500):
        digits = v.only_digits(filler.pedido())
        if len(digits) == 11:
            assert not v.is_valid_cpf(digits)
        if len(digits) == 14:
            assert not v.is_valid_cnpj(digits)
        assert not any(s.label.value == "TELEFONE" for s in recognize(filler.tel_empresa()))


def test_parse_markup() -> None:
    segments = g.parse_markup("Sou [Ana](NOME), CPF [123.456.789-09](CPF).\\nTchau")
    row = g.to_example("x", segments, "teste", "real", False)
    assert row["text"] == "Sou Ana, CPF 123.456.789-09.\nTchau"
    assert row["spans"] == [
        {"start": 4, "end": 7, "label": "NOME"},
        {"start": 13, "end": 27, "label": "CPF"},
    ]
    with pytest.raises(ValueError):
        g.parse_markup("[x](NAO_EXISTE)")


def test_test_real_file_parses() -> None:
    real = g.load_test_real()
    assert len(real) >= 200
    for row in real:
        text: str = row["text"]
        starts = {a for a, _ in word_spans(text)}
        for s in row["spans"]:
            assert s["start"] in starts or text[s["start"] - 1].isalnum() is False
