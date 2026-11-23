"""Demo estática: montagem da pasta do Space e a ponte Python que roda no Pyodide."""

import importlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

from pii_br.types import BIO_LABELS
from scripts.deploy_space import build

ROOT = Path(__file__).parents[1]
ALL = ["NOME", "CPF", "CNPJ", "RG", "TELEFONE", "EMAIL", "ENDERECO", "CEP", "DATA_NASC"]


@pytest.fixture
def bridge(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(ROOT / "space"))
    sys.modules.pop("bridge", None)
    return importlib.import_module("bridge")


def request(text: str, logits: object = None, **extra: object) -> str:
    body = {"text": text, "logits": logits, "strategy": "redact", "entities": ALL, "threshold": 0.5}
    return json.dumps(body | extra)


def test_build_writes_page_config_and_package(tmp_path: Path) -> None:
    out = build(tmp_path / "site", "dono/modelo", "abc123")
    names = {p.name for p in out.iterdir()}
    assert names == {"index.html", "style.css", "app.js", "README.md", "config.json", "pii_br.zip"}
    assert json.loads((out / "config.json").read_text()) == {
        "model": "dono/modelo",
        "revision": "abc123",
    }
    with zipfile.ZipFile(out / "pii_br.zip") as zf:
        files = set(zf.namelist())
    assert {"bridge.py", "pii_br/__init__.py", "pii_br/ner.py", "pii_br/detector.py"} <= files
    assert not any("__pycache__" in f for f in files)


def test_page_assets_are_pinned_with_integrity() -> None:
    html = (ROOT / "space" / "index.html").read_text(encoding="utf-8")
    scripts = [line for line in html.splitlines() if '<script src="https://' in line]
    assert len(scripts) == 2
    assert html.count('integrity="sha384-') == 2
    assert "@latest" not in html


def test_package_runs_with_stdlib_only(tmp_path: Path) -> None:
    """O zip roda num Python sem site-packages, como no Pyodide (sem torch nem numpy)."""
    out = build(tmp_path / "site", "dono/modelo", "main")
    code = (
        "import sys, json; sys.path.insert(0, sys.argv[1]); import bridge; "
        "print(bridge.analyze(json.dumps({'text': 'CPF 123.456.789-09', 'logits': None, "
        "'strategy': 'redact', 'entities': ['CPF'], 'threshold': 0.5})))"
    )
    done = subprocess.run(
        [sys.executable, "-I", "-S", "-c", code, str(out / "pii_br.zip")],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(done.stdout)["masked"] == "CPF [CPF]"


def test_rules_only_masks_and_flags_invalid(bridge: ModuleType) -> None:
    text = "CPF 123.456.789-09 e cpf ruim 123.456.789-00"
    result = json.loads(bridge.analyze(request(text)))
    assert result["masked"] == "CPF [CPF] e cpf ruim [CPF]"
    assert [s["valid"] for s in result["spans"]] == [True, False]
    assert result["cut"] is False


def test_long_text_is_cut(bridge: ModuleType) -> None:
    result = json.loads(bridge.analyze(request("a" * (bridge.MAX_CHARS + 10))))
    assert result["cut"] is True
    assert result["chars"] == bridge.MAX_CHARS


def test_unknown_strategy_is_rejected(bridge: ModuleType) -> None:
    with pytest.raises(ValueError, match="estratégia"):
        bridge.analyze(request("oi", strategy="apagar"))


def tokenizer_json() -> str:
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "Sou", "Ana", "Lima", ","]
    return json.dumps({"model": {"vocab": {w: i for i, w in enumerate(vocab)}}})


def test_model_logits_become_spans(bridge: ModuleType) -> None:
    bridge.load_tokenizer(tokenizer_json())
    text = "Sou Ana Lima, CPF 123.456.789-09"
    windows = json.loads(bridge.encode(text))
    assert len(windows) == 1
    labels = ["O"] * len(windows[0])
    labels[2], labels[3] = "B-NOME", "I-NOME"  # [CLS] Sou Ana Lima ...
    flat = [10.0 if BIO_LABELS[k] == lab else 0.0 for lab in labels for k in range(19)]
    result = json.loads(bridge.analyze(request(text, [flat])))
    assert result["masked"] == "Sou [NOME], CPF [CPF]"
    assert {(s["label"], s["source"]) for s in result["spans"]} == {
        ("NOME", "modelo"),
        ("CPF", "regex"),
    }


def test_logits_for_another_text_are_rejected(bridge: ModuleType) -> None:
    bridge.load_tokenizer(tokenizer_json())
    bridge.encode("Sou Ana")
    with pytest.raises(RuntimeError, match="outro texto"):
        bridge.analyze(request("Sou Lima", [[0.0] * 19 * 4]))


def test_encode_needs_tokenizer(bridge: ModuleType) -> None:
    bridge._session.tokenizer = None  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(RuntimeError, match="tokenizer"):
        bridge.encode("oi")


def test_pseudonyms_are_consistent_within_a_tab(bridge: ModuleType) -> None:
    text = "CPF 123.456.789-09"
    first = json.loads(bridge.analyze(request(text, strategy="pseudonym")))["masked"]
    second = json.loads(bridge.analyze(request(text, strategy="pseudonym")))["masked"]
    assert first == second
