"""Testes do app do Space sem modelo (só regras), incluindo a promessa de não logar texto."""

import csv
import importlib
import logging
import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

gradio = pytest.importorskip("gradio")

ROOT = Path(__file__).parents[1]
STRATEGY = "Redação: [CPF]"
ALL = ["NOME", "CPF", "CNPJ", "RG", "TELEFONE", "EMAIL", "ENDERECO", "CEP", "DATA_NASC"]


@pytest.fixture(scope="module")
def app(monkeypatch_module: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch_module.setenv("PII_BR_MODEL", "")
    monkeypatch_module.syspath_prepend(str(ROOT / "space"))
    sys.modules.pop("app", None)
    return importlib.import_module("app")


@pytest.fixture(scope="module")
def monkeypatch_module() -> Iterator[pytest.MonkeyPatch]:
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_analyze_masks_and_highlights(app: ModuleType, caplog: pytest.LogCaptureFixture) -> None:
    text = "CPF 123.456.789-09, tel (11) 98765-4321, cpf ruim 123.456.789-00"
    with caplog.at_level(logging.DEBUG):
        highlighted, masked, rows, summary, masker = app.analyze(text, STRATEGY, ALL, 0.5, None)
    assert masked == "CPF [CPF], tel [TELEFONE], cpf ruim [CPF]"
    assert [label for _, label in highlighted if label] == ["CPF", "TELEFONE", "ALERTA"]
    assert "".join(seg for seg, _ in highlighted) == text
    assert rows[2]["documento_valido"] is False and "inválido" in summary
    assert masker is not None
    assert "123.456.789" not in caplog.text and "98765" not in caplog.text


def test_pseudonyms_persist_within_session(app: ModuleType) -> None:
    _, first, _, _, masker = app.analyze(
        "CPF 123.456.789-09", "Pseudônimo consistente: <CPF_1>", ALL, 0.5, None
    )
    _, second, _, _, _ = app.analyze(
        "e 52998224725 e 12345678909", "Pseudônimo consistente: <CPF_1>", ALL, 0.5, masker
    )
    assert first == "CPF <CPF_1>" and second == "e <CPF_2> e <CPF_1>"


def test_empty_and_entity_filter(app: ModuleType) -> None:
    assert app.analyze("   ", STRATEGY, ALL, 0.5, None)[1] == ""
    _, masked, _, _, _ = app.analyze(
        "CPF 123.456.789-09 a@example.com", STRATEGY, ["EMAIL"], 0.5, None
    )
    assert masked == "CPF 123.456.789-09 [EMAIL]"


def test_csv_masks_only_chosen_columns_and_deletes_upload(app: ModuleType, tmp_path: Path) -> None:
    upload = tmp_path / "clientes.csv"
    upload.write_text("id,obs,email\n1,CPF 123.456.789-09,a@example.com\n", encoding="utf-8")
    assert app.read_columns(str(upload))["choices"] == ["id", "obs", "email"]
    out, summary, _, cleared = app.process_file(str(upload), ["obs"], STRATEGY, ALL, 0.5, None)
    assert not upload.exists()
    with open(out, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows == [{"id": "1", "obs": "CPF [CPF]", "email": "a@example.com"}]
    assert summary.startswith("1 ") and cleared is None
    assert Path(out).parent == app.OUTPUT_DIR


def test_txt_file(app: ModuleType, tmp_path: Path) -> None:
    upload = tmp_path / "nota.txt"
    upload.write_text("ligue (21) 2345-6789", encoding="utf-8")
    out, _, _, _ = app.process_file(str(upload), [], STRATEGY, ALL, 0.5, None)
    assert Path(out).read_text(encoding="utf-8") == "ligue [TELEFONE]"


def test_examples_cover_spec(app: ModuleType) -> None:
    results = [app.analyze(e, STRATEGY, ALL, 0.5, None) for e in app.EXAMPLES]
    assert len(app.EXAMPLES) == 6
    assert any(r[2] == [] for r in results)  # um exemplo sem PII
    assert any(any(row["documento_valido"] is False for row in r[2]) for r in results)
    assert any("ABC" in e for e in app.EXAMPLES)  # CNPJ alfanumérico


def _csv_rows(path: str, delimiter: str = ",") -> list[list[str]]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.reader(f, delimiter=delimiter))


def test_csv_semicolon_and_ragged_rows(app: ModuleType, tmp_path: Path) -> None:
    upload = tmp_path / "excel.csv"
    upload.write_text("nome;obs\nAna;CPF 123.456.789-09;extra\nBia\n", encoding="utf-8")
    assert app.read_columns(str(upload))["choices"] == ["nome", "obs"]
    out, summary, _, _ = app.process_file(str(upload), ["obs"], STRATEGY, ALL, 0.5, None)
    assert _csv_rows(out, ";") == [["nome", "obs"], ["Ana", "CPF [CPF]", "extra"], ["Bia"]]
    assert summary.startswith("1 ")


def test_second_click_after_upload_deleted(app: ModuleType, tmp_path: Path) -> None:
    upload = tmp_path / "nota.txt"
    upload.write_text("x", encoding="utf-8")
    app.process_file(str(upload), [], STRATEGY, ALL, 0.5, None)
    out, summary, _, _ = app.process_file(str(upload), [], STRATEGY, ALL, 0.5, None)
    assert out is None and "envie-o de novo" in summary
    assert app.read_columns(str(tmp_path / "sumiu.csv"))["visible"] is False


def test_text_cut_is_reported(app: ModuleType) -> None:
    text = "x" * app.MAX_CHARS + " cpf 123.456.789-09"
    _, masked, _, summary, _ = app.analyze(text, STRATEGY, ALL, 0.5, None)
    assert len(masked) == app.MAX_CHARS and "Texto cortado" in summary
    assert "20.000" in summary


def test_file_limits_cut_and_report(
    app: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app, "MAX_CELLS", 2)
    upload = tmp_path / "grande.csv"
    upload.write_text("obs\na\nb\nc\nd\n", encoding="utf-8")
    out, summary, _, _ = app.process_file(str(upload), ["obs"], STRATEGY, ALL, 0.5, None)
    assert _csv_rows(out) == [["obs"], ["a"], ["b"]]
    assert "linha 4" in summary

    monkeypatch.setattr(app, "MAX_FILE_CHARS", 10)
    upload = tmp_path / "grande.txt"
    upload.write_text("tel (21) 2345-6789", encoding="utf-8")
    out, summary, _, _ = app.process_file(str(upload), [], STRATEGY, ALL, 0.5, None)
    assert Path(out).read_text(encoding="utf-8") == "tel (21) 2" and "cortado" in summary


def test_broken_csv_and_bad_encoding(app: ModuleType, tmp_path: Path) -> None:
    upload = tmp_path / "quebrado.csv"
    upload.write_text('obs\n"' + "x" * 50 + "\n", encoding="utf-8")
    limit = csv.field_size_limit(10)
    try:
        _, summary, _, _ = app.process_file(str(upload), ["obs"], STRATEGY, ALL, 0.5, None)
    finally:
        csv.field_size_limit(limit)
    assert summary.startswith("Não consegui ler o CSV")
    upload = tmp_path / "latin1.txt"
    upload.write_bytes("ação".encode("latin-1"))
    _, summary, _, _ = app.process_file(str(upload), [], STRATEGY, ALL, 0.5, None)
    assert "UTF-8" in summary


def test_old_outputs_are_purged(app: ModuleType) -> None:
    app.OUTPUT_DIR.mkdir(exist_ok=True)
    old = app.OUTPUT_DIR / "pii-br-mascarado-velho.txt"
    old.write_text("x", encoding="utf-8")
    past = time.time() - app.OUTPUT_TTL_SECONDS - 1
    os.utime(old, (past, past))
    new = app._write_output("y", ".txt")  # pyright: ignore[reportPrivateUsage]
    assert not old.exists() and Path(new).exists()
