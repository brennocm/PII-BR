import ast
import json
from pathlib import Path

from pii_br.types import BIO_LABELS
from training.evaluate import Counts, match, score_split

ROOT = Path(__file__).parents[1]


def test_notebook_labels_match_package() -> None:
    nb = json.loads((ROOT / "training" / "train.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    labels = None
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "LABELS" for t in node.targets
        ):
            labels = ast.literal_eval(node.value)
    assert labels == list(BIO_LABELS)


def test_notebook_has_no_outputs_or_secrets() -> None:
    raw = (ROOT / "training" / "train.ipynb").read_text(encoding="utf-8")
    nb = json.loads(raw)
    assert all(not c.get("outputs") for c in nb["cells"] if c["cell_type"] == "code")
    assert "hf_" not in raw.replace("hf_token", "").replace("HF_", "").lower().replace(
        "get_hf_", ""
    )


def test_match_exact_and_overlap() -> None:
    gold = [(0, 5, "NOME"), (10, 24, "CPF")]
    pred = [(0, 5, "NOME"), (9, 24, "CPF"), (30, 35, "EMAIL")]
    assert match(gold, pred, exact=True) == ({0}, {0})
    assert match(gold, pred, exact=False) == ({0, 1}, {0, 1})


def test_score_split_counts() -> None:
    rows = [
        {"id": "a", "text": "Ana CPF 123.456.789-09", "spans": [
            {"start": 0, "end": 3, "label": "NOME"},
            {"start": 8, "end": 22, "label": "CPF"},
        ]},
    ]  # fmt: skip
    result = score_split(rows, lambda text: [(8, 22, "CPF"), (0, 3, "EMAIL")])
    assert result.overlap["CPF"].scores()["f1"] == 1.0
    assert result.overlap["NOME"].fn == 1 and result.overlap["EMAIL"].fp == 1
    assert result.errors[0]["missed"] == [("Ana", "NOME")]
    assert Counts().scores()["f1"] == 0.0
