"""As versões fixadas fora do uv (Space e notebook do Kaggle) têm que bater com o uv.lock,
que é o que o CI testa. Sem isso, o Space ou o treino rodariam versões nunca testadas."""

import json
import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
PIN = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s;#]+)")


def locked() -> dict[str, str]:
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    return {p["name"]: p["version"] for p in lock["package"]}


def pins(lines: list[str]) -> dict[str, str]:
    found = (PIN.match(line.strip().strip('",')) for line in lines)
    return {m[1].lower(): m[2] for m in found if m}


def test_space_requirements_match_lock() -> None:
    lines = (ROOT / "app" / "requirements.txt").read_text(encoding="utf-8").splitlines()
    space = pins(lines)
    assert set(space) == {"torch", "transformers", "gradio"}
    lock = locked()
    for name, version in space.items():
        assert lock[name] == version, name


@pytest.mark.parametrize("name", ["transformers", "datasets", "accelerate", "seqeval"])
def test_notebook_pins_match_lock(name: str) -> None:
    nb = json.loads((ROOT / "training" / "train.ipynb").read_text(encoding="utf-8"))
    lines = [line for cell in nb["cells"] for line in cell["source"]]
    assert pins(lines).get(name) == locked()[name]


def test_space_readme_sdk_version_matches_lock() -> None:
    readme = (ROOT / "app" / "README.md").read_text(encoding="utf-8")
    m = re.search(r"^sdk_version: (\S+)$", readme, re.MULTILINE)
    assert m is not None and m[1] == locked()["gradio"]
