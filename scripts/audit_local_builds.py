"""Audita pacotes do uv.lock com versão local (ex.: `torch 2.15.0+cpu`).

O `pip-audit` pula esses pacotes porque `2.15.0+cpu` não existe no PyPI. As vulnerabilidades
são publicadas para a versão pública (`2.15.0`), então auditamos essa.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[1]


def local_builds() -> list[str]:
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    return sorted(
        f"{p['name']}=={p['version'].split('+')[0]}"
        for p in lock["package"]
        if "+" in p.get("version", "")
    )


def main() -> int:
    pins = local_builds()
    if not pins:
        return 0
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("\n".join(pins) + "\n")
    try:
        cmd = ["pip-audit", "--no-deps", "--disable-pip", "-r", f.name]
        return subprocess.run(cmd, check=False).returncode  # noqa: S603
    finally:
        Path(f.name).unlink()


if __name__ == "__main__":
    sys.exit(main())
