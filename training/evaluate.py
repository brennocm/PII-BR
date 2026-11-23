"""Avalia o pipeline híbrido, o modelo sozinho e os baselines (regex e Presidio).

Uso:
    uv run --all-extras --group baseline python -m training.evaluate \\
        --model brennocm/PII-BR --data data --out outputs/eval [--write-card]

Métricas por span em caracteres, comparando com os `spans` de referência:
- exato: mesmo início, fim e rótulo;
- sobreposição: mesmo rótulo e algum caractere em comum (o que importa para mascarar).
O F1 "estrito" do seqeval (só do modelo) vem do notebook de treino (`metrics_seqeval.json`).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pii_br import Detector, Entity, Span, recognize

SPLITS = ("test", "test_hard", "test_real")
LABELS = [e.value for e in Entity]
Gold = tuple[int, int, str]
System = Callable[[str], list[Gold]]

LATENCY_TEXT = (
    "Prezados, segue o cadastro do novo cliente para conferência. Nome: Maria Aparecida "
    "Souza, CPF 123.456.789-09, RG 12.345.678-9, nascida em 03/07/1990. Endereço: Rua das "
    "Flores, 123, ap. 42, Jardim América, São Paulo/SP, CEP 01310-100. Contatos: celular "
    "(11) 98765-4321 e e-mail maria.souza@example.com. A empresa contratante é a Exemplo "
    "Serviços Ltda., CNPJ 12.ABC.345/01DE-35, representada por João Pedro Lima, telefone "
    "(21) 2345-6789. O pedido 2024000123 foi aberto em 15/11/2026 e o protocolo de "
    "atendimento é 98765432100123. Qualquer dúvida, ligar para a central 0800 123 4567. "
    "Observação: a cliente pediu que o boleto seja enviado para o endereço de cobrança, e "
    "não para o e-mail. Atenciosamente, equipe de cadastro."
)


# ----------------------------------------------------------------- métricas


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other: Counts) -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn

    def scores(self) -> dict[str, float]:
        p = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        r = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return {"precision": p, "recall": r, "f1": f, "support": self.tp + self.fn}


@dataclass
class SplitResult:
    exact: dict[str, Counts] = field(default_factory=lambda: {})
    overlap: dict[str, Counts] = field(default_factory=lambda: {})
    errors: list[dict[str, Any]] = field(default_factory=lambda: [])


def match(gold: list[Gold], pred: list[Gold], *, exact: bool) -> tuple[set[int], set[int]]:
    """Casamento um-para-um; devolve os índices casados de gold e de pred."""
    used_g: set[int] = set()
    used_p: set[int] = set()
    for gi, (gs, ge, gl) in enumerate(gold):
        for pi, (ps, pe, pl) in enumerate(pred):
            if pi in used_p or pl != gl:
                continue
            if (gs, ge) == (ps, pe) if exact else (ps < ge and gs < pe):
                used_g.add(gi)
                used_p.add(pi)
                break
    return used_g, used_p


def score_split(rows: list[dict[str, Any]], system: System) -> SplitResult:
    result = SplitResult()
    for table in (result.exact, result.overlap):
        for label in LABELS:
            table[label] = Counts()
    for row in rows:
        text: str = row["text"]
        gold: list[Gold] = [(s["start"], s["end"], s["label"]) for s in row["spans"]]
        pred = system(text)
        for exact, table in ((True, result.exact), (False, result.overlap)):
            mg, mp = match(gold, pred, exact=exact)
            for i, (_, _, label) in enumerate(gold):
                if i in mg:
                    table[label].tp += 1
                else:
                    table[label].fn += 1
            for i, (_, _, label) in enumerate(pred):
                if i not in mp and label in table:
                    table[label].fp += 1
            if not exact:
                missed = [gold[i] for i in range(len(gold)) if i not in mg]
                extra = [pred[i] for i in range(len(pred)) if i not in mp]
                if missed or extra:
                    result.errors.append(
                        {
                            "id": row["id"],
                            "text": text,
                            "missed": [(text[s:e], lab) for s, e, lab in missed],
                            "extra": [(text[s:e], lab) for s, e, lab in extra],
                        }
                    )
    return result


def summarize(table: dict[str, Counts]) -> dict[str, Any]:
    total = Counts()
    for counts in table.values():
        total.add(counts)
    out: dict[str, Any] = {label: c.scores() for label, c in table.items() if c.tp + c.fn}
    out["micro"] = total.scores()
    return out


# ----------------------------------------------------------------- sistemas


def as_gold(spans: list[Span]) -> list[Gold]:
    return [(s.start, s.end, s.label.value) for s in spans]


def regex_system() -> System:
    return lambda text: as_gold(recognize(text))


def presidio_system() -> System:
    """Presidio com recognizers padrão registrados para `pt` e spaCy pt_core_news_lg."""
    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_analyzer.predefined_recognizers import (
        CreditCardRecognizer,
        EmailRecognizer,
        IpRecognizer,
        PhoneRecognizer,
        SpacyRecognizer,
        UrlRecognizer,
    )

    nlp_engine = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "pt", "model_name": "pt_core_news_lg"}],
            "ner_model_configuration": {
                "model_to_presidio_entity_mapping": {
                    "PER": "PERSON",
                    "LOC": "LOCATION",
                    "ORG": "ORGANIZATION",
                },
                "labels_to_ignore": ["MISC"],
            },
        }
    ).create_engine()
    registry = RecognizerRegistry(supported_languages=["pt"])
    for recognizer in (
        EmailRecognizer(supported_language="pt"),
        PhoneRecognizer(supported_language="pt", supported_regions=("BR",)),
        CreditCardRecognizer(supported_language="pt"),
        IpRecognizer(supported_language="pt"),
        UrlRecognizer(supported_language="pt"),
        SpacyRecognizer(supported_language="pt"),
    ):
        registry.add_recognizer(recognizer)
    engine = AnalyzerEngine(registry=registry, nlp_engine=nlp_engine, supported_languages=["pt"])
    mapping = {
        "PERSON": "NOME",
        "EMAIL_ADDRESS": "EMAIL",
        "PHONE_NUMBER": "TELEFONE",
        "LOCATION": "ENDERECO",
    }

    def run(text: str) -> list[Gold]:
        results = engine.analyze(text=text, language="pt", score_threshold=0.4)
        return [
            (r.start, r.end, mapping[r.entity_type]) for r in results if r.entity_type in mapping
        ]

    return run


def model_systems(model_id: str, threshold: float) -> tuple[System, System, Detector]:
    from pii_br.model import NerModel

    model = NerModel(model_id, quantize=True)

    def model_only(text: str) -> list[Gold]:
        return as_gold([s for s in model.predict(text) if s.score >= threshold])

    detector = Detector(model, threshold=threshold)
    return model_only, lambda text: as_gold(detector.detect(text)), detector


def latency(detector: Detector, runs: int = 10) -> dict[str, float]:
    detector.detect(LATENCY_TEXT)  # aquecimento
    times: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        detector.detect(LATENCY_TEXT)
        times.append(time.perf_counter() - start)
    return {
        "chars": len(LATENCY_TEXT),
        "median_s": statistics.median(times),
        "max_s": max(times),
    }


# ----------------------------------------------------------------- relatório


def load_split(data: Path, split: str) -> list[dict[str, Any]]:
    path = data / f"{split}.jsonl"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def fmt(x: float) -> str:
    return f"{x:.3f}".replace(".", ",")


def report_markdown(results: dict[str, Any]) -> str:
    lines: list[str] = []
    systems = list(results["systems"])
    for split in results["splits"]:
        lines += [
            f"### `{split}`",
            "",
            "| Sistema | Precisão | Recall | F1 (sobreposição) | F1 (exato) |",
        ]
        lines.append("| --- | --- | --- | --- | --- |")
        for system in systems:
            r = results["systems"][system].get(split)
            if not r:
                continue
            o, e = r["overlap"]["micro"], r["exact"]["micro"]
            lines.append(
                f"| {system} | {fmt(o['precision'])} | {fmt(o['recall'])} "
                f"| {fmt(o['f1'])} | {fmt(e['f1'])} |"
            )
        lines.append("")
    best = "híbrido" if "híbrido" in results["systems"] else systems[-1]
    lines += [f"### Por entidade — {best} (sobreposição)", ""]
    header = "| Entidade | " + " | ".join(f"`{s}` P / R / F1" for s in results["splits"]) + " |"
    lines += [header, "| --- " * (len(results["splits"]) + 1) + "|"]
    for label in LABELS:
        cells: list[str] = []
        for split in results["splits"]:
            r = results["systems"][best].get(split, {}).get("overlap", {}).get(label)
            cells.append(
                f"{fmt(r['precision'])} / **{fmt(r['recall'])}** / {fmt(r['f1'])}" if r else "—"
            )
        lines.append(f"| `{label}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def errors_markdown(errors: list[dict[str, Any]], limit: int = 15) -> str:
    """Falsos negativos primeiro: deixar passar PII é o erro caro."""
    ordered = sorted(errors, key=lambda e: (not e["missed"], e["id"]))[:limit]
    lines = ["| Frase | Não detectado | Detectado a mais |", "| --- | --- | --- |"]
    for e in ordered:
        text = e["text"].replace("\n", " ").replace("|", "\\|")
        missed = "; ".join(f"`{t}` ({lab})" for t, lab in e["missed"]) or "—"
        extra = "; ".join(f"`{t}` ({lab})" for t, lab in e["extra"]) or "—"
        lines.append(f"| {text[:160]} | {missed} | {extra} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--model", help="repo do Hub ou pasta local; sem ele, só baselines")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("outputs/eval"))
    parser.add_argument("--splits", nargs="+", default=list(SPLITS))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--no-presidio", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="máx. de frases por split (0 = todas)")
    parser.add_argument("--write-card", action="store_true", help="gera cards/model.README.md")
    args = parser.parse_args()

    systems: dict[str, System] = {"só regex": regex_system()}
    if not args.no_presidio:
        systems["Presidio (padrão, pt)"] = presidio_system()
    detector = None
    if args.model:
        model_only, hybrid, detector = model_systems(args.model, args.threshold)
        systems["só modelo"] = model_only
        systems["híbrido"] = hybrid

    results: dict[str, Any] = {
        "model": args.model,
        "threshold": args.threshold,
        "splits": [],
        "systems": {},
    }
    errors: dict[str, list[dict[str, Any]]] = {}
    for split in args.splits:
        rows = load_split(args.data, split)
        if args.limit:
            rows = rows[: args.limit]
        if not rows:
            continue
        results["splits"].append(split)
        for name, system in systems.items():
            start = time.perf_counter()
            r = score_split(rows, system)
            results["systems"].setdefault(name, {})[split] = {
                "exact": summarize(r.exact),
                "overlap": summarize(r.overlap),
                "sentences": len(rows),
                "seconds": round(time.perf_counter() - start, 1),
            }
            if name in ("híbrido", "só regex"):
                errors.setdefault(f"{name}/{split}", r.errors)
            o = results["systems"][name][split]["overlap"]["micro"]
            print(
                f"{split:10} {name:22} P={o['precision']:.3f} R={o['recall']:.3f} F1={o['f1']:.3f}"
            )
    if detector is not None:
        results["latency"] = latency(detector)
        print("latência:", results["latency"])

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), "utf-8"
    )
    report = report_markdown(results)
    key = "híbrido/test_real" if "híbrido/test_real" in errors else next(iter(errors), "")
    error_table = errors_markdown(errors.get(key, []))
    (args.out / "report.md").write_text(f"{report}\n\n## Erros ({key})\n\n{error_table}\n", "utf-8")
    print(f"relatório: {args.out / 'report.md'}")
    if args.write_card and args.model:
        from training.model_card import write_card

        write_card(results, report, error_table, key)


if __name__ == "__main__":
    main()
