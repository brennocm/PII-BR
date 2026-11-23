"""PII-BR no Hugging Face Spaces: detecta, destaca e mascara dados pessoais em PT-BR.

Privacidade: o texto só existe em memória durante a requisição. O app não registra o
conteúdo em log, não usa flagging nem analytics, apaga o arquivo enviado assim que o lê e
apaga o arquivo mascarado depois de alguns minutos.

Limites: cada requisição processa no máximo `MAX_CHARS` caracteres de texto colado ou
`MAX_FILE_CHARS` caracteres / `MAX_CELLS` células de um arquivo. O Space gratuito atende
uma requisição por vez, então sem limite um arquivo grande travaria a demo para todos.
"""

from __future__ import annotations

import csv
import io
import logging
import os
import tempfile
import time
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
from gradio.themes import Base as BaseTheme

from pii_br import Detector, Entity, Masker, Span, Strategy

log = logging.getLogger("pii-br")  # só eventos de ciclo de vida, nunca o texto do usuário

# No Space, o HF define SPACE_AUTHOR_NAME; o modelo padrão é o repo homônimo do autor.
_AUTHOR = os.environ.get("SPACE_AUTHOR_NAME")
MODEL_ID = os.environ.get("PII_BR_MODEL", f"{_AUTHOR}/PII-BR" if _AUTHOR else "")
MODEL_REVISION = os.environ.get("PII_BR_MODEL_REVISION") or None  # commit fixo do modelo
MAX_CHARS = 20_000
MAX_FILE_CHARS = 100_000
MAX_CELLS = 2_000
MAX_UPLOAD_BYTES = 1_048_576
OUTPUT_DIR = Path(tempfile.gettempdir()) / "pii-br-saida"
OUTPUT_TTL_SECONDS = 600
csv.field_size_limit(MAX_UPLOAD_BYTES)
STRATEGY_LABELS: dict[str, Strategy] = {
    "Redação: [CPF]": "redact",
    "Máscara parcial: ***.456.789-**": "partial",
    "Pseudônimo consistente: <CPF_1>": "pseudonym",
    "Dado sintético válido": "synthetic",
}
ENTITY_NAMES = {
    Entity.NOME: "Nome",
    Entity.CPF: "CPF",
    Entity.CNPJ: "CNPJ",
    Entity.RG: "RG",
    Entity.TELEFONE: "Telefone",
    Entity.EMAIL: "E-mail",
    Entity.ENDERECO: "Endereço",
    Entity.CEP: "CEP",
    Entity.DATA_NASC: "Data de nascimento",
}
# Cores das categorias no destaque: tons médios que funcionam no tema claro e no escuro.
COLOR_MAP = {
    "NOME": "#3b6ea8",
    "CPF": "#2f8a68",
    "CNPJ": "#5c8a2f",
    "RG": "#8a6d2f",
    "TELEFONE": "#7a4fa0",
    "EMAIL": "#2f7f8a",
    "ENDERECO": "#a0524f",
    "CEP": "#6b7280",
    "DATA_NASC": "#a0712f",
    "ALERTA": "#c2410c",
}
NOTICE = (
    "O texto é processado em memória e não é armazenado nem registrado em log. "
    "Mesmo assim, não cole dados reais de terceiros numa demo pública: use os exemplos "
    "ou dados inventados."
)
EXAMPLES = [
    "Oi, sou Maria Aparecida Souza, meu CPF é 123.456.789-09 e moro na Rua das Flores, "
    "123, ap. 42, CEP 01310-100. Pode me ligar no (11) 98765-4321?",
    "manda o pix pro meu cpf 529.982.247-24 ou no zap 21 99876-5432 que eu confirmo",
    "CONTRATADA: Exemplo Serviços Ltda., inscrita no CNPJ sob o nº 12.ABC.345/01DE-35, "
    "representada por João Pedro Lima, RG 12.345.678-9, nascido em 3 de julho de 1985.",
    '2026-11-20T13:44:02Z INFO kyc aprovado nome="Fernanda Arantes" cpf=11144477735 '
    "email=fernanda.a@example.com cep=70040010",
    "Prezados, encaminho os dados do novo colaborador: Rafael Takeshi Nakamura, "
    "nascido em 14/06/1992, e-mail rafael.nakamura@example.com, celular +55 41 99652-0081.",
    "O pedido 2024000123 foi faturado em 15/11/2026 no valor de R$ 1.234,56. "
    "Dúvidas: central 0800 123 4567.",
]


# ------------------------------------------------------------------ detecção


@cache
def get_detector() -> tuple[Detector, str]:
    """Carrega o modelo uma vez. Sem modelo (ou se ele falhar), segue só com as regras."""
    rules_only = "regras determinísticas (CPF, CNPJ, telefone, e-mail, CEP)"
    if not MODEL_ID:
        return Detector(), f"Sem modelo configurado: só {rules_only}."
    try:
        from pii_br.model import NerModel

        model = NerModel(MODEL_ID, quantize=True, revision=MODEL_REVISION)
    except Exception as exc:  # o app continua útil mesmo sem o modelo
        log.warning("modelo indisponível (%s): %s", MODEL_ID, type(exc).__name__)
        return Detector(), f"Modelo `{MODEL_ID}` indisponível agora: só {rules_only}."
    return Detector(model), f"Modelo `{MODEL_ID}` (int8, CPU) + regras determinísticas."


def _detect(text: str, entities: list[str], threshold: float) -> list[Span]:
    detector, _ = get_detector()
    wanted = [Entity(e) for e in entities]
    return Detector(detector.model, threshold=threshold).detect(text, wanted)


def _highlight(text: str, spans: list[Span]) -> list[tuple[str, str | None]]:
    out: list[tuple[str, str | None]] = []
    last = 0
    for s in spans:
        if s.start > last:
            out.append((text[last : s.start], None))
        label = "ALERTA" if s.valid is False else s.label.value
        out.append((text[s.start : s.end], label))
        last = s.end
    if last < len(text):
        out.append((text[last:], None))
    return out


def _span_rows(spans: list[Span]) -> list[dict[str, Any]]:
    return [
        {
            "entidade": s.label.value,
            "texto": s.text,
            "inicio": s.start,
            "fim": s.end,
            "fonte": s.source,
            "confianca": round(s.score, 3),
            "documento_valido": s.valid,
        }
        for s in spans
    ]


def analyze(
    text: str, strategy_label: str, entities: list[str], threshold: float, masker: Masker | None
) -> tuple[list[tuple[str, str | None]], str, list[dict[str, Any]], str, Masker]:
    masker = masker or Masker()  # um sal por sessão, criado na primeira requisição
    text = text or ""
    cut = len(text) > MAX_CHARS
    text = text[:MAX_CHARS]
    if not text.strip():
        return [], "", [], "Cole um texto ou escolha um exemplo.", masker
    spans = _detect(text, entities, threshold)
    masked = masker.mask(text, spans, STRATEGY_LABELS[strategy_label])
    alerts = sum(s.valid is False for s in spans)
    summary = f"{len(spans)} dado(s) pessoal(is) encontrado(s)."
    if alerts:
        summary += (
            f" {alerts} documento(s) com dígito verificador inválido, mascarado(s) mesmo "
            "assim: pode ser erro de digitação de um documento real."
        )
    if cut:
        summary += (
            f" **Texto cortado:** só os primeiros {_br(MAX_CHARS)} caracteres foram analisados; "
            "o resto não aparece no resultado."
        )
    return _highlight(text, spans), masked, _span_rows(spans), summary, masker


# ------------------------------------------------------------------- arquivos


def _br(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def _dialect(sample: str) -> type[csv.Dialect]:
    """Separador mais frequente no cabeçalho: vírgula, ponto e vírgula (padrão do Excel em
    pt-BR), tab ou barra vertical. O `csv.Sniffer` erra quando as linhas têm tamanhos
    diferentes, o que é comum em planilhas exportadas."""
    header = sample.split("\n", 1)[0]
    delimiter = max(",;\t|", key=header.count)
    if not header.count(delimiter):
        return csv.excel

    class Dialect(csv.excel):
        pass

    Dialect.delimiter = delimiter
    return Dialect


def read_columns(file_path: str | None) -> dict[str, Any]:
    hidden = gr.update(choices=[], value=[], visible=False)
    if not file_path or not file_path.lower().endswith(".csv") or not Path(file_path).exists():
        return hidden
    try:
        with open(file_path, encoding="utf-8-sig", newline="") as f:
            sample = f.read(8192)
            f.seek(0)
            header = next(csv.reader(f, _dialect(sample)), [])
    except (UnicodeDecodeError, csv.Error):
        return hidden
    names = list(dict.fromkeys(header))
    return gr.update(choices=names, value=names, visible=True)


def _purge_outputs(now: float) -> None:
    """Apaga arquivos mascarados antigos: o Gradio já copiou cada um para o cache dele."""
    for old in OUTPUT_DIR.glob("pii-br-mascarado-*"):
        try:
            if now - old.stat().st_mtime > OUTPUT_TTL_SECONDS:
                old.unlink()
        except FileNotFoundError:
            pass


def _write_output(result: str, suffix: str) -> str:
    _purge_outputs(time.time())
    OUTPUT_DIR.mkdir(mode=0o700, exist_ok=True)
    fd, out_path = tempfile.mkstemp(prefix="pii-br-mascarado-", suffix=suffix, dir=OUTPUT_DIR)
    with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
        f.write(result)
    return out_path


class _Budget:
    """Conta caracteres e chamadas ao detector de um arquivo; ver os limites no topo."""

    def __init__(self) -> None:
        self.chars = 0
        self.cells = 0

    def take(self, value: str) -> bool:
        if self.chars + len(value) > MAX_FILE_CHARS or self.cells >= MAX_CELLS:
            return False
        self.chars += len(value)
        self.cells += 1
        return True


def _mask_csv(
    raw: str, columns: list[str], mask_text: Callable[[str], tuple[str, int]]
) -> tuple[str, int, str]:
    dialect = _dialect(raw)
    rows = csv.reader(io.StringIO(raw), dialect)
    header = next(rows, [])
    targets = [i for i, name in enumerate(header) if name in set(columns)]
    out = io.StringIO()
    writer = csv.writer(out, dialect)
    writer.writerow(header)
    budget, found, note = _Budget(), 0, ""
    for line, row in enumerate(rows, start=2):
        cells = [row[i] for i in targets if i < len(row) and row[i]]
        if not all(map(budget.take, cells)):
            note = (
                f" **Arquivo cortado na linha {line}:** o limite da demo é "
                f"{_br(MAX_FILE_CHARS)} caracteres ou {_br(MAX_CELLS)} células; "
                "as linhas seguintes não estão no resultado."
            )
            break
        for i in targets:
            if i < len(row) and row[i]:
                row[i], n = mask_text(row[i])
                found += n
        writer.writerow(row)
    return out.getvalue(), found, note


def process_file(
    file_path: str | None,
    columns: list[str],
    strategy_label: str,
    entities: list[str],
    threshold: float,
    masker: Masker | None,
) -> tuple[str | None, str, Masker, None]:
    """Devolve (arquivo mascarado, resumo, sessão, None). O último valor limpa o campo de
    upload, porque o arquivo enviado é apagado do disco assim que é lido."""
    masker = masker or Masker()
    if not file_path:
        return None, "Envie um arquivo .txt ou .csv.", masker, None
    source = Path(file_path)
    strategy = STRATEGY_LABELS[strategy_label]
    try:
        raw = source.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return None, "O arquivo já foi processado e apagado; envie-o de novo.", masker, None
    except UnicodeDecodeError:
        return None, "Não consegui ler o arquivo: salve-o em UTF-8 e envie de novo.", masker, None
    finally:
        source.unlink(missing_ok=True)  # o upload não fica no disco do Space

    def mask_text(value: str) -> tuple[str, int]:
        spans = _detect(value, entities, threshold)
        return masker.mask(value, spans, strategy), len(spans)

    if source.suffix.lower() == ".csv":
        try:
            result, found, note = _mask_csv(raw, columns, mask_text)
        except csv.Error:
            return None, "Não consegui ler o CSV: confira aspas e separadores.", masker, None
        suffix = ".csv"
    else:
        note = ""
        if len(raw) > MAX_FILE_CHARS:
            note = (
                f" **Arquivo cortado:** só os primeiros {_br(MAX_FILE_CHARS)} caracteres foram "
                "analisados; o resto não está no resultado."
            )
        result, found = mask_text(raw[:MAX_FILE_CHARS])
        suffix = ".txt"
    summary = f"{found} dado(s) pessoal(is) mascarado(s) no arquivo.{note}"
    return _write_output(result, suffix), summary, masker, None


# ------------------------------------------------------------------ interface

CSS = """
:root { --pii-signal: #c2410c; }
#masked textarea { font-family: ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
  font-size: 0.95rem; line-height: 1.55; }
#notice { font-size: 0.9rem; opacity: 0.85; }
#summary { font-variant-numeric: tabular-nums; }
::selection { background: color-mix(in oklch, var(--color-accent) 30%, transparent); }
"""


def build_demo() -> gr.Blocks:
    entity_choices = [(name, e.value) for e, name in ENTITY_NAMES.items()]
    with gr.Blocks(title="PII-BR — detector de dados pessoais", analytics_enabled=False,
                   delete_cache=(3600, 3600)) as demo:  # fmt: skip
        masker = gr.State(None)
        gr.Markdown(
            "# PII-BR\n"
            "Encontra e mascara dados pessoais em textos em português do Brasil: nome, CPF, "
            "CNPJ (inclusive o alfanumérico), RG, telefone, e-mail, endereço, CEP e data de "
            "nascimento. Combina um modelo treinado em português com validação de dígito "
            "verificador."
        )
        gr.Markdown(get_detector()[1])
        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                with gr.Tab("Texto"):
                    text_in = gr.Textbox(
                        label="Texto", lines=9, max_lines=30, placeholder="Cole aqui um texto…"
                    )
                    run = gr.Button("Mascarar texto", variant="primary")
                    gr.Examples(EXAMPLES, inputs=text_in, label="Exemplos")
                with gr.Tab("Arquivo"):
                    file_in = gr.File(
                        label="Arquivo .txt ou .csv (até 1 MB)",
                        file_types=[".txt", ".csv"],
                        type="filepath",
                    )
                    columns = gr.CheckboxGroup(label="Colunas a mascarar", visible=False)
                    run_file = gr.Button("Mascarar arquivo", variant="primary")
                    file_out = gr.File(label="Arquivo mascarado")
                    file_summary = gr.Markdown()
                with gr.Accordion("Opções", open=True):
                    strategy = gr.Radio(
                        list(STRATEGY_LABELS), value=next(iter(STRATEGY_LABELS)), label="Estratégia"
                    )
                    entities = gr.CheckboxGroup(
                        entity_choices,
                        value=[v for _, v in entity_choices],
                        label="Entidades",
                    )
                    threshold = gr.Slider(
                        0.1, 0.95, value=0.5, step=0.05, label="Confiança mínima do modelo"
                    )
            with gr.Column(scale=6):
                summary = gr.Markdown(elem_id="summary")
                masked = gr.Textbox(
                    label="Texto mascarado",
                    lines=9,
                    interactive=False,
                    buttons=["copy"],
                    elem_id="masked",
                )
                highlighted = gr.HighlightedText(
                    label="Dados encontrados",
                    color_map=COLOR_MAP,
                    show_legend=True,
                    combine_adjacent=False,
                )
                with gr.Accordion("Spans em JSON", open=False):
                    spans_json = gr.JSON(label="Spans")
        gr.Markdown(NOTICE, elem_id="notice")
        gr.Markdown(
            "A ferramenta reduz risco, mas não certifica anonimização: pela LGPD (art. 12), "
            "mascarar entidades não remove quasi-identificadores como cargo, bairro ou datas."
        )

        inputs = [text_in, strategy, entities, threshold, masker]
        outputs = [highlighted, masked, spans_json, summary, masker]
        run.click(analyze, inputs, outputs, show_progress="minimal")
        text_in.submit(analyze, inputs, outputs)
        file_in.change(read_columns, file_in, columns)
        run_file.click(
            process_file,
            [file_in, columns, strategy, entities, threshold, masker],
            [file_out, file_summary, masker, file_in],
        )
    return demo


demo = build_demo()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo.launch(
        css=CSS,
        theme=BaseTheme(primary_hue="emerald", neutral_hue="slate", radius_size="sm"),
        max_file_size="1mb",
        footer_links=["api"],
    )
