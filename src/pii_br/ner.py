"""Entrada e saída do modelo NER sem dependências: tokenização WordPiece, janelas e decodificação.

Serve a qualquer runtime que devolva logits (ONNX Runtime em Python ou no navegador). O
modelo foi treinado com o rótulo no primeiro subtoken de cada palavra de `words.py`; aqui a
decodificação usa as mesmas palavras, então cada palavra recebe o rótulo do seu primeiro
subtoken e o score é a probabilidade desse rótulo. Palavras vizinhas com `B-X`/`I-X` viram um
span; o score do span é a média das palavras.

O tokenizer reproduz o do BERTimbau (`BertNormalizer` sem lowercase nem remoção de acento,
`BertPreTokenizer` e `WordPiece` com prefixo `##`); `tests/test_ner.py` compara os ids com o
tokenizer do `transformers`.
"""

from __future__ import annotations

import json
import math
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from pii_br.types import BIO_LABELS, Entity, Span
from pii_br.words import word_spans

__all__ = ["WINDOW", "Encoding", "WordPiece", "decode", "predict", "to_spans"]

WINDOW = 256  # tokens por janela, com [CLS] e [SEP]
STRIDE = 64  # sobreposição entre janelas vizinhas
_MAX_WORD = 100  # `max_input_chars_per_word` do WordPiece


def _is_control(ch: str) -> bool:
    if ch in "\t\n\r":
        return False
    return ch in "\x00�" or unicodedata.category(ch) in ("Cc", "Cf")


def _is_punct(ch: str) -> bool:
    cp = ord(ch)
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
        return True
    return unicodedata.category(ch).startswith("P")


def _is_cjk(cp: int) -> bool:
    return (
        0x4E00 <= cp <= 0x9FFF
        or 0x3400 <= cp <= 0x4DBF
        or 0x20000 <= cp <= 0x2A6DF
        or 0x2A700 <= cp <= 0x2B73F
        or 0x2B740 <= cp <= 0x2B81F
        or 0x2B820 <= cp <= 0x2CEAF
        or 0xF900 <= cp <= 0xFAFF
        or 0x2F800 <= cp <= 0x2FA1F
    )


def _pre_tokens(word: str) -> list[str]:
    """Normaliza e quebra uma palavra como o `BertPreTokenizer`: espaço separa, pontuação e
    ideogramas viram peças próprias, caracteres de controle somem."""
    pieces: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            pieces.append("".join(current))
            current.clear()

    for ch in word:
        if _is_control(ch):
            continue
        if ch.isspace():
            flush()
        elif _is_punct(ch) or _is_cjk(ord(ch)):
            flush()
            pieces.append(ch)
        else:
            current.append(ch)
    flush()
    return pieces


class WordPiece:
    """Vocabulário WordPiece (o `model.vocab` de um `tokenizer.json`)."""

    def __init__(self, vocab: dict[str, int]) -> None:
        self.vocab = vocab
        self.unk = vocab["[UNK]"]
        self.cls = vocab["[CLS]"]
        self.sep = vocab["[SEP]"]

    @classmethod
    def from_tokenizer_json(cls, raw: str) -> WordPiece:
        data = json.loads(raw)
        return cls({str(k): int(v) for k, v in data["model"]["vocab"].items()})

    def _piece_ids(self, piece: str) -> list[int]:
        if len(piece) > _MAX_WORD:
            return [self.unk]
        ids: list[int] = []
        start = 0
        while start < len(piece):
            end = len(piece)
            found = None
            while start < end:
                sub = piece[start:end] if start == 0 else "##" + piece[start:end]
                if sub in self.vocab:
                    found = self.vocab[sub]
                    break
                end -= 1
            if found is None:
                return [self.unk]
            ids.append(found)
            start = end
        return ids

    def word_ids(self, word: str) -> list[int]:
        return [i for piece in _pre_tokens(word) for i in self._piece_ids(piece)]

    def encode(self, text: str) -> Encoding:
        ids: list[int] = []
        first: list[int] = []  # índice do primeiro token de cada palavra (-1: palavra vazia)
        spans = word_spans(text)
        for start, end in spans:
            word = self.word_ids(text[start:end])
            first.append(len(ids) if word else -1)
            ids.extend(word)
        return Encoding(text, spans, first, _windows(ids, self.cls, self.sep))


@dataclass(frozen=True, slots=True)
class Encoding:
    """Texto tokenizado: palavras (offsets), primeiro token de cada uma e janelas de ids.

    Cada janela é `(início, ids)`: `ids` começa com [CLS] e termina com [SEP], e o token
    `início + k` do texto está na posição `k + 1` da janela.
    """

    text: str
    words: list[tuple[int, int]]
    first: list[int]
    windows: list[tuple[int, list[int]]]


def _windows(ids: list[int], cls: int, sep: int) -> list[tuple[int, list[int]]]:
    size = WINDOW - 2
    step = size - STRIDE
    out: list[tuple[int, list[int]]] = []
    start = 0
    while True:
        out.append((start, [cls, *ids[start : start + size], sep]))
        if start + size >= len(ids):
            return out
        start += step


def _best_window(token: int, windows: list[tuple[int, list[int]]]) -> tuple[int, int]:
    """Janela em que o token fica mais longe da borda (mais contexto dos dois lados)."""
    best, best_margin = 0, -1
    for w, (start, ids) in enumerate(windows):
        pos = token - start
        n = len(ids) - 2
        if 0 <= pos < n:
            left = pos if start > 0 else WINDOW
            right = n - 1 - pos if w < len(windows) - 1 else WINDOW
            margin = min(left, right)
            if margin > best_margin:
                best, best_margin = w, margin
    return best, token - windows[best][0] + 1


def _softmax_best(row: Sequence[float]) -> tuple[int, float]:
    top = max(range(len(row)), key=row.__getitem__)
    peak = row[top]
    total = math.fsum(math.exp(x - peak) for x in row)
    return top, 1.0 / total


def decode(encoding: Encoding, logits: Sequence[Sequence[Sequence[float]]]) -> list[Span]:
    """Spans a partir dos logits de cada janela (`logits[janela][posição][rótulo]`)."""
    groups: list[dict[str, float | int | str]] = []
    prev_tag = "O"
    for (start, end), token in zip(encoding.words, encoding.first, strict=True):
        if token < 0:
            continue
        w, pos = _best_window(token, encoding.windows)
        label_id, score = _softmax_best(logits[w][pos])
        bio = BIO_LABELS[label_id]
        tag = bio[2:] if bio != "O" else "O"
        if groups and tag == prev_tag and not bio.startswith("B-"):
            g = groups[-1]
            g["end"] = end
            g["sum"] = float(g["sum"]) + score
            g["n"] = int(g["n"]) + 1
        else:
            groups.append({"entity_group": tag, "start": start, "end": end, "sum": score, "n": 1})
        prev_tag = tag
    raw = [
        {
            "entity_group": g["entity_group"],
            "start": g["start"],
            "end": g["end"],
            "score": float(g["sum"]) / int(g["n"]),
        }
        for g in groups
        if g["entity_group"] != "O"
    ]
    return to_spans(encoding.text, raw)


def predict(
    tokenizer: WordPiece, logits: Callable[[list[int]], Sequence[Sequence[float]]], text: str
) -> list[Span]:
    """Pipeline completo: tokeniza, roda `logits` em cada janela de ids e decodifica."""
    if not text.strip():
        return []
    encoding = tokenizer.encode(text)
    return decode(encoding, [logits(ids) for _, ids in encoding.windows])


def to_spans(text: str, raw: Sequence[Mapping[str, object]]) -> list[Span]:
    """Converte saídas no formato do pipeline do `transformers` (`entity_group`, `start`,
    `end`, `score`) em `Span`: ignora rótulos desconhecidos, apara espaços das bordas e une
    pedaços vizinhos do mesmo rótulo (ver `_joinable`)."""
    spans: list[Span] = []
    for item in raw:
        label = str(item["entity_group"])
        if label not in Entity.__members__:
            continue
        start, end = int(str(item["start"])), int(str(item["end"]))
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            score = float(str(item["score"]))
            spans.append(Span(start, end, Entity(label), text[start:end], "modelo", score))
    return _join_adjacent(spans, text)


def _joinable(label: Entity, gap: str) -> bool:
    """Pedaços vizinhos do mesmo rótulo que o pipeline às vezes separa. Vírgula só une
    endereço ("Rua A, 12"): entre nomes ("Ana, Bruno") ela separa duas pessoas."""
    if len(gap) > 2:
        return False
    allowed = " ,-" if label is Entity.ENDERECO else " -"
    return not gap.strip(allowed)


def _join_adjacent(spans: list[Span], text: str) -> list[Span]:
    out: list[Span] = []
    for span in sorted(spans, key=lambda s: s.start):
        if out:
            prev = out[-1]
            gap = text[prev.end : span.start]
            if prev.label is span.label and _joinable(span.label, gap):
                score = min(prev.score, span.score)
                out[-1] = Span(
                    prev.start, span.end, prev.label, text[prev.start : span.end], "modelo", score
                )
                continue
        out.append(span)
    return out
