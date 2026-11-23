"""Tokenização, janelas e decodificação de `pii_br.ner` (sem torch nem onnxruntime)."""

import json
from pathlib import Path

import pytest

from pii_br import Entity
from pii_br.ner import STRIDE, WINDOW, WordPiece, decode, predict
from pii_br.types import BIO_LABELS
from pii_br.words import words

SAMPLES = [
    "Oi, sou Maria Aparecida Souza, CPF 123.456.789-09, moro na Rua das Flores, 123.",
    "manda o pix pro zap 21 99876-5432 kkkk 🎉🎉 valeu!!!",
    "CNPJ 12.ABC.345/01DE-35 — e-mail joão.silva@example.com (urgente)",
    "Ação de São João: d'água, R$ 1.234,56 ½ ° § ¿qué? «aspas» “curvas”",
    "controle\x00aqui​junto\ttab\nlinha 東京 x",
    "palavra" * 20 + " fim",  # mais de 100 caracteres: [UNK]
    "",
    "   ",
]
VOCAB_WORDS = [
    *["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    *[
        "Oi",
        ",",
        "sou",
        "Maria",
        "Souza",
        "CPF",
        "123",
        ".",
        "456",
        "789",
        "-",
        "09",
        "moro",
        "na",
        "Rua",
        "das",
        "Flores",
        "manda",
        "o",
        "pix",
    ],
    *[
        "zap",
        "21",
        "998",
        "##76",
        "5432",
        "valeu",
        "!",
        "CNPJ",
        "12",
        "AB",
        "##C",
        "345",
        "/",
        "01",
        "##DE",
        "35",
        "e",
        "mail",
        "(",
        ")",
        ":",
        "R",
        "$",
    ],
    *[
        "Ação",
        "de",
        "São",
        "João",
        "d",
        "'",
        "água",
        "1",
        "234",
        "56",
        "東",
        "京",
        "x",
        "fim",
        "##e",
        "##s",
        "##a",
        "##o",
        "##u",
        "palavra",
    ],
    *["##palavra", "controle", "aqui", "junto", "tab", "linha", "Apa", "##re", "##cida"],
]


def small_vocab() -> dict[str, int]:
    return {w: i for i, w in enumerate(dict.fromkeys(VOCAB_WORDS))}


def oracle(vocab: dict[str, int]):
    """Tokenizer do HF com a mesma configuração do BERTimbau. Como no treino, ele recebe as
    palavras de `words.py` já separadas (`is_split_into_words`)."""
    tokenizers = pytest.importorskip("tokenizers")
    tok = tokenizers.Tokenizer(
        tokenizers.models.WordPiece(vocab, unk_token="[UNK]", max_input_chars_per_word=100)  # noqa: S106
    )
    tok.normalizer = tokenizers.normalizers.BertNormalizer(
        clean_text=True, handle_chinese_chars=True, strip_accents=None, lowercase=False
    )
    tok.pre_tokenizer = tokenizers.pre_tokenizers.BertPreTokenizer()
    return tok


@pytest.mark.parametrize("text", SAMPLES)
def test_ids_match_hf_tokenizer(text: str) -> None:
    vocab = small_vocab()
    ours = WordPiece(vocab).encode(text)
    flat = [i for _, ids in ours.windows for i in ids[1:-1]]
    assert (
        flat
        == oracle(vocab).encode(words(text), is_pretokenized=True, add_special_tokens=False).ids
    )


def test_ids_match_real_tokenizer_if_cached() -> None:
    """Confere com o vocabulário real do modelo publicado, se ele estiver no cache do HF."""
    cache = Path.home() / ".cache/huggingface/hub/models--brennocm--PII-BR/snapshots"
    found = sorted(cache.glob("*/tokenizer.json"))
    if not found:
        pytest.skip("tokenizer do modelo fora do cache")
    tokenizers = pytest.importorskip("tokenizers")
    raw = found[-1].read_text(encoding="utf-8")
    real = tokenizers.Tokenizer.from_str(raw)
    ours = WordPiece.from_tokenizer_json(raw)
    for text in SAMPLES:
        enc = ours.encode(text)
        flat = [i for _, ids in enc.windows for i in ids[1:-1]]
        assert (
            flat == real.encode(words(text), is_pretokenized=True, add_special_tokens=False).ids
        ), text


def test_from_tokenizer_json() -> None:
    raw = json.dumps({"model": {"vocab": small_vocab()}})
    assert WordPiece.from_tokenizer_json(raw).vocab == small_vocab()


def test_windows_cover_long_text_with_overlap() -> None:
    tok = WordPiece(small_vocab())
    enc = tok.encode("Maria " * 600)  # 600 tokens
    starts = [s for s, _ in enc.windows]
    assert all(len(ids) <= WINDOW for _, ids in enc.windows)
    assert all(ids[0] == tok.cls and ids[-1] == tok.sep for _, ids in enc.windows)
    assert starts == [0, 190, 380]  # passo = 254 - 64
    assert starts[1] - starts[0] == WINDOW - 2 - STRIDE
    last_start, last_ids = enc.windows[-1]
    assert last_start + len(last_ids) - 2 == 600


def one_hot(label: str, n: int = len(BIO_LABELS)) -> list[float]:
    row = [0.0] * n
    row[BIO_LABELS.index(label)] = 10.0
    return row


def test_decode_groups_bio_by_first_subtoken() -> None:
    tok = WordPiece(small_vocab())
    text = "Maria Aparecida Souza , CPF 123.456"
    enc = tok.encode(text)
    # Maria | Apa ##re ##cida | Souza | , | CPF | 123 . 456  →  tokens 0..9
    labels = ["B-NOME", "I-NOME", "B-CPF", "O", "I-NOME", "O", "O", "B-CPF", "I-CPF", "O"]
    logits = [[one_hot("O"), *(one_hot(lab) for lab in labels), one_hot("O")]]
    spans = decode(enc, logits)
    assert [(s.text, s.label) for s in spans] == [
        ("Maria Aparecida Souza", Entity.NOME),
        ("123.456", Entity.CPF),
    ]
    assert all(s.source == "modelo" and s.score > 0.99 for s in spans)


def test_decode_starts_new_span_on_b_and_averages_score() -> None:
    tok = WordPiece(small_vocab())
    enc = tok.encode("Maria Souza")
    weak = [0.0] * len(BIO_LABELS)
    weak[BIO_LABELS.index("I-NOME")] = 1.0  # probabilidade ~0,13 com 19 rótulos
    logits = [[one_hot("O"), one_hot("B-NOME"), weak, one_hot("O")]]
    (span,) = decode(enc, logits)
    assert span.text == "Maria Souza"
    assert 0.5 < span.score < 0.6  # média de ~1,0 e ~0,13


def test_decode_uses_most_central_window() -> None:
    tok = WordPiece(small_vocab())
    enc = tok.encode("Maria " * 400)
    target = 200  # está nas janelas 0 (posição 200 de 254) e 1 (posição 10): fica com a 0
    logits: list[list[list[float]]] = []
    for w, (start, ids) in enumerate(enc.windows):
        rows = [one_hot("O") for _ in ids]
        if w == 0:
            rows[target - start + 1] = one_hot("B-NOME")
        logits.append(rows)
    (span,) = decode(enc, logits)
    assert span.start == target * 6


def test_predict_skips_blank_text() -> None:
    calls: list[list[int]] = []

    def logits(ids: list[int]) -> list[list[float]]:
        calls.append(ids)
        return [one_hot("O") for _ in ids]

    tok = WordPiece(small_vocab())
    assert predict(tok, logits, "   ") == []
    assert calls == []
    assert predict(tok, logits, "Maria") == []
    assert len(calls) == 1


def test_control_only_word_is_skipped() -> None:
    tok = WordPiece(small_vocab())
    enc = tok.encode("\x00 Maria")
    assert enc.first[0] == -1
    logits = [[one_hot("O"), one_hot("B-NOME"), one_hot("O")]]
    assert [s.text for s in decode(enc, logits)] == ["Maria"]
