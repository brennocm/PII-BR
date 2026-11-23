"""Critério de aceite: os validadores batem 100% com fontes independentes do nosso código.

Comparar o validador com o nosso próprio gerador seria circular, então a referência é:
- `validate-docbr` (biblioteca de terceiros) para CPF e CNPJ numéricos;
- os geradores do Faker pt_BR, que produzem documentos válidos por conta própria;
- o exemplo oficial da Receita para o CNPJ alfanumérico (em test_validators.py).
"""

import random

from faker import Faker
from validate_docbr import CNPJ, CPF

from pii_br import validators as v

N = 600  # por fonte de candidatos; a suíte total passa de 1.000 por documento


def _random_digits(rng: random.Random, n: int) -> str:
    return "".join(rng.choice("0123456789") for _ in range(n))


def _mutate(rng: random.Random, value: str) -> str:
    """Troca um dígito: quase sempre invalida o documento."""
    positions = [i for i, c in enumerate(value) if c.isdigit()]
    i = rng.choice(positions)
    return value[:i] + str((int(value[i]) + rng.randint(1, 9)) % 10) + value[i + 1 :]


def test_cpf_matches_validate_docbr() -> None:
    rng = random.Random(10)
    oracle = CPF()
    candidates: list[str] = []
    for _ in range(N):
        valid = v.generate_cpf(rng, formatted=rng.random() < 0.5)
        candidates += [valid, _mutate(rng, valid), _random_digits(rng, 11)]
    candidates += [str(d) * 11 for d in range(10)]
    disagreements = [c for c in candidates if v.is_valid_cpf(c) != oracle.validate(c)]
    assert len(candidates) >= 1000
    assert disagreements == []
    assert sum(v.is_valid_cpf(c) for c in candidates) >= N  # a suíte tem válidos e inválidos


def test_cnpj_numeric_matches_validate_docbr() -> None:
    rng = random.Random(11)
    oracle = CNPJ()
    candidates: list[str] = []
    for _ in range(N):
        valid = v.generate_cnpj(rng, formatted=rng.random() < 0.5)
        candidates += [valid, _mutate(rng, valid), _random_digits(rng, 14)]
    candidates += [str(d) * 14 for d in range(10)]
    disagreements = [c for c in candidates if v.is_valid_cnpj(c) != oracle.validate(c)]
    assert len(candidates) >= 1000
    assert disagreements == []


def test_faker_documents_are_valid() -> None:
    fake = Faker("pt_BR")
    fake.seed_instance(12)
    for _ in range(500):
        assert v.is_valid_cpf(fake.cpf())
        assert v.is_valid_cnpj(fake.cnpj())
