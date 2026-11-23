"""Geradores de valores para os slots dos templates.

Cada função recebe um `Filler` (que guarda o `random.Random` e o Faker semeados) e devolve
o texto do valor. As entidades têm variantes de formato; os slots de preenchimento
(EMPRESA, PEDIDO, VALOR...) geram texto que *não* é PII e fica rotulado como `O`.
"""

from __future__ import annotations

import random
import string
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import TypeVar

from faker import Faker

from pii_br import validators as v

T = TypeVar("T")

# --------------------------------------------------------------------- listas

# Complementam o Faker pt_BR, cujos nomes são majoritariamente de origem europeia.
EXTRA_FIRST_NAMES = [
    "Iracema",
    "Jaciara",
    "Raoni",
    "Ubiratã",
    "Moema",
    "Iara",
    "Kauê",
    "Tainá",
    "Potira",
    "Jandira",
    "Araci",
    "Cauã",
    "Yara",
    "Kayode",
    "Zuri",
    "Dandara",
    "Abayomi",
    "Ayana",
    "Jamila",
    "Nala",
    "Akin",
    "Zumbi",
    "Malika",
    "Oluwa",
    "Ifeoma",
    "Amara",
    "Hiroshi",
    "Yuki",
    "Kenji",
    "Akemi",
    "Haruka",
    "Sakura",
    "Takeshi",
    "Mayumi",
    "Hiro",
    "Keiko",
    "Mohamed",
    "Fátima",
    "Yasmin",
    "Samir",
    "Nadia",
    "Karim",
    "Leila",
    "Omar",
    "Rania",
    "Tarik",
    "Wellington",
    "Jeferson",
    "Cleiton",
    "Sueli",
    "Rosângela",
    "Josefa",
    "Francisca",
    "Raimundo",
    "Edvaldo",
    "Gildásio",
    "Maria-Eduarda",
    "João-Pedro",
    "Ana-Clara",
    "Luiz-Felipe",
    "Kelly",
    "Jennifer",
    "Shirley",
    "Wesley",
    "Maicon",
    "Daiane",
]
EXTRA_LAST_NAMES = [
    "Guajajara",
    "Potiguara",
    "Tupinambá",
    "Pataxó",
    "Terena",
    "Xavante",
    "Krenak",
    "Yanomami",
    "Kaingang",
    "Nascimento",
    "Conceição",
    "Jesus",
    "Santana",
    "Bonfim",
    "Assunção",
    "Anunciação",
    "Paixão",
    "Sacramento",
    "Tanaka",
    "Suzuki",
    "Yamamoto",
    "Nakamura",
    "Kobayashi",
    "Sato",
    "Watanabe",
    "Ito",
    "Takahashi",
    "Haddad",
    "Khoury",
    "Nassar",
    "Salomão",
    "Jorge",
    "Calil",
    "Maluf",
    "Abdalla",
    "Schmidt",
    "Müller",
    "Becker",
    "Rossi",
    "Bianchi",
    "Colombo",
    "Kowalski",
    "Nowak",
    "Wolff",
]
CONNECTORS = ("da", "de", "do", "dos", "das")

STREET_TYPES = (
    ("Rua", 40), ("R.", 8), ("Avenida", 15), ("Av.", 10), ("Travessa", 5), ("Tv.", 2),
    ("Alameda", 4), ("Al.", 1), ("Praça", 3), ("Estrada", 3), ("Rodovia", 2), ("Largo", 1),
    ("Vila", 1), ("Beco", 1), ("Servidão", 1),
)  # fmt: skip
STREET_NAMES = (
    "das Flores", "dos Ipês", "das Palmeiras", "Sete de Setembro", "Quinze de Novembro",
    "XV de Novembro", "Tiradentes", "Dom Pedro II", "Santos Dumont", "Getúlio Vargas",
    "Marechal Deodoro", "Rui Barbosa", "Castro Alves", "Machado de Assis", "São João",
    "Santo Antônio", "São Sebastião", "Nossa Senhora Aparecida", "do Comércio", "da Paz",
    "Bela Vista", "Boa Esperança", "do Sol", "das Acácias", "dos Andradas", "José Bonifácio",
    "Princesa Isabel", "Duque de Caxias", "Barão do Rio Branco", "Visconde de Mauá",
    "Coronel Fonseca", "Padre Anchieta", "Monteiro Lobato", "Cecília Meireles", "Chico Mendes",
    "Zumbi dos Palmares", "Anita Garibaldi", "Maria Quitéria", "Ana Néri", "Carlos Gomes",
    "Vinte e Um de Abril", "Treze de Maio", "Brasil", "Amazonas", "Paraná", "Bahia",
    "dos Pinheiros", "das Mangueiras", "do Cajueiro", "Projetada A", "Um", "Dois", "Três",
)  # fmt: skip
NEIGHBORHOODS = (
    "Centro", "Jardim América", "Vila Nova", "Boa Vista", "Santa Cruz", "Liberdade",
    "Jardim das Flores", "Parque São Jorge", "Cidade Nova", "Bela Vista", "Industrial",
    "Vila Esperança", "Conjunto Habitacional", "Morada do Sol", "Planalto", "Santo Amaro",
)  # fmt: skip
CITIES = (
    ("São Paulo", "SP"), ("Rio de Janeiro", "RJ"), ("Belo Horizonte", "MG"), ("Salvador", "BA"),
    ("Fortaleza", "CE"), ("Recife", "PE"), ("Manaus", "AM"), ("Belém", "PA"), ("Curitiba", "PR"),
    ("Porto Alegre", "RS"), ("Goiânia", "GO"), ("Brasília", "DF"), ("Campinas", "SP"),
    ("São Luís", "MA"), ("Maceió", "AL"), ("Natal", "RN"), ("Teresina", "PI"),
    ("João Pessoa", "PB"), ("Aracaju", "SE"), ("Florianópolis", "SC"), ("Vitória", "ES"),
    ("Cuiabá", "MT"), ("Campo Grande", "MS"), ("Porto Velho", "RO"), ("Rio Branco", "AC"),
    ("Macapá", "AP"), ("Boa Vista", "RR"), ("Palmas", "TO"), ("Uberlândia", "MG"),
    ("Joinville", "SC"), ("Londrina", "PR"), ("Feira de Santana", "BA"), ("Juiz de Fora", "MG"),
    ("Caruaru", "PE"), ("Petrolina", "PE"), ("Santarém", "PA"), ("Sobral", "CE"),
)  # fmt: skip
FAMOUS_STREETS = (
    "Av. Paulista", "Avenida Paulista", "Rua Augusta", "Av. Atlântica", "Rua Oscar Freire",
    "Avenida Brasil", "Av. Rio Branco", "Rua 25 de Março", "Av. Faria Lima", "Marginal Tietê",
    "Linha Vermelha", "Av. Sete de Setembro", "Rua da Praia", "Av. Afonso Pena", "Eixo Monumental",
)  # fmt: skip
PRODUCTS = (
    "notebook", "geladeira", "celular", "fone de ouvido", "cadeira gamer", "kit de panelas",
    "smart TV", "micro-ondas", "tênis de corrida", "cafeteira", "plano fibra 500 mega",
    "seguro auto", "cartão de crédito", "assinatura premium", "máquina de lavar", "monitor",
)  # fmt: skip
EMAIL_DOMAINS = (
    "gmail.com", "hotmail.com", "outlook.com", "yahoo.com.br", "uol.com.br", "bol.com.br",
    "icloud.com", "terra.com.br", "live.com", "globo.com", "empresa.com.br", "example.com",
    "prefeitura.sp.gov.br", "usp.br", "loja.com.br", "consultoria.net",
)  # fmt: skip
MONTHS = (
    "janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto", "setembro",
    "outubro", "novembro", "dezembro",
)  # fmt: skip
EMOJIS = ("😊", "🙏", "👍", "😅", "😡", "🙂", "❤️", "😂", "👀", "✅")


def strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    ).replace("̧", "")


@dataclass
class Filler:
    """Gera valores de slot com um `random.Random` e um Faker semeados juntos."""

    rng: random.Random
    faker: Faker = field(init=False)
    invalid_dv_rate: float = 0.10

    def __post_init__(self) -> None:
        self.faker = Faker("pt_BR")
        self.faker.seed_instance(self.rng.randrange(2**32))

    # ------------------------------------------------------------- utilitários
    def chance(self, p: float) -> bool:
        return self.rng.random() < p

    def pick(self, options: Sequence[T]) -> T:
        return self.rng.choice(options)

    def weighted(self, options: tuple[tuple[str, int], ...]) -> str:
        return self.rng.choices([o for o, _ in options], weights=[w for _, w in options])[0]

    # ---------------------------------------------------------------- entidades
    def first_name(self) -> str:
        if self.chance(0.2):
            return self.pick(EXTRA_FIRST_NAMES).replace("-", " ")
        return str(self.faker.first_name())

    def last_name(self) -> str:
        if self.chance(0.2):
            return self.pick(EXTRA_LAST_NAMES)
        return str(self.faker.last_name())

    def nome(self) -> str:
        """Nome completo (1 a 3 sobrenomes, às vezes com conector) ou só o primeiro nome."""
        first = self.first_name()
        roll = self.rng.random()
        if roll < 0.12:
            return first
        if roll < 0.2 and " " not in first:
            first = f"{first} {self.first_name()}"
        parts = [first]
        for _ in range(self.rng.choice((1, 1, 2, 2, 3))):
            if self.chance(0.3):
                parts.append(self.pick(CONNECTORS))
            parts.append(self.last_name())
        return " ".join(parts)

    def cpf(self) -> str:
        valid = not self.chance(self.invalid_dv_rate)
        digits = v.generate_cpf(self.rng, formatted=False, valid=valid)
        style = self.rng.random()
        if style < 0.6:
            return v.format_cpf(digits)
        if style < 0.85:
            return digits
        if style < 0.95:
            return f"{digits[:3]} {digits[3:6]} {digits[6:9]} {digits[9:]}"
        return f"{digits[:9]}-{digits[9:]}"

    def cnpj(self) -> str:
        valid = not self.chance(self.invalid_dv_rate)
        alnum = self.chance(0.3)
        value = v.generate_cnpj(self.rng, alphanumeric=alnum, formatted=False, valid=valid)
        if alnum and self.chance(0.1):
            value = value.lower()
        return v.format_cnpj(value) if self.chance(0.7) else value

    def rg(self) -> str:
        digits = "".join(self.pick(string.digits) for _ in range(8))
        dv = self.pick(string.digits + "X")
        style = self.rng.random()
        if style < 0.45:
            return f"{digits[:2]}.{digits[2:5]}.{digits[5:]}-{dv}"
        if style < 0.7:
            return digits + dv
        if style < 0.85:
            uf = self.pick(("SP", "MG", "RJ", "BA", "PR", "RS", "PE", "CE"))
            return f"{uf}-{digits[:2]}.{digits[2:5]}.{digits[5:]}"
        return f"{digits[1:2]}.{digits[2:5]}.{digits[5:]}"  # formato curto de alguns estados

    def telefone(self) -> str:
        ddd = self.pick(sorted(v.VALID_DDDS))
        mobile = self.chance(0.75)
        if mobile:
            number = "9" + "".join(self.pick(string.digits) for _ in range(8))
            head, tail = number[:5], number[5:]
        else:
            number = self.pick("2345") + "".join(self.pick(string.digits) for _ in range(7))
            head, tail = number[:4], number[4:]
        style = self.rng.random()
        if style < 0.35:
            return f"({ddd}) {head}-{tail}"
        if style < 0.5:
            return f"{ddd} {head}-{tail}"
        if style < 0.62:
            return f"+55 {ddd} {head}-{tail}"
        if style < 0.74:
            return f"{ddd}{number}"
        if style < 0.82:
            return f"({ddd}){head}-{tail}"
        if style < 0.88:
            return f"+55{ddd}{number}"
        if style < 0.93 and mobile:
            return f"{ddd} 9 {number[1:5]}-{tail}"
        return f"{head}-{tail}"  # sem DDD

    def email(self, name_hint: str | None = None) -> str:
        if name_hint and self.chance(0.6):
            parts = strip_accents(name_hint).lower().split()
            parts = [p for p in parts if p not in CONNECTORS]
            sep = self.pick((".", "_", "", "-"))
            local = sep.join(parts[:2]) if len(parts) > 1 else parts[0]
            if self.chance(0.4):
                local += str(self.rng.randint(1, 99))
        else:
            local = str(self.faker.user_name())
        address = f"{local}@{self.pick(EMAIL_DOMAINS)}"
        return address.upper() if self.chance(0.05) else address

    def endereco(self) -> str:
        street = f"{self.weighted(STREET_TYPES)} {self.pick(STREET_NAMES)}"
        number = str(self.rng.randint(1, 4999)) if not self.chance(0.05) else "s/n"
        sep = self.pick((", ", ", nº ", " ", ", n. "))
        parts = [f"{street}{sep}{number}"]
        if self.chance(0.4):
            parts.append(
                self.pick(("ap. ", "apto ", "apartamento ", "casa ", "bloco ", "sala ", "fundos"))
            )
            if not parts[-1].endswith("fundos"):
                parts[-1] += str(self.rng.randint(1, 1500))
        address = ", ".join(parts)
        if self.chance(0.35):
            address += f" - {self.pick(NEIGHBORHOODS)}"
        return address

    def cep(self) -> str:
        digits = f"{self.rng.randint(1_000_000, 99_999_999):08d}"
        style = self.rng.random()
        if style < 0.7:
            return f"{digits[:5]}-{digits[5:]}"
        if style < 0.9:
            return digits
        return f"{digits[:2]}.{digits[2:5]}-{digits[5:]}"

    def any_date(self, start: date, end: date) -> str:
        d = start + timedelta(days=self.rng.randint(0, (end - start).days))
        style = self.rng.random()
        if style < 0.55:
            return d.strftime("%d/%m/%Y")
        if style < 0.65:
            return f"{d.day}/{d.month}/{d.year}"
        if style < 0.75:
            return d.strftime("%d/%m/%y")
        if style < 0.83:
            return d.strftime("%d-%m-%Y")
        if style < 0.9:
            return d.strftime("%d.%m.%Y")
        return f"{d.day} de {MONTHS[d.month - 1]} de {d.year}"

    def data_nasc(self) -> str:
        return self.any_date(date(1935, 1, 1), date(2010, 12, 31))

    # ------------------------------------------------- preenchimentos (rótulo O)
    def empresa(self) -> str:
        return str(self.faker.company())

    def pedido(self) -> str:
        """Números de pedido: inclui 11 e 14 dígitos corridos, negativos difíceis para CPF/CNPJ."""
        n = self.pick((6, 8, 10, 11, 11, 14, 14))
        digits = "".join(self.pick(string.digits) for _ in range(n))
        if n == 11 and v.is_valid_cpf(digits):
            digits = digits[:-1] + str((int(digits[-1]) + 1) % 10)
        if n == 14 and v.is_valid_cnpj(digits):
            digits = digits[:-1] + str((int(digits[-1]) + 1) % 10)
        return f"#{digits}" if self.chance(0.2) else digits

    def protocolo(self) -> str:
        year = self.rng.randint(2019, 2026)
        tail = "".join(self.pick(string.digits) for _ in range(self.pick((6, 8, 10))))
        return self.pick((f"{year}{tail}", f"{year}/{tail}", f"{year}-{tail}", tail))

    def codigo(self) -> str:
        """Código de barras, SKU ou hash curto."""
        roll = self.rng.random()
        if roll < 0.4:
            return "789" + "".join(self.pick(string.digits) for _ in range(10))  # EAN-13
        if roll < 0.7:
            letters = "".join(self.pick(string.ascii_uppercase) for _ in range(3))
            return f"{letters}-{self.rng.randint(1000, 99999)}"
        return "".join(self.pick("0123456789abcdef") for _ in range(12))

    def valor(self) -> str:
        reais = self.rng.randint(5, 25000)
        cents = self.rng.randint(0, 99)
        formatted = f"{reais:,}".replace(",", ".")
        return f"R$ {formatted},{cents:02d}"

    def data(self) -> str:
        return self.any_date(date(2022, 1, 1), date(2026, 12, 31))

    def hora(self) -> str:
        h, m = self.rng.randint(0, 23), self.rng.randint(0, 59)
        return self.pick((f"{h:02d}:{m:02d}", f"{h}h{m:02d}", f"{h}h"))

    def cidade(self) -> str:
        city, uf = self.pick(CITIES)
        return self.pick((city, f"{city}/{uf}", f"{city} - {uf}"))

    def rua_famosa(self) -> str:
        return self.pick(FAMOUS_STREETS)

    def produto(self) -> str:
        return self.pick(PRODUCTS)

    def tel_empresa(self) -> str:
        roll = self.rng.random()
        rest = "".join(self.pick(string.digits) for _ in range(7))
        if roll < 0.6:
            return self.pick(("0800 ", "0800-", "0300 ")) + f"{rest[:3]} {rest[3:]}"
        return f"{self.pick(('4004', '4003', '3003', '4020'))}-{rest[:4]}"

    def num(self) -> str:
        return str(self.rng.randint(1, 500))

    def timestamp(self) -> str:
        d = date(2026, 1, 1) + timedelta(days=self.rng.randint(0, 320))
        h, m, s = self.rng.randint(0, 23), self.rng.randint(0, 59), self.rng.randint(0, 59)
        return self.pick(
            (
                f"{d.isoformat()}T{h:02d}:{m:02d}:{s:02d}Z",
                f"{d.isoformat()} {h:02d}:{m:02d}:{s:02d}",
            )
        )

    def uuid(self) -> str:
        hexd = "".join(self.pick("0123456789abcdef") for _ in range(32))
        return f"{hexd[:8]}-{hexd[8:12]}-{hexd[12:16]}-{hexd[16:20]}-{hexd[20:]}"


ENTITY_SLOTS: dict[str, Callable[[Filler], str]] = {
    "NOME": Filler.nome,
    "CPF": Filler.cpf,
    "CNPJ": Filler.cnpj,
    "RG": Filler.rg,
    "TELEFONE": Filler.telefone,
    "EMAIL": Filler.email,
    "ENDERECO": Filler.endereco,
    "CEP": Filler.cep,
    "DATA_NASC": Filler.data_nasc,
}
PLAIN_SLOTS: dict[str, Callable[[Filler], str]] = {
    "EMPRESA": Filler.empresa,
    "PEDIDO": Filler.pedido,
    "PROTOCOLO": Filler.protocolo,
    "CODIGO": Filler.codigo,
    "VALOR": Filler.valor,
    "DATA": Filler.data,
    "HORA": Filler.hora,
    "CIDADE": Filler.cidade,
    "RUA_FAMOSA": Filler.rua_famosa,
    "PRODUTO": Filler.produto,
    "TEL_EMPRESA": Filler.tel_empresa,
    "NUM": Filler.num,
    "TIMESTAMP": Filler.timestamp,
    "UUID": Filler.uuid,
}
