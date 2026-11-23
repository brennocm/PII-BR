---
language:
  - pt
license: cc-by-4.0
pretty_name: PII-BR
task_categories:
  - token-classification
task_ids:
  - named-entity-recognition
tags:
  - pii
  - lgpd
  - privacy
  - anonymization
  - synthetic
  - brazilian-portuguese
size_categories:
  - 10K<n<100K
---

# PII-BR — dataset sintético de PII em português do Brasil

Frases sintéticas em português do Brasil anotadas com 9 tipos de dado pessoal (PII), para treinar e avaliar detectores de PII voltados à LGPD. **Nenhum dado é de pessoa real**: nomes, documentos, telefones e endereços são gerados por templates, pelo Faker `pt_BR` e por geradores próprios.

*Synthetic Brazilian Portuguese sentences annotated with 9 PII entity types (names, CPF, CNPJ including the new alphanumeric format, RG, phone, e-mail, address, postal code and birth date). No real personal data.*

- Modelo treinado com este dataset: [{{MODEL_REPO}}](https://huggingface.co/{{MODEL_REPO}})
- Demo: [{{SPACE_REPO}}](https://huggingface.co/spaces/{{SPACE_REPO}})
- Código (gerador, treino, avaliação): [GitHub]({{GITHUB_URL}})

## Entidades

| Rótulo | Exemplo | Observação |
| --- | --- | --- |
| `NOME` | Maria Aparecida Souza | Títulos (Sr., Dra.) ficam fora do span |
| `CPF` | 123.456.789-09 | Com e sem pontuação; ~10% com dígito verificador errado, **rotulados como CPF** |
| `CNPJ` | 12.ABC.345/01DE-35 | ~30% no formato alfanumérico (IN RFB nº 2.229/2024) |
| `RG` | 12.345.678-9 | Formatos de vários estados |
| `TELEFONE` | (11) 98765-4321 | Com e sem DDD, +55; 0800 e 4004 são `O` |
| `EMAIL` | maria.souza@example.com | |
| `ENDERECO` | Rua das Flores, 123, ap. 42 | Logradouro, número, complemento e bairro |
| `CEP` | 01310-100 | |
| `DATA_NASC` | 03/07/1990 | Só datas de nascimento; outras datas são `O` |

Rótulos BIO (`ner_tags`): {{LABELS}}.

## Splits

| Split | Frases | Como foi feito |
| --- | --- | --- |
{{SPLIT_TABLE}}

O `test_hard` usa 20% dos templates, sorteados por domínio e nunca vistos no treino: mede generalização para estruturas de frase novas. O `test_real` tem frases escritas à mão fora do sistema de templates, revisadas por uma pessoa, e é o número mais próximo do uso real.

### Entidades por split

{{ENTITY_TABLE}}

## Campos

| Campo | Tipo | Descrição |
| --- | --- | --- |
| `id` | string | Identificador da frase |
| `text` | string | Texto original |
| `tokens` | list[string] | Palavras (regra abaixo) |
| `ner_tags` | list[ClassLabel] | Rótulo BIO de cada palavra |
| `spans` | list[{start, end, label}] | Entidades em offsets de caractere sobre `text` (fim exclusivo), para avaliação independente de tokenizador |
| `domain` | string | atendimento, whatsapp, email, contrato, cadastro, ticket, log, ocorrencia (e outros, no test_real) |
| `template_id` | string | Template de origem (`real` no test_real) |
| `noisy` | bool | Se a frase recebeu ruído (erros de digitação, sem acento, caixa baixa, emoji, quebra de linha) |
| `has_invalid_dv` | bool | Se algum CPF/CNPJ da frase tem dígito verificador errado |

**Tokenização:** uma palavra é uma sequência de letras/dígitos que pode conter `.`, `-`, `/`, `@`, `+` ou apóstrofo entre caracteres (então `123.456.789-09` e `maria@example.com` são um token); qualquer outro símbolo vira um token. Implementação em `pii_br.words`.

## Como foi gerado

1. {{TEMPLATES_TOTAL}} templates escritos à mão em 8 domínios, com slots de entidade (`{NOME}`, `{CPF}`...) e slots de preenchimento que **não** são PII (`{PEDIDO}`, `{PROTOCOLO}`, `{EMPRESA}`, `{VALOR}`, datas que não são de nascimento, ruas famosas fora de contexto pessoal).
2. Valores do Faker `pt_BR` ({{FAKER_VERSION}}) e de geradores próprios: CPF/CNPJ com dígito verificador real, CNPJ alfanumérico, telefones com DDD válido, nomes complementados por listas de origem indígena, africana, japonesa, árabe e de grafia popular brasileira.
3. Variações de formato em cada entidade e ruído em cerca de 30% das frases.
4. Negativos difíceis: números de pedido com 11 e 14 dígitos (que não fecham o dígito de CPF/CNPJ), códigos de barras, protocolos, telefones 0800.
5. Cerca de 15% das frases sem nenhuma PII.

A anotação é feita por construção (sem anotação manual) e a geração é reprodutível: `python -m data_gen.generate --seed {{SEED}}`.

## Limitações

- **Texto sintético.** As frases seguem templates e são menos variadas que texto real; o `test_real` existe para medir essa diferença.
- **Viés de nomes.** O Faker favorece nomes de origem europeia; as listas extras reduzem, mas não eliminam o viés.
- **RG** não tem formato nacional; os formatos cobertos são aproximações.
- **Endereços** são de ruas plausíveis, não de um cadastro real de logradouros.
- **Fora do escopo:** dados sensíveis do art. 5º, II da LGPD (saúde, religião etc.), placas, cartões, PIS, CNH.

## Licença

CC-BY-4.0. Os valores são fictícios; qualquer coincidência com dados reais é acidental.
