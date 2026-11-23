---
language:
  - pt
license: mit
library_name: transformers
pipeline_tag: token-classification
base_model: neuralmind/bert-base-portuguese-cased
datasets:
  - {{DATASET_REPO}}
tags:
  - pii
  - lgpd
  - privacy
  - anonymization
  - ner
  - brazilian-portuguese
---

# PII-BR — detector de dados pessoais em português do Brasil

Modelo de reconhecimento de entidades (NER) que encontra 9 tipos de dado pessoal em texto em português do Brasil, para anonimizar logs, tickets, e-mails e documentos sob a LGPD. É um fine-tuning do [BERTimbau base](https://huggingface.co/neuralmind/bert-base-portuguese-cased) no dataset sintético [PII-BR](https://huggingface.co/datasets/{{DATASET_REPO}}).

O melhor uso é no **pipeline híbrido** do pacote [`pii_br`]({{GITHUB_URL}}): o modelo cobre o que depende de contexto (nomes, endereços, RG, data de nascimento), e regras determinísticas com dígito verificador cobrem CPF, CNPJ (inclusive o alfanumérico), telefone, e-mail e CEP.

- Demo: [{{SPACE_REPO}}](https://huggingface.co/spaces/{{SPACE_REPO}})
- Dataset: [{{DATASET_REPO}}](https://huggingface.co/datasets/{{DATASET_REPO}})
- Código e notebook de treino: [GitHub]({{GITHUB_URL}})

## Entidades

`NOME`, `CPF`, `CNPJ`, `RG`, `TELEFONE`, `EMAIL`, `ENDERECO`, `CEP`, `DATA_NASC` (rótulos BIO).

## Como usar

```python
from transformers import pipeline

ner = pipeline(
    "token-classification",
    model="{{MODEL_REPO}}",
    aggregation_strategy="first",
    stride=64,  # textos longos em janelas sobrepostas
)
ner("Sou Maria Souza, CPF 123.456.789-09, moro na Rua das Flores, 123.")
```

Pipeline híbrido com mascaramento (recomendado):

```python
from pii_br import Detector, Masker
from pii_br.model import NerModel

detector = Detector(NerModel("{{MODEL_REPO}}"))
texto = "Sou Maria Souza, CPF 123.456.789-09, whats (11) 98765-4321."
spans = detector.detect(texto)
print(Masker().mask(texto, spans, "redact"))
# Sou [NOME], CPF [CPF], whats [TELEFONE].
```

## Resultados

Métricas por span em caracteres. **Sobreposição** conta acerto quando o rótulo é o mesmo e o trecho se sobrepõe ao de referência (o que importa para mascarar); **exato** exige as mesmas bordas. Limiar de confiança: {{THRESHOLD}}.

- `test`: mesmos templates do treino, valores novos (o número mais otimista).
- `test_hard`: templates nunca vistos no treino.
- `test_real`: frases escritas à mão, fora do sistema de templates (o mais próximo do uso real).

{{RESULTS}}

Baselines: **só regex** é a parte determinística do pacote, sem modelo; **Presidio** usa os recognizers padrão do `presidio-analyzer` registrados para português, com spaCy `pt_core_news_lg` (`PERSON`→`NOME`, `LOCATION`→`ENDERECO`, `PHONE_NUMBER`→`TELEFONE`, `EMAIL_ADDRESS`→`EMAIL`). O Presidio não tem recognizers para CPF, CNPJ, RG ou CEP, então essas entidades contam como não detectadas.

{{SEQEVAL}}

**Latência** do pipeline híbrido (modelo int8 dinâmico, CPU, texto de {{LATENCY_CHARS}} caracteres): mediana de {{LATENCY}} s.

## Erros mais comuns

Falsos negativos primeiro, porque deixar passar um dado pessoal é o erro caro. Amostra de `{{ERRORS_SPLIT}}`:

{{ERRORS}}

## Treino

- Base: `neuralmind/bert-base-portuguese-cased`; rótulo no primeiro subtoken de cada palavra.
- `lr=5e-5`, 4 épocas com early stopping pelo F1 de entidade na validação, batch 16, `max_length=256`, warmup 10%, weight decay 0,01, fp16, seed 42.
- GPU gratuita (Kaggle T4). Notebook em `training/train.ipynb` no repositório.

## Limitações

- **Treinado só com dados sintéticos.** O desempenho em texto real tende a ser menor que no `test`; o `test_real` é pequeno (~200 frases) e serve de sinal, não de garantia.
- Nomes sem contexto ("Jorge chegou") e nomes que também são palavras comuns são os casos mais difíceis.
- RG não tem formato nacional; o modelo depende do contexto ("RG", "identidade").
- Não detecta dados sensíveis do art. 5º, II da LGPD (saúde, religião, orientação sexual), nem placas, cartões, PIS ou CNH.
- **Não é garantia de anonimização.** Pela LGPD (art. 12), dado anonimizado é o que não pode ser revertido por meios razoáveis; mascarar entidades não remove quasi-identificadores (cargo, bairro, datas). Use como redutor de risco, com revisão humana onde importa.

## Licença

MIT, a mesma do modelo base.
