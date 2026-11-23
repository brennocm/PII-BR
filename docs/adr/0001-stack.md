# 0001. Stack: Python 3.11, uv, pacote único em layout src

- Status: accepted
- Date: 2026-11-23

## Context

O projeto tem quatro partes (gerador de dataset, treino, avaliação e Space) em torno de um pacote de regras reutilizável. Treino e inferência usam o ecossistema Hugging Face (`transformers`, `datasets`, `gradio`), que é Python. Uma pessoa mantém o projeto com ajuda de um agente de código. O treino roda no Kaggle/Colab e o Space em CPU gratuita (ver `SPEC.md`, seção de viabilidade).

## Decision

- Python 3.11, fixado em `.python-version` e `.mise.toml`, para casar com o Kaggle e o Space.
- `uv` para ambiente, lockfile e build (`uv_build`); `mise` para versões de ferramentas e comandos padrão.
- Um único pacote, `pii_br`, em layout `src/`, sem workspaces: as demais partes (`data_gen/`, `training/`, `space/`) são scripts que importam o pacote.
- O núcleo do pacote só usa a biblioteca padrão; dependências pesadas (`torch`, `transformers`, `presidio-analyzer`) entram como extras opcionais.
- Ruff (formatação e lint), pyright estrito e pytest; cobertura mínima de 90% no pacote.

## Alternatives considered

- **Workspace uv com `apps/` e `packages/`** (padrão de monorepo): exagero para um pacote e três pastas de scripts; atrasaria o projeto sem benefício.
- **Pacote na raiz (`pii_br/`, como na v1 da spec)**: os testes poderiam importar o diretório de trabalho em vez do pacote instalado e esconder erro de empacotamento.
- **Python 3.12** (o do sistema local): o Kaggle e o template padrão dos Spaces ainda usam 3.10/3.11.

## Consequences

- O Space recebe uma cópia de `src/pii_br/` no deploy, sem precisar publicar no PyPI.
- Se o pacote for publicado no PyPI, o layout já está pronto.
- Rever a decisão de workspaces se surgir um segundo pacote com dependências próprias.
