# Publicação do PII-BR

Passo a passo para colocar no ar as quatro peças do projeto: repositório no GitHub, dataset, modelo e Space no Hugging Face. Tudo é gratuito. Nenhum valor de segredo aparece aqui; só os nomes.

## 1. Situação

**Pronto e verificado localmente**

| Item | Evidência |
| --- | --- |
| Pacote `pii_br` (validadores, recognizers, fusão, mascaramento, modelo) | `mise run check:all`: testes, cobertura ≥ 90%, pyright estrito, ruff, pip-audit (inclusive o torch), gitleaks |
| Validadores conferidos contra fontes independentes | `tests/test_oracle.py` (validate-docbr, Faker, exemplo oficial da Receita) |
| Dataset (38.000 frases + 213 do `test_real`), reprodutível por seed | `uv run python -m data_gen.generate --out data`; hash igual entre processos |
| Card do dataset e montagem do `DatasetDict` | `python -m data_gen.push_dataset --repo x/PII-BR --dry-run` |
| Notebook de treino | rodou de ponta a ponta em CPU com `SMOKE=1` |
| Avaliação com baselines (regex e Presidio) | `python -m training.evaluate --data data` |
| App do Space montado isoladamente | `python scripts/deploy_space.py --repo x/PII-BR --dry-run` + import do app na pasta gerada |
| Pipeline com um modelo de verdade | Modelo pequeno treinado em CPU (`outputs/mini`): `tests/test_model.py` roda o híbrido num texto de 1.400 caracteres |
| Versões do Space e do notebook iguais às do `uv.lock` | `tests/test_pins.py` |

**Pronto, mas só verificável depois da publicação**

- Download do dataset pelo notebook no Kaggle, treino em GPU e `push_to_hub` do modelo.
- Workflow `deploy-space` no GitHub Actions e o Space rodando no Hugging Face.
- Latência no hardware do Space (medida local: ~0,3 s para 1.000 caracteres com int8).

**Depende de você** (passos abaixo): contas, tokens, revisão do `test_real`, rodar o treino, criar os secrets.

## 2. Contas e custos

| Serviço | Para quê | Plano | Custo mensal | Link |
| --- | --- | --- | --- | --- |
| GitHub | Código, CI e deploy do Space | Free | R$ 0 | github.com |
| Hugging Face | Dataset, modelo e Space (estático) | Free | R$ 0 | huggingface.co |
| Kaggle | GPU para o treino (~30 min de 30 h/semana) | Free | R$ 0 | kaggle.com |

Total: **R$ 0**. Nada aqui pede cartão de crédito. Spaces Gradio e Docker agora exigem o plano PRO; o Space deste projeto é estático justamente para continuar grátis (ADR 0003).

## 3. Segredos e configuração

| Nome | Para quê | Onde obter | Onde configurar | Necessário para |
| --- | --- | --- | --- | --- |
| Token HF "pii-br-publicar" (escrita) | Publicar dataset e card do modelo a partir da sua máquina | huggingface.co → Settings → Access Tokens | Login local: `! uv run hf auth login` (o token fica no cache do HF, fora do repositório) | Passo 4 |
| `HF_TOKEN` no Kaggle (escrita) | O notebook publica o modelo | Mesmo token acima, ou um próprio | Notebook no Kaggle → Add-ons → Secrets | Passo 5 |
| `HF_TOKEN` no GitHub (escrita só no Space) | A Action publica o Space | Token fine-grained com escrita apenas no Space `PII-BR` | `! gh secret set HF_TOKEN` (pede o valor sem eco) | Passo 7 |
| `HF_SPACE` no GitHub (variável, não segredo) | Qual Space publicar | `brennocm/PII-BR` | `! gh variable set HF_SPACE --body "brennocm/PII-BR"` | Passo 7 |

Regras: nunca cole um token no chat, num commit ou num issue. Um token por finalidade. Se algum vazar, revogue em Settings → Access Tokens antes de qualquer outra coisa.

## 4. Passo a passo

### Passo 0 — Revisões (≈ 1 h 30)

1. **Regra do CNPJ alfanumérico.** Leia o PDF de perguntas e respostas da Receita sobre o CNPJ alfanumérico e confira o que o validador assume: letras A–Z nas 12 primeiras posições, valor `ASCII − 48`, mesmos pesos e dígitos verificadores numéricos. O exemplo oficial `12.ABC.345/01DE-35` já está nos testes. Se a norma excluir alguma letra, me avise.
2. **`test_real`.** Abra `data_gen/test_real.tsv` (213 frases, uma por linha: `domínio<TAB>texto com [valor](RÓTULO)`). Corrija rótulos, apague frases artificiais e reescreva umas 30 com o seu jeito de escrever. Convenção: títulos como "Sr." e "Dra." ficam fora do span de `NOME`. Depois da revisão, volte a frase "revisadas por uma pessoa" ao `test_real` em `cards/dataset.md` (ela só pode estar lá quando for verdade).
3. **Amostra do dataset.** Rode `uv run python -m data_gen.generate --out data` e leia umas 100 linhas de `data/train.jsonl` (ou peça que eu mostre uma amostra formatada).

**Verificar:** `mise run check` continua verde depois das edições.

### Passo 1 — GitHub (≈ 10 min)

1. Login: `! gh auth login` (GitHub.com → HTTPS → navegador).
2. Criar e enviar: `! gh repo create PII-BR --public --source . --push`
3. **Esperado:** repositório público com a branch `main` e o workflow `ci` rodando em Actions.
4. **Verificar:** `gh run list --limit 1` mostra `ci` com `completed success`.
5. **Se falhar:** abra o log com `gh run view --log-failed`; o mais provável é versão de ferramenta, e eu ajusto.

6. **Renovate:** instale o app do Renovate em github.com/apps/renovate e dê acesso só ao `PII-BR`; ele lê o `renovate.json` e abre PRs semanais com 14 dias de quarentena para versões novas.

### Passo 2 — Hugging Face: conta e token (≈ 10 min)

1. Crie a conta em huggingface.co e ative 2FA em Settings → Authentication.
2. Settings → Access Tokens → *Create new token* → tipo **Write**, nome `pii-br-publicar`.
3. Login local: `! uv run hf auth login` e cole o token quando ele pedir (não aparece na tela).
4. **Verificar:** `uv run hf auth whoami` mostra o seu usuário.

### Passo 4 — Publicar o dataset (≈ 5 min)

```bash
uv run python -m data_gen.generate --out data
uv run --group hub python -m data_gen.push_dataset --repo brennocm/PII-BR \
  --github https://github.com/brennocm/PII-BR
```

- **Esperado:** `Publicado: https://huggingface.co/datasets/brennocm/PII-BR`, com 5 splits e o card.
- **Verificar:** a página do dataset mostra o *Dataset Viewer* com `train`, `validation`, `test`, `test_hard` e `test_real`.
- **Se falhar com 401/403:** o token não é de escrita; gere outro e refaça o login.

### Passo 5 — Treinar no Kaggle (≈ 10 min seu + 30 min de máquina)

1. Crie a conta no kaggle.com e verifique o telefone (Settings → Phone verification). Sem isso não há GPU nem internet no notebook.
2. *Create → New Notebook → File → Import Notebook* → envie `training/train.ipynb`.
3. Painel direito: *Session options → Accelerator: GPU T4 x2* (ou P100) e *Internet on*.
4. *Add-ons → Secrets → Add secret*: nome `HF_TOKEN`, valor = o token de escrita; marque para este notebook.
5. A primeira célula de código já usa `HF_USER = "brennocm"`; não precisa mudar.
6. *Save Version → Save & Run All (Commit)*. Pode fechar a aba.
7. **Esperado:** em ~30 min a versão termina com `Modelo publicado: https://huggingface.co/brennocm/PII-BR`.
8. **Verificar:** a página do modelo tem `model.safetensors` e `metrics_seqeval.json`.
9. **Se falhar:** abra o log da versão. Erro de `kaggle_secrets` = secret não marcado para o notebook; erro de download = internet desligada; CUDA out of memory = troque `BATCH = 16` por `8`. Se a sessão cair, rode de novo: o notebook retoma do último checkpoint.

### Passo 6 — Avaliar e publicar o model card (≈ 15 min, roda na sua máquina)

```bash
uv sync --all-extras --group baseline --group hub
uv run python -m training.evaluate --model brennocm/PII-BR --data data --write-card
uv run hf upload brennocm/PII-BR outputs/model_card/README.md README.md
```

- **Esperado:** tabelas híbrido × só modelo × só regex × Presidio nos três splits, latência e erros no card.
- **Verificar:** a página do modelo mostra o card novo. Me mande `outputs/eval/report.md` (não tem dado real) e eu atualizo a tabela de resultados do README.

### Passo 7 — Space (≈ 10 min)

O Space é **estático** (ADR 0003): o Hugging Face passou a cobrar por Spaces Gradio, e a página roda o pacote e o modelo no navegador. Já foi criado e publicado a partir desta máquina (`spaces/brennocm/PII-BR`, commit do modelo fixado em `scripts/deploy_space.py`). Falta só o deploy automático:

1. Token só para o Space: Settings → Access Tokens → *Fine-grained* → em *Repositories permissions*, escolha o Space `brennocm/PII-BR` com **Write**. Nome: `pii-br-space-deploy`.
2. No terminal:
   ```
   ! gh auth login
   ! gh secret set HF_TOKEN
   ! gh variable set HF_SPACE --body "brennocm/PII-BR"
   ! gh workflow run deploy-space
   ```
3. **Esperado:** o workflow `deploy-space` termina verde em ~1 min (não há build no Space estático).
4. **Verificar:** abra o Space e confira:
   - o status diz "Modelo e regras prontos" depois do download (~15 s na primeira visita, depois do cache);
   - o exemplo com CPF inválido mostra o alerta;
   - o exemplo sem PII não marca nada;
   - um texto de ~1.000 caracteres responde em menos de 3 s.
5. **Se falhar:** abra o console do navegador (F12). "Modelo indisponível" = o arquivo `onnx/model_int8.onnx` não existe no commit fixado.

A partir daqui, todo push na `main` com CI verde republica o Space sozinho.

**Modelo novo:** depois de treinar, rode `uv run python -m training.export_onnx --model brennocm/PII-BR --data data --push`, copie o hash do commit que ele cria (página do modelo → *Files → History*) e troque `MODEL_REVISION` em `scripts/deploy_space.py` (ou defina a variável `PII_BR_MODEL_REVISION` no GitHub). Sem isso a demo continua no modelo anterior, de propósito.

### Passo 8 — Portfólio (≈ 30 min)

1. Grave um GIF de 10–15 s usando o Space (ScreenToGif no Windows) e salve em `docs/assets/demo.gif`; eu coloco no README.
2. Adicione o card ao seu site em "Projetos", com o link principal para o Space.
3. Fixe o repositório no seu perfil do GitHub (*Customize your pins*).
4. Publique o post do LinkedIn (rascunho em `notes/post-linkedin.md`, fora do git).

## 5. Checklist antes de divulgar

- [ ] 2FA ativo no GitHub, Hugging Face e Kaggle; códigos de recuperação guardados fora do computador.
- [ ] Token do GitHub Actions com escrita **só** no Space; token pessoal não está em nenhum CI.
- [ ] `test_real` revisado por você (o card diz "revisadas por uma pessoa").
- [ ] Space respondendo, exemplos conferidos, status com o modelo carregado.
- [ ] Cards cruzados: dataset → modelo → Space → GitHub (os links dos cards apontam para os repositórios certos).
- [ ] Nenhum dado real em lugar nenhum: exemplos e `test_real` só com valores fictícios.

## 6. Como desfazer

- **Space:** cada deploy é um commit no repositório do Space. Volte pelo histórico do Space (*Files → History*) ou reverta o commit no GitHub (`git revert <sha> && git push`), que dispara um novo deploy. Para tirar o Space do ar na hora: *Settings → Pause Space*.
- **Modelo ou dataset:** também são repositórios git no Hub. Reverta pelo histórico na interface ou republique a versão anterior. Para esconder: *Settings → Make private*.
- **Token vazado:** revogue em Settings → Access Tokens, gere outro e atualize com `gh secret set HF_TOKEN` e no Kaggle.

## 7. Depois de publicar

- **Primeiras 24 h:** veja os *Logs* do Space e se os exemplos continuam funcionando depois de ele dormir e acordar.
- **Mensal:** Renovate/atualizações de dependências, `mise run check:all` verde, `pip-audit` sem alertas.
- **Quando mudar o gerador ou os templates:** gerar de novo, publicar o dataset, treinar de novo e reavaliar. O card registra a seed e a versão do Faker.
