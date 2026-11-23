# 0003. Demo estática no navegador (Pyodide + ONNX)

- Status: accepted
- Date: 2026-11-23
- Substitui, em parte: 0002 (a demo deixa de ser um Space Gradio)

## Context

Em novembro de 2026 o Hugging Face passou a exigir plano pago (PRO) para Spaces Gradio e Docker; só o Space `static` continua gratuito para contas free. O projeto precisa custar zero, e a demo é a peça mais vista do portfólio.

O núcleo `pii_br` só usa a biblioteca padrão, então roda no Pyodide (CPython em WebAssembly) sem alteração. O modelo é um BERT base que, exportado para ONNX com int8 dinâmico, cabe em ~190 MB e roda no `onnxruntime-web`.

## Decision

- **Space `static`**: `index.html` + `app.js` + `style.css`. O Pyodide carrega o pacote (`pii_br.zip`, gerado no deploy) e a ponte `space/bridge.py`; o `onnxruntime-web` (backend WASM) roda o modelo.
- **Tokenização e decodificação em Python puro** (`pii_br/ner.py`): WordPiece compatível com o tokenizer do BERTimbau, janelas de 256 tokens com sobreposição de 64 e rótulo no primeiro subtoken de cada palavra de `words.py`, exatamente como no treino. O mesmo código serve ao `NerModel` (torch), ao `OnnxModel` (Python) e ao navegador.
- **Modelo ONNX** em `onnx/model_int8.onnx` no repositório do modelo, gerado por `training/export_onnx.py`, que falha se o int8 perder mais de 0,5 ponto de F1 contra o fp32. A saída do FFN fica em fp32 (ver `AGENTS.md`).
- **Versões fixas**: Pyodide e `onnxruntime-web` com versão exata e SRI; o modelo por commit (`PII_BR_MODEL_REVISION`).
- O app Gradio vai para `app/` e segue como app local (`mise run dev`), pronto para um Space pago se um dia fizer sentido.

## Alternatives considered

- **HF PRO (~US$ 9/mês)**: mantém o app Gradio sem mudança de código, mas quebra o custo zero.
- **Só regras no navegador**: grátis e simples, mas perde nome, endereço e RG, o ponto forte do projeto.
- **Outro host para o Gradio (VM free tier, PaaS)**: costuma exigir cartão e dá operação a manter.
- **Transformers.js em vez de onnxruntime-web + Python**: exigiria reescrever regras, fusão e mascaramento em JavaScript, com duas implementações para manter iguais.

## Consequences

- **Privacidade melhor**: o texto nunca sai do navegador; não há servidor para logar nada.
- Primeira visita baixa ~200 MB (modelo + Pyodide); depois o modelo fica no Cache Storage. Até o modelo carregar, a página funciona só com as regras.
- Latência depende da máquina do visitante: ~2 s para 1.000 caracteres numa thread de WASM (o Space estático não envia os cabeçalhos de isolamento que liberariam threads).
- Upload de `.txt`/`.csv` fica só no app Gradio local.
- O Space estático não dorme.
