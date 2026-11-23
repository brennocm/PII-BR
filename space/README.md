---
title: PII-BR
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: static
app_file: index.html
pinned: false
license: apache-2.0
short_description: Detecta e mascara dados pessoais (LGPD) em PT-BR
models:
  - brennocm/PII-BR
datasets:
  - brennocm/PII-BR
---

# PII-BR: detector e anonimizador LGPD

Cole um texto em português do Brasil e veja os dados pessoais destacados e mascarados: nome, CPF, CNPJ (inclusive o alfanumérico), RG, telefone, e-mail, endereço, CEP e data de nascimento.

**Tudo roda no seu navegador.** O pacote `pii_br` (regras com dígito verificador, fusão e mascaramento) roda em Python via Pyodide, e o modelo NER (BERTimbau fine-tuned, ONNX int8) roda no `onnxruntime-web`. O texto que você cola não é enviado a nenhum servidor. Na primeira visita a página baixa cerca de 200 MB (o modelo), que ficam no cache do navegador.

Estratégias de mascaramento: redação, máscara parcial, pseudônimo consistente (HMAC com sal por aba) e substituição por dado sintético válido.

[Código](https://github.com/brennocm/PII-BR) · [modelo](https://huggingface.co/brennocm/PII-BR) · [dataset](https://huggingface.co/datasets/brennocm/PII-BR)
