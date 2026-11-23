---
title: PII-BR
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.32.0
python_version: "3.11"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Detecta e mascara dados pessoais (LGPD) em português
---

# PII-BR — detector e anonimizador LGPD

Cole um texto em português do Brasil e veja os dados pessoais destacados e mascarados: nome, CPF, CNPJ (inclusive o alfanumérico), RG, telefone, e-mail, endereço, CEP e data de nascimento.

O pipeline combina um modelo NER (BERTimbau fine-tuned, quantizado em int8 para CPU) com regras determinísticas que validam dígito verificador, DDD e CEP. Estratégias de mascaramento: redação, máscara parcial, pseudônimo consistente (HMAC com sal por sessão) e substituição por dado sintético válido.

**Privacidade:** o texto é processado em memória e não é armazenado nem registrado em log. Mesmo assim, não cole dados reais de terceiros numa demo pública.

O Space gratuito dorme depois de 48 h sem acesso; a primeira visita pode levar cerca de 1 minuto para acordar.

Código, dataset e modelo: veja os links na página do projeto no GitHub (PII-BR).
