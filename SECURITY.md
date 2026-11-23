# Segurança

## Como reportar uma vulnerabilidade

Use o reporte privado do GitHub: aba **Security → Report a vulnerability** neste repositório. Não abra issue pública para falhas que possam expor dados pessoais.

Inclua o texto de entrada (com dados fictícios), o comportamento esperado e o observado. A resposta inicial sai em até 7 dias.

## O que conta como vulnerabilidade

- Dado pessoal que deveria ser detectado e sai sem máscara por causa de um bug (não por limite do modelo, que é medido no model card).
- Entrada que trava ou derruba o detector ou o Space (ReDoS, consumo excessivo de memória).
- Pseudônimo que permite recuperar o valor original.
- Vazamento de texto do usuário em logs, arquivos temporários ou analytics do Space.

## Fora de escopo

- Erros de detecção do modelo em frases específicas: abra uma issue normal com o exemplo, usando só dados fictícios.
- O PII-BR reduz risco, mas não certifica anonimização no sentido do art. 12 da LGPD.
