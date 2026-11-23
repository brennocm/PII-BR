# 0002. Deploy: Hugging Face Hub e Spaces, treino no Kaggle

- Status: accepted (a demo foi substituída pelo ADR 0003)
- Date: 2026-11-23

## Context

O projeto é de portfólio e precisa custar zero. Entrega três peças públicas (dataset, modelo, demo) que devem ser fáceis de testar por quem não instala nada. A máquina de desenvolvimento não tem GPU CUDA: o treino completo levaria cerca de 16 h em CPU (medido; ver `SPEC.md`).

## Decision

- **Dataset e modelo** no Hugging Face Hub, em repositórios públicos.
- **Demo** num Space Gradio em CPU basic gratuita (2 vCPU, 16 GB), com o modelo quantizado em int8 dinâmico.
- **Treino** num notebook no Kaggle (GPU T4, 30 h/semana grátis), com Colab como alternativa. O token do Hugging Face fica nos secrets da plataforma.
- **Sync do Space** por GitHub Action: a cada push na `main` com testes verdes, `huggingface_hub.upload_folder` envia `space/` e `src/pii_br/`.

## Alternatives considered

- **Treino local em CPU**: ~16 h para o treino completo; fica como plano B reduzido.
- **VPS ou PaaS para a demo**: custo mensal e operação sem ganho para um app sem estado.
- **GPU paga (Colab Pro, Lambda, RunPod)**: desnecessária para um treino de 30 min.

## Consequences

- O Space gratuito dorme após 48 h sem acesso; o primeiro acesso leva de 30 s a 1 min.
- O app não tem estado nem armazena texto; não há banco nem backup a operar.
- Rever se a demo precisar de latência garantida ou de uso por terceiros em produção (aí, um endpoint pago ou self-hosted).
