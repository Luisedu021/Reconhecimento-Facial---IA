# ADR-003 — Threshold de reconhecimento

- **Status:** aceita em 2026-10-24 (critério decidido; o valor numérico sai da validação do modelo escolhido no ADR-001)
- **Data:** 2026-10-24
- **Decisores:** Luís Eduardo Lopes dos Santos, Gustavo Borges Machado
- **Relacionados:** [SPEC-001](../specs/001-protocolo-experimental.md), [ADR-001](ADR-001-face-model.md), E04,
  [requirements.md](../requirements.md) RNF-01 e RNF-02, [enunciado](../enunciado-parte1.md) itens 29 e 32

## Contexto

O sistema precisa responder "pessoa reconhecida" ou "pessoa desconhecida" (item 32). A decisão é:

```
similaridade de cosseno >= threshold  →  reconhecida
similaridade de cosseno <  threshold  →  desconhecida
```

A escolha deve ser justificada experimentalmente e não pode usar o conjunto de teste (item 29). Os dois erros
possíveis puxam o threshold para lados opostos:

- **FAR**: aceitar uma pessoa não cadastrada (ou confundir duas pessoas). Diminui com threshold **alto**.
- **FRR**: rejeitar uma pessoa cadastrada. Diminui com threshold **baixo**.

## Alternativas consideradas

| Alternativa | Prós | Contras |
|---|---|---|
| **A. FAR-alvo (1%) na validação** | Controla diretamente o erro mais grave num cenário de acesso; valor fácil de explicar | FRR maior: a pessoa cadastrada às vezes precisa tentar de novo |
| B. EER na validação | Equilibra os dois erros; não exige escolher um alvo | Trata os dois erros como igualmente graves, o que raramente é verdade |
| C. Máxima acurácia na validação | É o critério do protocolo do LFW | Depende da proporção de pares iguais e diferentes, que no uso real não é 50/50 |
| D. Ajustar nas fotos privadas | Usaria o domínio real | **Proibido**: as fotos privadas são teste (item 29); com 5 pessoas o valor seria instável |

## Decisão

**Alternativa A: FAR ≤ 1% na validação.** O threshold é o menor valor com FAR ≤ 1% nos pares de validação do
CelebA, que são identidades nunca vistas no treino. Ele é calculado por `facerec.evaluate` e gravado em
`reports/thresholds.json`. Depois disso não muda: o LFW e o dataset privado só medem o que acontece com ele.

O grupo considera aceitar uma pessoa não cadastrada o erro mais grave. Rejeitar uma pessoa cadastrada é
contornável: ela tenta de novo. Isso fica registrado como RNF-01 em [requirements.md](../requirements.md).
Configuração: `threshold.criterion: far` e `threshold.far_target: 0.01` em `configs/base.yaml`.

## Evidências

| Evidência | Valor |
|---|---|
| Threshold escolhido em CelebA-val (modelo do ADR-001) | _a preencher_ |
| LFW com esse threshold: FAR / FRR | _a preencher_ |
| Dataset privado com esse threshold: rejeição de desconhecidos / FRR (agregado) | _a preencher_ |
| Gráficos | `reports/figures/<experimento>-roc.png` e `-lfw-similaridade.png` |

## Consequências

- **Positivas:**
  - O threshold tem origem auditável: arquivo, critério e conjunto usados ficam registrados.
  - O mesmo valor serve para a aplicação da Parte 2.
- **Negativas e riscos:**
  - **Mudança de domínio:** um threshold ajustado em fotos de celebridades pode não valer para selfies. O E04
    mede exatamente isso e, se o efeito for grande, vira uma limitação a discutir (item 33).
- **Privacidade:** nenhum dado privado participa da escolha.
