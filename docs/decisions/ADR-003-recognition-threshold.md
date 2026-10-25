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
| Threshold escolhido em CelebA-val (modelo do ADR-001: E06) | **0,4386** (FAR 1,00%, FRR 3,73% na validação) — `reports/thresholds.json` |
| LFW com esse threshold: FAR / FRR | 1,40% / 0,60% (acima do alvo de 1%; para comparação, o E01 com 0,400 deu 0,67% / 0,67%) |
| Dataset privado com esse threshold: rejeição de desconhecidos / FRR (agregado, [E04](../../experiments/E04-dataset-privado.md)) | rodízio: 56% (protótipo e k-NN), ou seja FAR de 44%; LFW como desconhecido: 97,0% / 95,6%; FRR 0% nos dois (5 pessoas, 15 consultas) |
| Sensibilidade no dataset privado (só análise, CA-9) | FAR do rodízio: 44% em 0,45, 36% em 0,50–0,60, 28% em 0,70, 0–4% em 0,75; FRR só sobe em 0,80 (13%). FAR com rostos do LFW < 1% a partir de 0,50 — `reports/E04-sensibilidade.csv` |
| Gráficos | `reports/figures/<experimento>-roc.png`, `-lfw-similaridade.png`, `E04-privado-similaridade.png` e `E04-sensibilidade-threshold.png` |

## Consequências

- **Positivas:**
  - O threshold tem origem auditável: arquivo, critério e conjunto usados ficam registrados.
  - O mesmo valor serve para a aplicação da Parte 2.
- **Negativas e riscos:**
  - **Mudança de domínio:** um threshold ajustado em fotos de celebridades pode não valer para selfies. O E04
    mede exatamente isso e, se o efeito for grande, vira uma limitação a discutir (item 33).
  - Já no LFW o efeito apareceu: os thresholds do E05 e do E06 deram FAR de 1,40%, acima do alvo. O valor
    **não** foi reajustado no LFW (item 29); fica registrado como limitação.
  - **No dataset privado o efeito foi grande:** 44% dos participantes fora do cadastro foram aceitos como outro
    participante (36–40% com o E05), embora as consultas S2 tenham separado perfeitamente mesma pessoa × pessoas
    diferentes (AUC = 1,0). A análise de sensibilidade mostra que, nesta amostra, o FAR entre participantes só cai
    perto de 0,75, longe do 0,439 escolhido em celebridades. Esse valor **não** foi adotado: foi observado no
    teste (CA-5). Parte desse efeito vem de o grupo incluir participantes com parentesco, caso em que pessoas
    diferentes ficam muito parecidas; com rostos do LFW, o FAR fica abaixo de 1% a partir de 0,50.
  - **Revisão prevista para a Parte 2:** escolher de novo na validação do CelebA com um critério mais rígido
    (por exemplo FAR ≤ 0,1%) e registrar a mudança como nova versão deste ADR, citando o E04 como motivação. Se
    ainda não bastar, buscar uma validação pública mais parecida com selfies.
- **Privacidade:** nenhum dado privado participa da escolha.
