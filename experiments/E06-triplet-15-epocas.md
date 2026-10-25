# E06 — Triplet loss com 15 épocas

- **Data:** 2026-10-25
- **Responsável:** Luís Eduardo Lopes dos Santos
- **Config:** [`configs/e06.yaml`](../configs/e06.yaml) (herda do [`e05.yaml`](../configs/e05.yaml))
- **Commit:** `8157baa` + adendo da SPEC-001 e `configs/e06.yaml` (registrados antes de rodar)

## Hipótese

Mais épocas de Triplet loss (15 em vez de 5) reduzem o EER em CelebA-val abaixo dos 2,57% do
[E05](E05-triplet-loss.md), cuja curva de validação ainda descia na última época. Registrada no adendo da
[SPEC-001](../docs/specs/001-protocolo-experimental.md) antes da execução.

## Configuração

Igual ao E05 (fine-tuning parcial de `block8`, `last_linear`, `last_bn`; Triplet loss com margem 0,2 e mineração
semi-hard; batch P×K 16×4; AdamW, lr 1e-4; seed 42), exceto:

| Item | Valor |
|---|---|
| Épocas | 15 (E05: 5) |
| Escolha da época | menor EER em CelebA-val (mesmo critério) |
| Parâmetros treináveis | 2.519.296 |
| Hardware / versões | RTX A1000 4 GB, AMP / iguais ao E05 (coluna `versions` de `reports/results.csv`) |
| Tempo de treino / inferência | 2,9 min (15 épocas) / 2,9 ms por imagem |

## Dados

Mesmos do E02/E05: 17.970 imagens de treino (900 identidades), 6.000 pares de validação (100 identidades nunca
vistas no treino), LFW View 2 como teste. Split por identidade, sem augmentation fora do treino, 6 quase-duplicatas
do LFW removidas, threshold escolhido só na validação.

## Métricas

As mesmas do E05, definidas antes de rodar.

## Resultados

| Métrica | CelebA-val | LFW |
|---|---|---|
| Accuracy (10 folds) | — | **99,37% ± 0,41%** |
| EER | **2,38%** | **0,63%** |
| TAR @ FAR = 1% | 96,27% | 99,37% |
| ROC-AUC | 0,9949 | 0,9993 |
| Accuracy / F1 no threshold da validação | 97,63% / 0,9760 | 99,00% / 0,9900 |
| Precision / recall no threshold da validação | 98,97% / 96,27% | 98,61% / 99,40% |
| FAR / FRR no threshold da validação (0,439) | 1,00% / 3,73% | 1,40% / 0,60% |
| Tempo de treino / parâmetros treináveis | 2,9 min / 2,52 M | |

Matriz de confusão no threshold da validação (0,439), em pares. "Positivo" = par da mesma pessoa:

| | CelebA-val: previsto mesma | CelebA-val: previsto diferente | LFW: previsto mesma | LFW: previsto diferente |
|---|---|---|---|---|
| **Real: mesma pessoa** (3.000) | 2.888 (VP) | 112 (FN) | 2.982 (VP) | 18 (FN) |
| **Real: pessoas diferentes** (3.000) | 30 (FP) | 2.970 (VN) | 42 (FP) | 2.958 (VN) |

Evolução por época (`reports/E06-historico.csv`):

| Época | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | **9** | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EER val (%) | 2,93 | 2,83 | 2,65 | 2,72 | 2,57 | 2,57 | 2,40 | 2,53 | **2,38** | 2,62 | 2,77 | 2,78 | 2,87 | 2,72 | 2,80 |

Gráficos: `reports/figures/E06-roc.png`, `reports/figures/E06-lfw-similaridade.png`.

## Interpretação

- **A hipótese se confirmou.** O EER de validação caiu até **2,38% na época 9**, o menor de todos os experimentos
  (E01: 2,82%; E05: 2,57%), com TAR @ FAR = 1% de 96,27% (E05: 95,27%).
- **Depois da época 9 vem o overfitting:** da época 10 em diante a validação piora (até 2,87%) enquanto a perda de
  treino continua caindo devagar. A seleção pela validação descarta essas épocas. É o mesmo padrão do E02/E03,
  mas bem mais tardio e suave: a Triplet só gera gradiente nos trios difíceis.
- **Desta vez o ganho aparece no teste:** no LFW, 99,37% em 10 folds e EER de 0,63%, os dois melhores que o
  baseline (99,27% e 0,65%). A diferença continua dentro do desvio entre folds (± 0,41%), então o LFW confirma a
  tendência sem provar significância. O LFW não participou da escolha (item 29).
- **O threshold continua transferindo mal:** FAR de 1,40% no LFW com o threshold da validação, acima do alvo de 1%.
  No threshold oficial de cada modelo, o E06 erra **mais** no LFW (60 pares: 42 FP e 18 FN) que o E05 (57) e o
  E01 (40), embora tenha o menor EER. Ou seja, os embeddings separam melhor (EER 0,63%), mas o threshold escolhido
  no CelebA cai num ponto pior do LFW, com mais falsas aceitações, o erro grave. A causa é a calibração do
  threshold entre domínios, não a separação dos embeddings.
- **Reprodutibilidade:** as épocas 1 a 5 do E06 repetiram exatamente as do E05 (mesma seed, mesmo código), o que
  confirma o CA-7 com dados reais.
- **Custo:** 2,9 min de treino, ainda mais barato que o E02 (3,5 min) e o E03 (4,6 min).
- **Limitações:** uma seed; o melhor ponto (época 9) foi escolhido entre 15 avaliações na mesma validação, então
  parte da vantagem pode ser sorte de seleção, da ordem de grandeza do erro amostral (≈ 0,3 p.p.).
