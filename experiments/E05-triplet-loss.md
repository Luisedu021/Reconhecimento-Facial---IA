# E05 — Fine-tuning parcial com Triplet loss (mineração semi-hard)

- **Data:** 2026-10-25
- **Responsável:** Luís Eduardo Lopes dos Santos
- **Config:** [`configs/e05.yaml`](../configs/e05.yaml)
- **Commit:** `b323315` + correção do pré-processamento (`fix(preprocess)`, mesmo dia; não altera treino nem avaliação)

## Hipótese

Com o mesmo fine-tuning parcial do [E02](E02-finetuning-parcial.md), a Triplet loss (comparação direta entre
exemplos do batch) gera embeddings **piores** que a ArcFace, que usa todas as identidades em cada passo.

## Configuração

Igual ao E02, exceto:

| Item | Valor |
|---|---|
| Função de perda | Triplet, margem 0,2, distância euclidiana ao quadrado entre embeddings normalizados |
| Mineração | semi-hard dentro do batch (negativo mais próximo que ainda está mais longe que o positivo) |
| Batch | amostragem P×K: 16 pessoas × 4 fotos = 64 |
| Aquecimento | nenhum (não há cabeça nova: a perda atua direto nos embeddings) |
| Parâmetros treináveis | 2.519.296 (`block8`, `last_linear`, `last_bn`) |
| Seed / hardware / versões | 42 / RTX A1000 4 GB, AMP / Python 3.10.11, torch 2.2.2+cu121, facenet-pytorch 2.6.0 (coluna `versions` de `reports/results.csv`) |
| Tempo de treino / inferência | 55 s (5 épocas) / 2,0 ms por imagem |

## Dados

Mesmos do E02: 17.970 imagens de treino (900 identidades), 6.000 pares de validação (100 identidades nunca
vistas no treino), LFW View 2 como teste. Split por identidade (CA-1/CA-2), sem augmentation fora do treino
(CA-3), 6 quase-duplicatas do LFW removidas (CA-4). Threshold escolhido só na validação.

## Métricas

Definidas antes de rodar (SPEC-001): EER e TAR @ FAR = 1% na validação (critério de escolha: menor EER);
accuracy em 10 folds, EER e FAR/FRR com o threshold da validação no LFW; tempo de treino e parâmetros.

## Resultados

| Métrica | CelebA-val | LFW |
|---|---|---|
| Accuracy (10 folds) | — | 99,17% ± 0,38% |
| EER | **2,57%** | 0,83% |
| TAR @ FAR = 1% | 95,27% | 99,30% |
| ROC-AUC | 0,9943 | 0,9994 |
| Accuracy / F1 no threshold da validação | 97,13% / 0,9708 | 99,05% / 0,9905 |
| Precision / recall no threshold da validação | 98,96% / 95,27% | 98,61% / 99,50% |
| FAR / FRR no threshold da validação (0,450) | 1,00% / 4,73% | 1,40% / 0,50% |
| Tempo de treino / parâmetros treináveis | 55 s / 2,52 M | |

Matriz de confusão no threshold da validação (0,450), em pares. "Positivo" = par da mesma pessoa:

| | CelebA-val: previsto mesma | CelebA-val: previsto diferente | LFW: previsto mesma | LFW: previsto diferente |
|---|---|---|---|---|
| **Real: mesma pessoa** (3.000) | 2.858 (VP) | 142 (FN) | 2.985 (VP) | 15 (FN) |
| **Real: pessoas diferentes** (3.000) | 30 (FP) | 2.970 (VN) | 42 (FP) | 2.958 (VN) |

Comparado ao E01 no LFW: 5 falsas rejeições a menos (15 contra 20) e 22 falsas aceitações a mais (42 contra 20).
O total de erros sobe de 40 para 57, e o tipo de erro muda para o mais grave no cenário (aceitar um estranho).

Evolução por época (`reports/E05-historico.csv`):

| Época | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Perda de treino | 0,0148 | 0,0121 | 0,0129 | 0,0114 | 0,0122 |
| EER CelebA-val | 2,93% | 2,83% | 2,65% | 2,72% | **2,57%** |

Gráficos: `reports/figures/E05-roc.png`, `reports/figures/E05-lfw-similaridade.png`.

## Interpretação

- **A hipótese foi refutada.** A Triplet loss teve o **menor EER de validação** de todos os experimentos
  (2,57% contra 2,82% do E01/E02/E03), enquanto a ArcFace piorou a validação a cada época (E02, E03).
- **Por que a triplet funcionou melhor aqui:** a ArcFace com s = 64 e m = 0,5 empurra os embeddings para 900
  centros de classe novos, um sinal forte que reorganiza o espaço aprendido no VGGFace2 (esquecimento).
  A triplet com mineração semi-hard só gera gradiente quando um negativo invade a margem; a perda já começa
  baixa (0,015) porque o modelo pré-treinado separa bem quase todos os trios. O ajuste é **suave e local**:
  corrige os poucos casos difíceis sem desmontar o que já funcionava.
- **A melhora é pequena:** 0,25 ponto percentual de EER, da ordem do erro amostral da validação
  (≈ 0,3 p.p. com 3.000 pares de cada tipo). A curva ainda descia na época 5, então mais épocas poderiam ajudar.
- **No teste (LFW) o ganho não se confirmou:** accuracy 99,17% contra 99,27% do E01 (dentro do desvio entre
  folds) e FAR de 1,40% com o threshold da validação, acima do alvo de 1% (RNF-01). O threshold do E05 é mais
  alto (0,450 contra 0,400) e transfere um pouco pior para o LFW. Esse resultado **não** muda a escolha, que é
  feita só pela validação (item 29), mas fica registrado como limitação no [ADR-001](../docs/decisions/ADR-001-face-model.md).
- **Custo:** o mais barato dos fine-tunings (55 s, contra 3,5 min do E02 e 4,6 min do E03), porque não há
  cabeça de 900 classes e a perda é calculada só dentro do batch.
- **Limitações:** uma seed, 5 épocas, um valor de margem; diferenças no LFW perto da saturação (> 99%).
