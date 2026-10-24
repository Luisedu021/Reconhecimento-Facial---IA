# E02 — Fine-tuning parcial com ArcFace loss

- **Data:** _AAAA-MM-DD_
- **Responsável:** _nome_
- **Config:** [`configs/e02.yaml`](../configs/e02.yaml)
- **Commit:** _hash do código usado_

## Hipótese

Ajustar só as camadas finais (block8 + camada de embedding) com ArcFace loss no CelebA deixa os embeddings
mais discriminativos que o baseline (E01), sem esquecer o que o modelo já aprendeu.

## Configuração

| Item | Valor |
|---|---|
| Modelo / backbone | InceptionResnetV1, pesos `vggface2` |
| Estratégia | parcial: treinam `block8`, `last_linear`, `last_bn`; o resto fica congelado (BatchNorm em modo eval) |
| Função de perda | ArcFace, s = 64, m = 0,5 rad |
| Comparação | similaridade de cosseno |
| Optimizer | AdamW, weight decay 5e-4 |
| Learning rate | 1e-3 na cabeça ArcFace, 1e-4 nas camadas finais |
| Batch / épocas | 64 / 5 (1ª época de aquecimento: só a cabeça ArcFace) |
| Augmentations (só treino) | flip horizontal, ColorJitter 0,2 |
| Escolha da época | menor EER em CelebA-val |
| Seed | 42 |
| Hardware | RTX A1000 4 GB, AMP |

## Dados

| Item | Valor |
|---|---|
| Treino | CelebA, ~900 identidades × 20 fotos, sem as quase-duplicatas do LFW |
| Validação | CelebA, ~100 identidades nunca vistas no treino |
| Teste | LFW View 2 |
| Checagem de leakage | split por identidade (CA-1/CA-2), sem augmentation fora do treino (CA-3), quase-duplicatas removidas (CA-4) |

## Resultados

| Métrica | CelebA-val | LFW |
|---|---|---|
| Accuracy (10 folds) | — | |
| EER | | |
| TAR @ FAR = 1% | | |
| FAR / FRR no threshold da validação | | |
| Tempo de treino / parâmetros treináveis | | |

Evolução por época: `reports/E02-historico.csv`. Gráficos: `reports/figures/E02-*.png`.

## Interpretação

_A preencher depois de rodar: melhorou ou piorou em relação ao E01? Por quê?_
