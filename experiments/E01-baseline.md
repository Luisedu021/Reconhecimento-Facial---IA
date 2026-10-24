# E01 — Baseline: modelo pré-treinado sem fine-tuning

- **Data:** _AAAA-MM-DD_
- **Responsável:** _nome_
- **Config:** [`configs/e01.yaml`](../configs/e01.yaml)
- **Commit:** _hash do código usado_

## Hipótese

O InceptionResnetV1 pré-treinado no VGGFace2, sem nenhum ajuste, já separa bem a mesma pessoa de pessoas
diferentes e serve de linha de base para medir o efeito do fine-tuning.

## Configuração

| Item | Valor |
|---|---|
| Modelo / backbone | InceptionResnetV1 (FaceNet), embedding de 512 dimensões |
| Pesos iniciais | `vggface2` do facenet-pytorch (licença: _ver ADR-001_) |
| Estratégia | sem fine-tuning |
| Função de perda | — |
| Comparação | similaridade de cosseno |
| Pré-processamento | MTCNN, recorte 160×160, margem 14 px |
| Threshold | escolhido em CelebA-val com FAR ≤ 1% |
| Seed | 42 |
| Hardware | RTX A1000 4 GB, AMP |
| Versões | ver coluna `versions` em `reports/results.csv` |

## Dados

| Item | Valor |
|---|---|
| Validação | CelebA, ~100 identidades fora do treino, 3.000 pares iguais + 3.000 diferentes |
| Teste | LFW View 2, 6.000 pares em 10 folds |
| Checagem de leakage | nada é treinado; threshold só na validação |

## Resultados

| Métrica | CelebA-val | LFW |
|---|---|---|
| Accuracy (10 folds) | — | |
| EER | | |
| TAR @ FAR = 1% | | |
| FAR / FRR no threshold da validação | | |
| Tempo por imagem | | |

Gráficos: `reports/figures/E01-roc.png`, `reports/figures/E01-lfw-similaridade.png`.

## Interpretação

_A preencher depois de rodar._
