# E01 — Baseline: modelo pré-treinado sem fine-tuning

- **Data:** 2026-10-25
- **Responsável:** Luís Eduardo Lopes dos Santos
- **Config:** [`configs/e01.yaml`](../configs/e01.yaml)
- **Commit:** `b323315` + correção do pré-processamento (`fix(preprocess)`, mesmo dia; não altera treino nem avaliação)

## Hipótese

O InceptionResnetV1 pré-treinado no VGGFace2, sem nenhum ajuste, já separa bem a mesma pessoa de pessoas
diferentes e serve de linha de base para medir o efeito do fine-tuning.

## Configuração

| Item | Valor |
|---|---|
| Modelo / backbone | InceptionResnetV1 (FaceNet), embedding de 512 dimensões |
| Pesos iniciais | `vggface2` do facenet-pytorch (código MIT; termos dos pesos no [ADR-001](../docs/decisions/ADR-001-face-model.md)) |
| Estratégia | sem fine-tuning |
| Função de perda | — |
| Comparação | similaridade de cosseno |
| Pré-processamento | MTCNN, recorte 160×160, margem 14 px |
| Threshold | escolhido em CelebA-val com FAR ≤ 1% |
| Seed | 42 |
| Hardware | RTX A1000 4 GB, AMP |
| Versões | Python 3.10.11, torch 2.2.2+cu121, torchvision 0.17.2, facenet-pytorch 2.6.0, numpy 1.26.4, scikit-learn 1.5.2 |
| Tempo de inferência | 3,5 ms por imagem (recorte já em cache) |

## Dados

| Item | Valor |
|---|---|
| Validação | CelebA, 100 identidades fora do treino, 3.000 pares iguais + 3.000 diferentes |
| Teste | LFW View 2, 6.000 pares em 10 folds (13.233 imagens; MTCNN achou rosto em todas) |
| Checagem de leakage | nada é treinado; threshold só na validação |

## Métricas

Definidas antes de rodar (SPEC-001): EER, TAR @ FAR = 1%, ROC-AUC, accuracy e F1; no LFW, também a accuracy em
10 folds e FAR/FRR com o threshold da validação.

## Resultados

| Métrica | CelebA-val | LFW |
|---|---|---|
| Accuracy (10 folds) | — | 99,27% ± 0,32% |
| Accuracy / F1 no threshold da validação | 97,08% / 0,9703 | 99,33% / 0,9933 |
| EER | 2,82% | 0,65% |
| TAR @ FAR = 1% | 95,17% | 99,47% |
| ROC-AUC | 0,9928 | 0,9995 |
| FAR / FRR no threshold da validação (0,400) | 1,00% / 4,83% | 0,67% / 0,67% |
| Tempo por imagem | 3,5 ms | |

Gráficos: `reports/figures/E01-roc.png`, `reports/figures/E01-lfw-similaridade.png`.

## Interpretação

- **A hipótese se confirmou.** Sem nenhum treino, o modelo chega a 99,27% no LFW e a 2,82% de EER no CelebA-val.
  O README do facenet-pytorch informa 99,65% no LFW para esses pesos, com o alinhamento deles; a diferença é
  compatível com detalhes de recorte e margem. A linha de base é alta e difícil de superar.
- **O CelebA-val é mais difícil que o LFW** (EER 2,82% contra 0,65%). As imagens alinhadas do CelebA são
  menores (178×218) e o domínio difere do VGGFace2 (maquiagem, poses de foto de celebridade). Por isso a
  validação separa melhor os experimentos do que o LFW, que está perto da saturação.
- **O threshold generalizou:** escolhido para FAR = 1% no CelebA-val, deu FAR de 0,67% no LFW, dentro do alvo.
- **Erros:** com 0,67% de FAR e 0,67% de FRR no LFW, sobram cerca de 40 pares errados em 6.000; a região de
  sobreposição das duas distribuições aparece em `E01-lfw-similaridade.png`. Não inspecionamos imagens
  individuais.
- **Custo:** nenhum treino; 3,5 ms por imagem na GPU.
- **Limitações:** uma única seed para os pares de validação; o LFW perto da saturação esconde diferenças pequenas.
