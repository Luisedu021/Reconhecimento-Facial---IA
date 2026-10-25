# E02 — Fine-tuning parcial com ArcFace loss

- **Data:** 2026-10-25
- **Responsável:** Luís Eduardo Lopes dos Santos
- **Config:** [`configs/e02.yaml`](../configs/e02.yaml)
- **Commit:** `b323315` + correção do pré-processamento (`fix(preprocess)`, mesmo dia; não altera treino nem avaliação)

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
| Parâmetros treináveis | 2.980.096 (backbone 2.519.296 + cabeça ArcFace 460.800) |
| Seed | 42 |
| Hardware | RTX A1000 4 GB, AMP |
| Versões | ver coluna `versions` em `reports/results.csv` (iguais às do E01) |
| Tempo de treino / inferência | 3,5 min (5 épocas) / 2,6 ms por imagem |

## Dados

| Item | Valor |
|---|---|
| Treino | CelebA, 900 identidades, 17.970 imagens (20 por pessoa, menos as sem rosto detectado e as quase-duplicatas do LFW; no subconjunto todo foram 26 e 6) |
| Validação | CelebA, 100 identidades nunca vistas no treino, 6.000 pares |
| Teste | LFW View 2 |
| Checagem de leakage | split por identidade (CA-1/CA-2), sem augmentation fora do treino (CA-3), quase-duplicatas removidas (CA-4) |

## Métricas

Definidas antes de rodar (SPEC-001): as mesmas do E01, mais tempo de treino e parâmetros treináveis.

## Resultados

| Métrica | CelebA-val | LFW |
|---|---|---|
| Accuracy (10 folds) | — | 99,27% ± 0,32% |
| EER | 2,82% | 0,65% |
| TAR @ FAR = 1% | 95,17% | 99,47% |
| FAR / FRR no threshold da validação (0,400) | 1,00% / 4,83% | 0,67% / 0,67% |
| Tempo de treino / parâmetros treináveis | 3,5 min / 2,98 M | |

Os números são **idênticos aos do E01** porque o checkpoint escolhido foi o da época 1 (ver abaixo), inclusive
precision/recall (98,96% / 95,17% na validação; 99,33% / 99,33% no LFW) e a matriz de confusão (ver
[E01](E01-baseline.md)).

Evolução por época (`reports/E02-historico.csv`):

| Época | 1 (aquecimento) | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Perda de treino | 27,40 | 14,72 | 8,19 | 6,01 | 4,92 |
| EER CelebA-val | **2,82%** | 2,93% | 3,00% | 3,18% | 3,07% |

Gráficos: `reports/figures/E02-roc.png`, `reports/figures/E02-lfw-similaridade.png`.

## Interpretação

- **A hipótese foi refutada.** Assim que as camadas finais passaram a treinar (época 2), o EER de validação
  **subiu** e não voltou ao nível inicial. A melhor época foi a 1, em que só a cabeça ArcFace treina e o
  backbone fica intacto: na prática, o modelo salvo é o próprio E01.
- **Sinal de overfitting e esquecimento:** a perda de treino caiu de 27,4 para 4,9, mas a validação piorou.
  A rede aprendeu a separar as 900 pessoas do treino à custa da generalização para pessoas novas. Com apenas
  ~18 mil imagens, o ajuste das camadas finais desloca um espaço de embeddings aprendido com 3,3 milhões de
  fotos do VGGFace2.
- **Por que a ArcFace prejudicou:** com s = 64 e m = 0,5, a perda força cada embedding na direção de um de
  900 centros novos e aleatórios. É um sinal forte, pensado para treino em larga escala; em poucos dados e
  poucas épocas ele reorganiza o espaço mais do que melhora. O [E05](E05-triplet-loss.md) testa a alternativa
  de sinal mais suave (Triplet loss).
- **A seleção por validação funcionou como esperado:** como o checkpoint só é salvo quando o EER melhora,
  o fine-tuning ruim não chegou ao modelo final. O custo foi 3,5 min de treino sem ganho.
- **Limitações:** uma seed, 5 épocas, uma combinação de learning rate e margem; não testamos m menor nem
  learning rate menor com ArcFace.
