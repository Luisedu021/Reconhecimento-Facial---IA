# E03 — Fine-tuning completo com ArcFace loss

- **Data:** 2026-10-25
- **Responsável:** Luís Eduardo Lopes dos Santos
- **Config:** [`configs/e03.yaml`](../configs/e03.yaml)
- **Commit:** `b323315` + correção do pré-processamento (`fix(preprocess)`, mesmo dia; não altera treino nem avaliação)

## Hipótese

Atualizar a rede inteira (learning rate 10× menor no backbone) adapta melhor ao domínio que o fine-tuning
parcial (E02), mas com mais custo e mais risco de overfitting.

## Configuração

Igual ao [E02](E02-finetuning-parcial.md), exceto:

| Item | Valor |
|---|---|
| Estratégia | completa: todos os parâmetros treinam |
| Learning rate | 1e-3 na cabeça ArcFace, 1e-4 nas camadas finais, 1e-5 no restante do backbone |
| Parâmetros treináveis | 23.943.424 (backbone 23.482.624 + cabeça ArcFace 460.800) |
| Tempo de treino / inferência | 4,6 min (5 épocas) / 3,1 ms por imagem |

Batch, épocas, augmentations e seed são os mesmos do E02: só a estratégia muda, para isolar o efeito dela.

## Dados

Mesmos do E02.

## Métricas

Mesmas do E02.

## Resultados

| Métrica | CelebA-val | LFW |
|---|---|---|
| Accuracy (10 folds) | — | 99,27% ± 0,32% |
| EER | 2,82% | 0,65% |
| TAR @ FAR = 1% | 95,17% | 99,47% |
| FAR / FRR no threshold da validação (0,400) | 1,00% / 4,83% | 0,67% / 0,67% |
| Tempo de treino / parâmetros treináveis | 4,6 min / 23,9 M | |

Como no E02, o checkpoint escolhido foi o da época 1 (aquecimento, backbone intacto), por isso os números
coincidem com os do E01.

Evolução por época (`reports/E03-historico.csv`), comparada ao E02:

| Época | 1 (aquecimento) | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Perda de treino E03 | 27,40 | 15,12 | 7,77 | 5,18 | 3,82 |
| EER CelebA-val E03 | **2,82%** | 3,12% | 3,12% | 3,12% | 3,37% |
| EER CelebA-val E02 | **2,82%** | 2,93% | 3,00% | 3,18% | 3,07% |

Gráficos: `reports/figures/E03-roc.png`, `reports/figures/E03-lfw-similaridade.png`.

## Interpretação

- **A primeira parte da hipótese foi refutada e a segunda, confirmada.** O fine-tuning completo não adaptou
  melhor: piorou a validação já na época 2 e terminou com o pior EER (3,37%). O risco de overfitting previsto
  se concretizou: a perda de treino caiu mais que no E02 (3,82 contra 4,92), e a validação piorou mais.
- **Mais parâmetros, mais esquecimento:** com 23,9 M parâmetros livres (8× o E02) e 900 identidades, a rede
  tem capacidade de sobra para decorar o treino. Mesmo com learning rate 10× menor no backbone, as camadas
  iniciais também se deslocam do que foi aprendido no VGGFace2.
- **Custo versus ganho:** 30% mais tempo de treino que o E02 (4,6 contra 3,5 min) e mais memória, sem nenhum
  ganho. Entre as estratégias com ArcFace, nenhuma superou o baseline.
- **Limitações:** uma seed e 5 épocas; com mais dados (todas as ~10 mil identidades do CelebA) o resultado
  poderia mudar, mas isso não cabe no prazo nem no hardware.
