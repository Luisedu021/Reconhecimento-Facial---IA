# E03 — Fine-tuning completo com ArcFace loss

- **Data:** _AAAA-MM-DD_
- **Responsável:** _nome_
- **Config:** [`configs/e03.yaml`](../configs/e03.yaml)
- **Commit:** _hash do código usado_

## Hipótese

Atualizar a rede inteira (learning rate 10× menor no backbone) adapta melhor ao domínio que o fine-tuning
parcial (E02), mas com mais custo e mais risco de overfitting.

## Configuração

Igual ao [E02](E02-finetuning-parcial.md), exceto:

| Item | Valor |
|---|---|
| Estratégia | completa: todos os parâmetros treinam |
| Learning rate | 1e-3 na cabeça ArcFace, 1e-4 nas camadas finais, 1e-5 no restante do backbone |

Batch, épocas, augmentations e seed são os mesmos do E02: só a estratégia muda, para isolar o efeito dela.

## Dados

Mesmos do E02.

## Resultados

| Métrica | CelebA-val | LFW |
|---|---|---|
| Accuracy (10 folds) | — | |
| EER | | |
| TAR @ FAR = 1% | | |
| FAR / FRR no threshold da validação | | |
| Tempo de treino / parâmetros treináveis | | |

## Interpretação

_A preencher: o ganho (ou perda) em relação ao E02 compensa o custo maior?_
