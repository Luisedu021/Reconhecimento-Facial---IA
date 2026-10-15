# E0N — Título do experimento

- **Data:** AAAA-MM-DD
- **Responsável:** _nome_
- **Config:** `configs/…yaml`
- **Commit:** `hash` do código usado

## Hipótese

Uma afirmação testável. Ex.: "O fine-tuning das últimas camadas de um modelo ArcFace produz embeddings
mais discriminativos (menor EER) que o modelo pré-treinado sem ajuste."

## Configuração

| Item | Valor |
|---|---|
| Modelo / backbone | |
| Pesos iniciais (URL oficial + licença) | |
| Estratégia | sem fine-tuning / parcial (camadas treinadas: …) / completo |
| Função de perda | |
| Comparação / classificador | cosseno / euclidiana / k-NN / SVM / protótipo |
| Optimizer / learning rate / scheduler | |
| Batch size / épocas | |
| Transformações / augmentations | |
| Seed | |
| Hardware | ex.: RTX A1000 4 GB, AMP ativado |
| Versões (Python, torch, …) | |
| Tempo de treino / inferência | |

## Dados

| Item | Valor |
|---|---|
| Dataset(s) | |
| Treino / validação / teste | tamanhos e critério de separação (por identidade, por sessão) |
| Checagem de leakage | duplicatas entre splits? identidades em comum entre datasets? threshold escolhido só na validação? |

## Métricas

Escolhidas **antes** de rodar. Ex.: accuracy, F1, ROC-AUC, FAR, FRR, EER, TAR@FAR.

## Resultados

| Métrica | Validação | Teste |
|---|---|---|
| | | |

Gráficos (somente agregados) em `reports/figures/E0N-*.png`: curva ROC, distribuição de similaridade
mesma pessoa × pessoas diferentes, matriz de confusão.

Dataset privado: só números agregados, sem identificar participantes.

## Interpretação

- A hipótese se confirmou? Por quê?
- O que melhorou e o que piorou em relação ao experimento anterior?
- Erros observados (tipos de imagem em que falha).
- Custo computacional versus ganho.
- Limitações.
