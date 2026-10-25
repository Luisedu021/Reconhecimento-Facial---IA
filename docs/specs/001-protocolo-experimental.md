# SPEC-001 — Protocolo experimental da Parte 1

- **Status:** implementada (E01–E06 executados em 2026-10-25)
- **Parte:** 1 (experimentação)
- **Responsáveis:** Luís Eduardo Lopes dos Santos, Gustavo Machado Borges Daniel
- **Relacionados:** [ADR-001](../decisions/ADR-001-face-model.md) (modelo), [ADR-003](../decisions/ADR-003-recognition-threshold.md) (threshold), E01–E05, [requirements.md](../requirements.md)

## Problema

Descobrir, com evidência e sem data leakage, qual configuração de modelo e de fine-tuning reconhece melhor
as pessoas cadastradas e rejeita as desconhecidas (itens 6 e 32 do enunciado).

## Requisitos

### Funcionais

- **RF-1:** Um único pipeline para todos os dados: detecção (MTCNN) → alinhamento e recorte 160×160 →
  InceptionResnetV1 → embedding de 512 dimensões normalizado (L2) → comparação por cosseno.
- **RF-2:** A detecção e o recorte rodam uma vez por imagem e ficam em cache: públicos em `data/public/cache/`,
  privados em `data/private/cache/`.
- **RF-3:** Cada experimento é definido por um arquivo YAML em `configs/`; os scripts de treino e avaliação
  recebem apenas esse arquivo.
- **RF-4:** A avaliação grava as métricas em `reports/results.csv` (uma linha por experimento e conjunto)
  e os gráficos em `reports/figures/`.
- **RF-5:** A avaliação do dataset privado produz apenas números agregados.
- **RF-6:** O teste de "desconhecido" no dataset privado usa rodízio: em cada rodada, 1 das 5 pessoas fica
  fora do cadastro; são 5 rodadas.

### Não funcionais

- **RNF-1 (reprodutibilidade):** seed fixa (42) e registro de versões das bibliotecas, hiperparâmetros
  e hardware em cada linha de `results.csv`.
- **RNF-2 (hardware):** todo experimento roda na RTX A1000 de 4 GB, com AMP e batch ≤ 64.
- **RNF-3 (privacidade):** dados privados são processados só localmente; nenhum artefato derivado deles
  é versionado ([privacy.md](../privacy.md)).

### Privacidade (LGPD)

- Cria: recortes e embeddings das fotos privadas em `data/private/`; o k-NN ajustado no cadastro privado
  também é privado.
- Nada sai da máquina; as métricas privadas publicadas são agregadas.
- Exclusão de uma pessoa: apagar `data/private/<ID>/` e o cache correspondente.

## Dados e divisão

| Conjunto | Papel | Divisão |
|---|---|---|
| CelebA, subconjunto de ~1.000 identidades | treino do fine-tuning | 90% das identidades para treino |
| CelebA, identidades restantes (~10%) | **validação**: escolha de hiperparâmetros, do melhor modelo e do threshold | pares mesma pessoa × pessoas diferentes gerados com seed fixa |
| LFW, View 2 (10 folds de `pairs.txt`) | **teste** público, resultado final | protocolo oficial |
| Dataset privado: 5 pessoas × 5 fotos | **teste** em condições reais | S1 (2 fotos) = cadastro; S2 (3 fotos, outro dia) = consulta |

Nenhuma escolha (modelo, hiperparâmetro, threshold) usa o LFW nem o dataset privado.

## Experimentos

| ID | Configuração | Validação | Teste |
|---|---|---|---|
| E01 | InceptionResnetV1 pré-treinado (VGGFace2), sem fine-tuning (baseline) | CelebA-val | LFW |
| E02 | Fine-tuning parcial: só o último bloco e a camada de embedding treinam; ArcFace loss | CelebA-val | LFW |
| E03 | Fine-tuning completo: toda a rede treina, learning rate menor no backbone; ArcFace loss | CelebA-val | LFW |
| E04 | Melhor modelo de E01–E03, E05 e E06 pela validação: protótipo (média do cadastro) + cosseno × k-NN (k = 1); threshold de CelebA-val | — | privado (agregado) |
| E05 | _Opcional (executado):_ E02 com Triplet loss (mineração semi-hard) | CelebA-val | LFW |

O E05 usa o mesmo protocolo e o mesmo critério de escolha, por isso concorre com E01–E03. Resultado: o E05
teve o menor EER em CelebA-val e é o modelo do E04 ([ADR-001](../decisions/ADR-001-face-model.md)).

**Adendo de 2026-10-25 (registrado antes de rodar):**

| ID | Configuração | Validação | Teste |
|---|---|---|---|
| E06 | E05 com 15 épocas em vez de 5 (a curva de validação do E05 ainda descia na época 5) | CelebA-val | LFW |

- Hipótese do E06: mais épocas de Triplet loss reduzem o EER em CelebA-val abaixo dos 2,57% do E05.
- O E06 concorre com o mesmo critério (menor EER em CelebA-val). Se vencer, passa a ser o modelo do E04, com o
  threshold dele, também escolhido na validação.
- **Análise de sensibilidade do threshold (E04):** além do threshold oficial, o E04 passa a reportar as taxas
  agregadas (FRR, FAR no rodízio, FAR com rostos do LFW) para uma grade fixa de thresholds definida na config
  (0,30 a 0,80, passo 0,05). É só análise do impacto do threshold (item 33): **nenhum valor é escolhido** com
  esses dados (CA-5).
- **Resultado:** o E06 teve EER de 2,38% em CelebA-val (melhor época 9), superou o E05 e passou a ser o modelo do
  E04 ([ADR-001](../decisions/ADR-001-face-model.md)). A sensibilidade foi gravada em `reports/E04-sensibilidade.csv`.

Hiperparâmetros iniciais: AdamW, learning rate 1e-3 na cabeça ArcFace (pesos novos), 1e-4 nas camadas finais
e 1e-5 no restante do backbone (E03); batch 64 em todos (medido: 0,33 GB no E02 e ~1 GB no E03); 5 épocas, sendo a primeira
de aquecimento (só a cabeça ArcFace treina); AMP; augmentations só no treino (flip horizontal e leve
variação de cor). A melhor época é a de menor EER na validação. Valores finais em `configs/`.

## Métricas

- **Verificação (CelebA-val, LFW):** accuracy, precision, recall, F1, matriz de confusão (pares), ROC-AUC, EER e TAR @ FAR = 1%; curva ROC e histograma
  de similaridade mesma pessoa × pessoas diferentes.
- **Identificação (privado):** acerto rank-1 das pessoas cadastradas; taxa de rejeição correta das
  desconhecidas; FAR e FRR no threshold escolhido.
- **Custo:** tempo de treino, tempo por imagem e número de parâmetros treináveis.

## Critérios de aceitação

- **CA-1:** **Dado** o split do CelebA, **quando** ele é gerado, **então** nenhuma identidade aparece em treino e validação ao mesmo tempo.
- **CA-2:** **Dado** qualquer split, **quando** ele é gerado, **então** nenhum arquivo de imagem aparece em dois conjuntos.
- **CA-3:** **Dado** o pipeline de dados, **quando** monta validação ou teste, **então** nenhuma augmentation é aplicada.
- **CA-4:** **Dado** o subconjunto do CelebA e o LFW, **quando** a checagem de sobreposição roda, **então** os pares com cosseno > 0,9 entre os dois são listados e removidos do treino.
- **CA-5:** **Dado** o script de avaliação privada, **quando** ele roda, **então** recebe o threshold como parâmetro e não o calcula com dados privados.
- **CA-6:** **Dado** o script de avaliação privada, **quando** gera a saída, **então** ela não contém IDs, nomes, caminhos de arquivo nem embeddings.
- **CA-7:** **Dado** o mesmo YAML e a mesma seed, **quando** um experimento roda duas vezes, **então** as métricas coincidem (tolerância de 0,5 ponto percentual com AMP).
- **CA-8:** **Dado** um experimento concluído, **quando** `results.csv` é lido, **então** a linha tem seed, versões, hiperparâmetros e hardware.
- **CA-9:** **Dado** a análise de sensibilidade do E04, **quando** ela roda, **então** cada linha tem só o threshold da grade e taxas agregadas, as taxas variam de forma monotônica com o threshold, e nenhum threshold é escolhido a partir delas.

## Plano de implementação

1. `src/facerec/preprocess.py`: detecção, alinhamento e cache dos recortes.
2. `src/facerec/data.py`: subconjunto e split do CelebA por identidade, pares do LFW, leitura do dataset privado por sessão.
3. `src/facerec/model.py`: backbone e estratégias de congelamento (nenhum, parcial, completo).
4. `src/facerec/train.py`: fine-tuning com ArcFace ou Triplet loss, AMP e seed.
5. `src/facerec/metrics.py`: ROC, EER, TAR @ FAR e escolha do threshold.
6. `src/facerec/evaluate.py`: avaliação pública, gráficos e `results.csv`.
7. `src/facerec/evaluate_private.py`: avaliação privada agregada.
8. `tests/`: testes com dados sintéticos para cada critério de aceitação.

## Testes

| Critério | Teste | Status |
|---|---|---|
| CA-1, CA-2 | `tests/test_data.py::test_split_sem_sobreposicao` | passando |
| CA-3 | `tests/test_data.py::test_sem_augmentation_fora_do_treino` | passando |
| CA-4 | `tests/test_data.py::test_remove_sobreposicao_celeba_lfw` | passando (execução real: 6 de 19.974 imagens removidas) |
| CA-5, CA-6 | `tests/test_evaluate_private.py::test_saida_agregada` | passando |
| CA-7, CA-8 | `tests/test_reproducibility.py` | passando |
| RF-2 (lote com imagem sem rosto) | `tests/test_preprocess.py::test_lote_com_imagem_sem_rosto_nao_quebra` | passando |
| CA-9 | `tests/test_evaluate_private.py::test_sensibilidade_agregada_e_monotonica` | passando |

Execução de 2026-10-25: 37 testes passando (`pytest`). CA-7 também conferido com dados reais: a reexecução das
avaliações de E01, E02, E03 e E05 reproduziu exatamente as mesmas métricas.

## Em aberto

- [x] FAR-alvo para escolher o threshold: FAR ≤ 1% na validação, decidido no [ADR-003](../decisions/ADR-003-recognition-threshold.md).
- [x] Licença do código (MIT) e dos pesos VGGFace2 do `facenet-pytorch` registradas no [ADR-001](../decisions/ADR-001-face-model.md).
