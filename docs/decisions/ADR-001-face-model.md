# ADR-001 — Modelo de reconhecimento facial

- **Status:** aceita
- **Data:** 2026-10-24 (família do modelo) · 2026-10-25 (configuração final, depois de E01–E05)
- **Decisores:** Luís Eduardo Lopes dos Santos, Gustavo Borges Machado
- **Relacionados:** [SPEC-001](../specs/001-protocolo-experimental.md), [E01](../../experiments/E01-baseline.md), [E02](../../experiments/E02-finetuning-parcial.md), [E03](../../experiments/E03-finetuning-completo.md), [E05](../../experiments/E05-triplet-loss.md), [enunciado](../enunciado-parte1.md) itens 7, 19 e 29

## Contexto

O item 7 do enunciado pede uma escolha de modelo tecnicamente justificada. As restrições do projeto são:

- **Hardware:** uma GPU NVIDIA RTX A1000 de 4 GB, no Windows.
- **Prazo:** cerca de 10 dias para a Parte 1.
- **Pipeline completo:** o modelo precisa vir com detecção e alinhamento, e permitir fine-tuning parcial e completo em PyTorch.
- **LGPD:** dados privados processados só localmente. O ideal é que nenhum checkpoint derive das fotos dos participantes (item 19).

## Alternativas consideradas

| Alternativa | Prós | Contras |
|---|---|---|
| **A. FaceNet (InceptionResnetV1) + MTCNN, via `facenet-pytorch`** | Detecção, alinhamento e modelo numa biblioteca só, em PyTorch puro; pesos públicos pré-treinados no VGGFace2; fine-tuning direto (são módulos PyTorch comuns) | Arquitetura de 2015, menos precisa que modelos ArcFace recentes; fixa versões antigas (torch 2.2, numpy < 2) |
| B. ArcFace (IResNet) via `insightface` | Estado da arte em benchmarks; detector RetinaFace incluído | Modelos distribuídos em ONNX (fine-tuning exige outros pesos em PyTorch); instalação no Windows pede compilação; pesos só para pesquisa não comercial |
| C. MobileFaceNet | Leve e rápido, bom para câmera em tempo real (Parte 2) | Poucos pesos pré-treinados prontos e confiáveis em PyTorch; precisão menor |
| D. Vision Transformer para faces | Arquitetura moderna | Pesado para 4 GB; poucos pesos públicos de faces; treino mais lento |

## Decisão

Usar a **alternativa A** como família de modelo em todos os experimentos. A configuração final é a de **menor EER
na validação (CelebA-val)** entre os experimentos públicos. O LFW e o dataset privado não participam da escolha
(item 29).

**Configuração escolhida: E05**, fine-tuning parcial (`block8`, `last_linear`, `last_bn`) com Triplet loss e
mineração semi-hard, checkpoint da época 5, **threshold 0,450** (FAR = 1% em CelebA-val, [ADR-003](ADR-003-recognition-threshold.md)).
É a configuração usada no E04 (`configs/e04.yaml`) e na Parte 2.

- O E05 estava previsto na SPEC-001 como opcional e foi avaliado com o mesmo protocolo e o mesmo critério dos
  demais, por isso concorre em igualdade com E01–E03.
- E02 e E03 (ArcFace) pioraram a validação a cada época de treino; o checkpoint escolhido de ambos foi o da
  época de aquecimento, idêntico ao E01.

## Evidências

| Evidência | Valor |
|---|---|
| Memória de treino medida (dados aleatórios, AMP) | fine-tuning parcial com batch 64: 0,33 GB; completo com batch 32: 0,96 GB |
| Tempo por passo de treino | ~0,28 s (parcial, batch 64); ~0,23 s (completo, batch 32) |
| Testes do pipeline | 33 testes com dados sintéticos passando |
| EER em CelebA-val: E01 / E02 / E03 / **E05** | 2,82% / 2,82% / 2,82% / **2,57%** |
| TAR @ FAR = 1% em CelebA-val: E01 / E02 / E03 / E05 | 95,17% / 95,17% / 95,17% / 95,27% |
| Acurácia no LFW (10 folds): E01 / E02 / E03 / E05 | 99,27% / 99,27% / 99,27% / 99,17% (± 0,32–0,38%) |
| FAR / FRR no LFW com o threshold da validação: E01 / E05 | 0,67% / 0,67% · 1,40% / 0,50% |
| Tempo de treino: E02 / E03 / E05 | 3,5 min / 4,6 min / 0,9 min |

Fonte: `reports/results.csv` e `reports/E0x-historico.csv`, gerados em 2026-10-25 (seed 42, RTX A1000).

**Ressalvas registradas junto com a decisão:**

- A vantagem do E05 na validação é pequena (0,25 p.p. de EER), da ordem do erro amostral com 3.000 pares de
  cada tipo (≈ 0,3 p.p.). Seguimos o critério definido antes dos experimentos, sem trocá-lo depois de ver o
  resultado.
- No teste, o E05 não confirmou o ganho: acurácia no LFW 0,1 p.p. abaixo do E01 (dentro do desvio entre folds)
  e FAR de 1,40% com o threshold da validação, acima do alvo de 1% (RNF-01). Pelo item 29, esse resultado de
  teste é reportado, mas não muda a escolha. O E04 vai medir se o threshold 0,450 funciona nas selfies reais.

## Licenças

- **Código:** `facenet-pytorch` ([repositório oficial](https://github.com/timesler/facenet-pytorch)): **licença MIT**
  (Copyright (c) 2019 Timothy Esler), conforme o `LICENSE.md` do repositório, consultado em 2026-10-25.
  Permite uso, modificação e redistribuição mantendo o aviso de copyright.
- **Pesos `vggface2`:** baixados pela própria biblioteca e não versionados. Foram treinados no dataset VGGFace2
  (Cao et al., *VGGFace2: A dataset for recognising faces across pose and age*, FG 2018). A página oficial do
  dataset ([VGG, Oxford](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/)) informa que ele **não está mais
  disponível para download** e não traz texto de licença. Por prudência, tratamos os pesos como de **uso
  acadêmico e não comercial**, que é o caso deste trabalho, e não os redistribuímos.
- **CelebA** (fine-tuning do E05): somente pesquisa não comercial, sem redistribuição (termos na
  [página oficial](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)). O checkpoint do E05 deriva do CelebA e
  por isso também não é versionado.

## Consequências

- **Positivas:**
  - Um só código para o treino, o cadastro e o reconhecimento: o mesmo recorte de 160×160 em toda parte.
  - Cabe com folga na GPU, com épocas de poucos minutos.
- **Negativas e riscos:**
  - O modelo base é mais antigo que o estado da arte.
  - O LFW está perto da saturação, então as diferenças entre experimentos podem ser pequenas. Por isso também se reportam o EER e a TAR @ FAR = 1%.
  - A Parte 2 depende de um checkpoint local (`checkpoints/E05.pt`), que precisa ser regerado com
    `python -m facerec.train --config configs/e05.yaml` em cada máquina (menos de 1 minuto na RTX A1000).
  - O FAR de 1,40% do E05 no LFW indica que o threshold pode precisar de revisão na Parte 2; qualquer revisão
    será feita na validação e registrada no ADR-003.
- **Privacidade:** o fine-tuning usa só o CelebA, que é público. Nenhum checkpoint deriva das fotos dos participantes, o que atende ao item 19 por construção. Os checkpoints não são versionados mesmo assim (tamanho e licença dos pesos de origem).
