# ADR-001 — Modelo de reconhecimento facial

- **Status:** proposta (família do modelo decidida; configuração final depende de E01–E03)
- **Data:** 2026-10-24
- **Decisores:** Luís Eduardo Lopes dos Santos, Gustavo Borges Machado
- **Relacionados:** [SPEC-001](../specs/001-protocolo-experimental.md), E01, E02, E03, [enunciado](../enunciado-parte1.md) itens 7 e 19

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
na validação (CelebA-val)** entre E01, E02 e E03: _a preencher depois dos experimentos_. O LFW e o dataset privado
não participam da escolha (item 29).

## Evidências

| Evidência | Valor |
|---|---|
| Memória de treino medida (dados aleatórios, AMP) | fine-tuning parcial com batch 64: 0,33 GB; completo com batch 32: 0,96 GB |
| Tempo por passo de treino | ~0,28 s (parcial, batch 64); ~0,23 s (completo, batch 32) |
| Testes do pipeline | 27 testes com dados sintéticos passando |
| EER em CelebA-val: E01 / E02 / E03 | _a preencher_ |
| Acurácia no LFW (10 folds): E01 / E02 / E03 | _a preencher_ |

## Licenças

- **Código:** `facenet-pytorch` ([repositório oficial](https://github.com/timesler/facenet-pytorch)) — _confirmar a licença no LICENSE do repositório._
- **Pesos `vggface2`:** baixados pela própria biblioteca e não versionados. Foram treinados no dataset VGGFace2: _confirmar os termos de uso do VGGFace2 (finalidade acadêmica e não comercial) e registrar aqui._

## Consequências

- **Positivas:**
  - Um só código para o treino, o cadastro e o reconhecimento: o mesmo recorte de 160×160 em toda parte.
  - Cabe com folga na GPU, com épocas de poucos minutos.
- **Negativas e riscos:**
  - O modelo base é mais antigo que o estado da arte.
  - O LFW está perto da saturação, então as diferenças entre experimentos podem ser pequenas. Por isso também se reportam o EER e a TAR @ FAR = 1%.
- **Privacidade:** o fine-tuning usa só o CelebA, que é público. Nenhum checkpoint deriva das fotos dos participantes, o que atende ao item 19 por construção. Os checkpoints não são versionados mesmo assim (tamanho e licença dos pesos de origem).
