# Reconhecimento Facial — Trabalho de IA (UFMS)

> **Status:** em desenvolvimento — Parte 1 (experimentação)

Sistema de reconhecimento facial desenvolvido em duas partes:

- **Parte 1 — Experimentação:** comparar modelos, estratégias de treinamento/fine-tuning, funções de perda
  e métodos de comparação para decidir, com evidências, a melhor configuração.
- **Parte 2 — Aplicação:** usar a melhor configuração da Parte 1 em uma aplicação com cadastro,
  reconhecimento (incluindo "pessoa desconhecida") e exclusão de pessoas.

**Integrantes:** Luís Eduardo Lopes dos Santos, Gustavo Borges Machado

**Vídeos:** Parte 1 — _link_ · Parte 2 — _link_

## Pipeline

```
Imagem → Detecção da face → Alinhamento → Rede neural → Embedding → Comparação → Threshold → Reconhecido / Desconhecido
```

## Estrutura do repositório

```
├── src/                 código-fonte
├── tests/               testes automatizados
├── configs/             configurações dos experimentos e da aplicação
├── experiments/         registro dos experimentos (hipótese, configuração, resultado)
├── reports/figures/     gráficos AGREGADOS (únicos PNG permitidos no repositório)
├── data/README.md       como obter os dados (nenhum dado é versionado)
└── docs/
    ├── enunciado-parte1.md  enunciado da disciplina (fonte dos requisitos)
    ├── requirements.md  requisitos funcionais e não funcionais
    ├── plano.md         cronograma e escopo da Parte 1
    ├── specs/           especificações com critérios de aceitação
    ├── decisions/       ADRs (decisões de arquitetura)
    └── privacy.md       inventário de dados, regras de LGPD e checklist de entrega
```

## Como reproduzir

Ambiente usado: Windows 11, Python 3.10, PyTorch 2.2.2 com CUDA 12.1, GPU NVIDIA RTX A1000 (4 GB).
Todas as versões estão fixadas em [requirements.txt](requirements.txt).

```bash
py -3.10 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
pre-commit install
```

O `pre-commit install` é obrigatório: instala o hook que bloqueia dados privados no commit.

Depois de baixar os datasets ([data/README.md](data/README.md)), os experimentos rodam nesta ordem.
O protocolo completo está na [SPEC-001](docs/specs/001-protocolo-experimental.md).

```bash
# 1. Detecção e recorte das faces (uma vez por dataset)
python -m facerec.preprocess --dataset lfw
python -m facerec.preprocess --dataset celeba
python -m facerec.overlap                     # remove quase-duplicatas CelebA x LFW

# 2. Experimentos públicos (validação no CelebA, teste no LFW)
python -m facerec.evaluate --config configs/e01.yaml
python -m facerec.train    --config configs/e02.yaml
python -m facerec.evaluate --config configs/e02.yaml
python -m facerec.train    --config configs/e03.yaml
python -m facerec.evaluate --config configs/e03.yaml
python -m facerec.train    --config configs/e05.yaml   # opcional na SPEC-001; vencedor (ADR-001)
python -m facerec.evaluate --config configs/e05.yaml

# 3. Dataset privado (local, saída só agregada)
python -m facerec.preprocess --dataset private
python -m facerec.evaluate_private --config configs/e04.yaml

# Testes (dados sintéticos)
pytest
```

As métricas vão para `reports/results.csv`, os thresholds para `reports/thresholds.json` e os gráficos
agregados para `reports/figures/`.

## Datasets

Nenhum dataset é versionado neste repositório. Instruções de obtenção em [data/README.md](data/README.md).

| Dataset | Fonte oficial | Finalidade no projeto | Licença / termos | Como foi usado |
|---|---|---|---|---|
| LFW — Labeled Faces in the Wild (Huang et al., UMass Amherst, Tech. Report 07-49, 2007) | http://vis-www.cs.umass.edu/lfw/ (fora do ar em out/2026; obtido do espelho do figshare usado pelo scikit-learn, `lfw.tgz` original, 180.566.744 bytes) | **teste** público final (nunca usado para escolher modelo ou threshold) | sem licença formal; publicado para pesquisa em verificação facial; o site avisa que o LFW não serve para validar algoritmos comerciais e tem pouca diversidade (poucas crianças, idosos e mulheres); exige citar o relatório técnico | 13.233 imagens, protocolo View 2 (`pairs.txt`: 6.000 pares em 10 folds) e 500 rostos como "desconhecidos" no E04; não redistribuído |
| CelebA (Liu et al., *Deep Learning Face Attributes in the Wild*, ICCV 2015) | https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html | **treino** do fine-tuning e **validação** (escolha de modelo, época e threshold) | somente pesquisa não comercial; proibido copiar, publicar ou redistribuir qualquer parte; identidades liberadas só para pesquisa | `img_align_celeba` + `identity_CelebA.txt`; subconjunto de 1.000 identidades × 20 fotos (seed 42): 900 para treino, 100 para validação; 26 imagens sem rosto e 6 quase-duplicatas do LFW removidas; não redistribuído |
| Dataset privado do grupo | coletado pelos integrantes, com consentimento | avaliação em condições reais | privado — não publicado (ver [docs/privacy.md](docs/privacy.md)) | 5 adultos com termo assinado, 5 selfies cada (25 fotos): S1 (2 fotos, cadastro) e S2 (3 fotos, outro dia, consulta); IDs pseudônimos P01–P05; resultados só agregados |

Modelos pré-treinados são referenciados pela URL oficial; a licença dos **pesos** (e o dataset em que foram treinados)
está registrada no ADR correspondente.

## Resultados

Modelo base em todos: InceptionResnetV1 pré-treinado no VGGFace2 ([ADR-001](docs/decisions/ADR-001-face-model.md)).
Accuracy e F1 no threshold escolhido na validação (FAR = 1% em CelebA-val, [ADR-003](docs/decisions/ADR-003-recognition-threshold.md)).
Seed 42, RTX A1000 4 GB. Dados completos em [reports/results.csv](reports/results.csv).

| Experimento | Estratégia / perda | Dataset | Accuracy | F1 | EER | TAR@FAR1% | Observação |
|---|---|---|---|---|---|---|---|
| [E01](experiments/E01-baseline.md) | sem fine-tuning | CelebA-val | 97,08% | 0,9703 | 2,82% | 95,17% | baseline |
| | | LFW | 99,33% (10 folds: 99,27% ± 0,32%) | 0,9933 | 0,65% | 99,47% | FAR 0,67% no LFW |
| [E02](experiments/E02-finetuning-parcial.md) | parcial / ArcFace | CelebA-val | 97,08% | 0,9703 | 2,82% | 95,17% | treino piorou a validação; melhor época = aquecimento (= E01) |
| | | LFW | 99,33% (10 folds: 99,27% ± 0,32%) | 0,9933 | 0,65% | 99,47% | treino 3,5 min |
| [E03](experiments/E03-finetuning-completo.md) | completo / ArcFace | CelebA-val | 97,08% | 0,9703 | 2,82% | 95,17% | piorou mais que o E02 (EER 3,37% na época 5); melhor época = aquecimento |
| | | LFW | 99,33% (10 folds: 99,27% ± 0,32%) | 0,9933 | 0,65% | 99,47% | treino 4,6 min |
| **[E05](experiments/E05-triplet-loss.md)** | parcial / Triplet semi-hard | CelebA-val | 97,13% | 0,9708 | **2,57%** | 95,27% | **vencedor pela validação** |
| | | LFW | 99,05% (10 folds: 99,17% ± 0,38%) | 0,9905 | 0,83% | 99,30% | FAR 1,40% no LFW (acima do alvo); treino 0,9 min |
| [E04](experiments/E04-dataset-privado.md) | E05 + protótipo | privado (agregado, 5 pessoas) | rank-1 100% | — | 0% | — | FRR 0%; rejeição de desconhecidos 64% (rodízio) e 97,0% (LFW) |
| | E05 + k-NN (k = 1) | privado (agregado, 5 pessoas) | rank-1 100% | — | 0% | — | FRR 0%; rejeição de desconhecidos 60% (rodízio) e 95,6% (LFW) |

**Conclusão:** o modelo pré-treinado já é forte (99,27% no LFW). O fine-tuning com ArcFace em 900 identidades
causou overfitting e não superou o baseline. A Triplet loss, de ajuste mais suave, teve o menor EER na validação
e foi escolhida pelo critério definido antes dos experimentos. A vantagem é pequena (0,25 p.p.) e não se
confirmou no LFW, o que está registrado como limitação no [ADR-001](docs/decisions/ADR-001-face-model.md).
Nas selfies do grupo (E04), o modelo reconheceu 100% dos cadastrados e separou perfeitamente as pessoas
(AUC = 1,0), mas o threshold escolhido em fotos de celebridades deixou passar 36–40% dos participantes fora do
cadastro: a mudança de domínio prevista no [ADR-003](docs/decisions/ADR-003-recognition-threshold.md). O threshold
não foi reajustado com dados de teste; a revisão, na validação, fica para a Parte 2.

Resultados do dataset privado são apresentados apenas de forma agregada, sem identificar participantes.

## Privacidade e LGPD

Este projeto trata dados biométricos (dados pessoais sensíveis — LGPD, art. 5º, II).
Fotografias, embeddings, bancos de cadastro, checkpoints derivados de dados privados e termos de consentimento
**nunca** são publicados. Detalhes em [docs/privacy.md](docs/privacy.md).

Ao clonar o projeto, a aplicação começa sem nenhuma pessoa cadastrada; cada usuário cria seu próprio banco local.

## Uso de agentes de IA

O desenvolvimento usa agentes de IA (Claude Code) dentro de um processo de Spec-Driven Development.
As regras dadas aos agentes estão em [CLAUDE.md](CLAUDE.md); o acesso deles aos dados privados é bloqueado
em [.claude/settings.json](.claude/settings.json).
