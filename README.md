# Reconhecimento Facial — Trabalho de IA (UFMS)

> **Status:** em desenvolvimento — Parte 1 (experimentação)

Sistema de reconhecimento facial desenvolvido em duas partes:

- **Parte 1 — Experimentação:** comparar modelos, estratégias de treinamento/fine-tuning, funções de perda
  e métodos de comparação para decidir, com evidências, a melhor configuração.
- **Parte 2 — Aplicação:** usar a melhor configuração da Parte 1 em uma aplicação com cadastro,
  reconhecimento (incluindo "pessoa desconhecida") e exclusão de pessoas.

**Integrantes:** _a preencher_

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
    ├── specs/           especificações com critérios de aceitação
    ├── decisions/       ADRs (decisões de arquitetura)
    └── privacy.md       inventário de dados, regras de LGPD e checklist de entrega
```

## Como reproduzir

> _A preencher quando o ambiente estiver definido (versão do Python, gerenciador de dependências, comandos)._

Configuração obrigatória após clonar (instala o hook que bloqueia dados privados no commit):

```bash
pip install pre-commit
pre-commit install
```

## Datasets

Nenhum dataset é versionado neste repositório. Instruções de obtenção em [data/README.md](data/README.md).

| Dataset | Fonte oficial | Finalidade no projeto | Licença / termos | Como foi usado |
|---|---|---|---|---|
| LFW — Labeled Faces in the Wild | http://vis-www.cs.umass.edu/lfw/ | _a preencher_ | _a verificar na fonte_ | _a preencher_ |
| CelebA | https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html | _a preencher_ | uso acadêmico não comercial; redistribuição restrita (_confirmar nos termos_) | _a preencher_ |
| Dataset privado do grupo | coletado pelos integrantes, com consentimento | avaliação em condições reais | privado — não publicado (ver [docs/privacy.md](docs/privacy.md)) | _a preencher_ |

Modelos pré-treinados são referenciados pela URL oficial; a licença dos **pesos** (e o dataset em que foram treinados)
está registrada no ADR correspondente.

## Resultados

| Experimento | Modelo | Estratégia | Dataset | Accuracy | F1 | EER | Observação |
|---|---|---|---|---|---|---|---|
| E01 | | | | | | | |

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
