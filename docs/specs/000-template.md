# SPEC-NNN — Título curto

- **Status:** rascunho | em revisão | aprovada | implementada
- **Parte:** 1 (experimentação) | 2 (aplicação)
- **Responsáveis:** _nomes_
- **Relacionados:** ADR-NNN, E0N, SPEC-NNN

## Problema

Qual problema esta spec resolve e por que ele importa. Uma ou duas frases.

## Requisitos

### Funcionais

- **RF-1:** O sistema deve …
- **RF-2:** …

### Não funcionais

- **RNF-1:** Desempenho, confiabilidade, usabilidade… (com número sempre que possível, ex.: "< 500 ms por frame em CPU").

### Privacidade (LGPD)

- Quais dados pessoais esta funcionalidade cria, lê, armazena ou apaga?
- Onde ficam e como são excluídos?
- Algo sai da máquina local? (a resposta esperada é "não")

## Fora de escopo

- O que esta spec explicitamente **não** cobre.

## Critérios de aceitação

Cada critério vira pelo menos um teste automatizado.

- **CA-1:** **Dado** … **quando** … **então** …
- **CA-2:** **Dado** uma imagem sem nenhuma face, **quando** … **então** …

## Casos de erro

| Situação | Comportamento esperado |
|---|---|
| _ex.: nenhuma face detectada_ | _ex.: mensagem clara, nada é gravado_ |

## Plano de implementação

Preenchido depois da aprovação dos requisitos (pode ser proposto pelo agente e revisado pelo grupo).

1. …

## Testes

| Critério | Teste | Status |
|---|---|---|
| CA-1 | `tests/…::test_…` | pendente |

## Notas de revisão

Decisões tomadas na revisão do código e pontos em aberto.
