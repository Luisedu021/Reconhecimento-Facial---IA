# Requisitos do sistema

- **Status:** rascunho
- **Última revisão:** 25/10/2026 (requisitos medidos na Parte 1; os da aplicação são fechados na Parte 2)

> Como preencher: escreva cada requisito como "O sistema deve…". Requisito não funcional sempre com
> **número e forma de medir** ("rápido" não é requisito; "≤ 500 ms por imagem na CPU do notebook" é).
> A coluna **Origem** diz de onde veio: um item do [enunciado](enunciado-parte1.md) ou uma decisão do grupo.

## 1. Problema e cenário de uso

_Qual problema o sistema resolve? Em que situação concreta ele seria usado? Ex.: controle de acesso a
um laboratório, registro de presença. Duas ou três frases._

## 2. Escopo

**Dentro do escopo:**
- _…_

**Fora do escopo:**
- Vigilância, monitoramento ou identificação de pessoas que não consentiram (item 26).
- _…_

## 3. Restrições

| Restrição | Valor |
|---|---|
| Prazo | Parte 1: __/__ · Parte 2: __/__ |
| Hardware | notebook com GPU NVIDIA RTX A1000 Laptop (4 GB), Windows 11; treino com AMP e batch ≤ 64 |
| Dados | datasets públicos com licença verificada + dataset privado com consentimento |
| Privacidade | LGPD; dados privados processados só localmente ([privacy.md](privacy.md)) |

## 4. Requisitos funcionais

| ID | Requisito | Prioridade | Parte | Origem | Spec |
|---|---|---|---|---|---|
| RF-01 | _Cadastro de pessoa: O sistema deve…_ | obrigatório | 2 | item 37 | |
| RF-02 | _Reconhecimento a partir de imagem ou câmera: …_ | obrigatório | 2 | item 38 | |
| RF-03 | _Resposta "pessoa desconhecida" abaixo do threshold: …_ | obrigatório | 2 | item 32 | |
| RF-04 | _Exclusão de pessoa e de todos os dados associados: …_ | obrigatório | 2 | item 39 | |
| RF-05 | _Tratamento de nenhuma face / múltiplas faces / imagem inadequada: …_ | obrigatório | 2 | item 37 | |
| RF-06 | _Tratamento de cadastro duplicado: …_ | obrigatório | 2 | item 37 | |
| RF-07 | _Iniciar sem nenhuma pessoa cadastrada após clonar: …_ | obrigatório | 2 | item 43 | |
| RF-08 | _…_ | | | grupo | |

Prioridade: **obrigatório** (exigido pelo enunciado) · **desejável** · **opcional**.

## 5. Requisitos não funcionais

| ID | Categoria | Requisito | Como medir | Origem |
|---|---|---|---|---|
| RNF-01 | Qualidade do reconhecimento | O sistema deve aceitar no máximo 1% das pessoas não cadastradas (FAR ≤ 1%), com o threshold fixado na validação | FAR nos pares de validação do CelebA, onde o threshold é escolhido; depois reportada no LFW e no dataset privado (agregado) | grupo ([ADR-003](decisions/ADR-003-recognition-threshold.md)) |
| RNF-02 | Qualidade do reconhecimento | _FRR máxima aceitável: …_ | | grupo |
| RNF-03 | Desempenho | _Tempo por imagem ou FPS na câmera: …_ | | grupo |
| RNF-04 | Privacidade | O sistema deve processar fotos e embeddings privados só na máquina local, sem enviá-los a serviços externos; embeddings privados ficam só em memória na avaliação | revisão de código + `tests/test_evaluate_private.py` (saída só agregada, recusa de dado individual) | item 23 |
| RNF-05 | Privacidade | O repositório não deve conter fotos, embeddings, bancos, checkpoints derivados de dados privados nem termos | `.gitignore` + hooks do `.pre-commit-config.yaml` bloqueando o commit; checklist do Anexo B em [privacy.md](privacy.md) | itens 18–20 |
| RNF-06 | Reprodutibilidade | Cada experimento deve registrar seed, versões das bibliotecas, hiperparâmetros e hardware, e repetir as métricas com tolerância de 0,5 p.p. | colunas de `reports/results.csv`; `tests/test_reproducibility.py` (CA-7, CA-8) | item 34 |
| RNF-07 | Manutenibilidade | Nenhum caminho ou hiperparâmetro fixo no código: tudo em `configs/*.yaml`; testes automatizados com dados sintéticos | `pytest`: 33 testes passando em 2026-10-25 | item 41 |
| RNF-08 | Usabilidade | _…_ | | grupo |

## 6. Perguntas em aberto

Respostas que ainda faltam para fechar os requisitos acima.

- [ ] Entrada: foto enviada, câmera em tempo real ou as duas?
- [x] Qual erro é pior no cenário: aceitar um desconhecido (FAR) ou rejeitar um cadastrado (FRR)?
  **Aceitar um desconhecido** — por isso FAR ≤ 1% (RNF-01, ADR-003).
- [ ] Quantas pessoas cadastradas e quantas fotos por pessoa no cadastro?
- [ ] Onde a aplicação roda: CPU ou GPU?
- [ ] Interface: web, desktop ou outra?
