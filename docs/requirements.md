# Requisitos do sistema

- **Status:** rascunho
- **Última revisão:** 25/10/2026 (requisitos medidos na Parte 1; os da aplicação são fechados na Parte 2)

> Como preencher: escreva cada requisito como "O sistema deve…". Requisito não funcional sempre com
> **número e forma de medir** ("rápido" não é requisito; "≤ 500 ms por imagem na CPU do notebook" é).
> A coluna **Origem** diz de onde veio: um item do [enunciado](enunciado-parte1.md) ou uma decisão do grupo.

## 1. Problema e cenário de uso

O sistema identifica, a partir de uma foto do rosto, se a pessoa é um dos membros cadastrados de um grupo pequeno
e consentido, ou se é desconhecida. O cenário de referência é o **controle de acesso a uma sala ou laboratório**:
quem está cadastrado é reconhecido; quem não está deve ser recusado. Por isso o erro mais grave é aceitar um
desconhecido (FAR), e não pedir a um membro que tente de novo (FRR), como decidido no
[ADR-003](decisions/ADR-003-recognition-threshold.md).

## 2. Escopo

**Dentro do escopo:**
- Parte 1: comparar, com evidência experimental, modelo pré-treinado, fine-tuning parcial e completo, funções de
  perda (ArcFace, Triplet) e métodos de comparação (cosseno, protótipo, k-NN), e escolher o threshold
  ([SPEC-001](specs/001-protocolo-experimental.md)).
- Parte 2: cadastro de pessoas com consentimento, reconhecimento com resposta "desconhecido" e exclusão de uma
  pessoa com todos os dados associados.
- Apenas adultos com termo de consentimento assinado (itens 14 e 16).

**Fora do escopo:**
- Vigilância, monitoramento ou identificação de pessoas que não consentiram (item 26).
- Uso comercial (as licenças do CelebA e dos pesos VGGFace2 são acadêmicas, [ADR-001](decisions/ADR-001-face-model.md)).
- Detecção de ataques de apresentação (foto de foto, vídeo, máscara): o sistema não verifica se o rosto é real.
- Grupos grandes: o protocolo foi avaliado com 5 participantes.

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
| RF-01 | O sistema deve cadastrar uma pessoa a partir de fotos do rosto, gerando os embeddings com o mesmo pré-processamento da Parte 1 e guardando-os só localmente, ligados a um ID pseudônimo | obrigatório | 2 | item 37 | Parte 2 |
| RF-02 | O sistema deve reconhecer uma pessoa a partir de uma imagem (e, se houver tempo, da câmera), devolvendo o cadastrado mais parecido | obrigatório | 2 | item 38 | Parte 2 |
| RF-03 | O sistema deve responder "pessoa desconhecida" quando a maior similaridade ficar abaixo do threshold do [ADR-003](decisions/ADR-003-recognition-threshold.md) | obrigatório | 2 | item 32 | Parte 2; protocolo testado no E04 |
| RF-04 | O sistema deve excluir uma pessoa e todos os dados associados (fotos, embeddings, cadastro, cache) | obrigatório | 2 | item 39 | Parte 2; procedimento manual em [privacy.md](privacy.md) |
| RF-05 | O sistema deve recusar imagens sem rosto, com mais de um rosto ou inadequadas, com mensagem ao usuário | obrigatório | 2 | item 37 | Parte 2 |
| RF-06 | O sistema deve detectar um cadastro que já existe (rosto muito parecido com alguém cadastrado) e pedir confirmação | obrigatório | 2 | item 37 | Parte 2 |
| RF-07 | O sistema deve iniciar sem nenhuma pessoa cadastrada depois de um `git clone` | obrigatório | 2 | item 43 | Parte 2 |

Prioridade: **obrigatório** (exigido pelo enunciado) · **desejável** · **opcional**.

## 5. Requisitos não funcionais

| ID | Categoria | Requisito | Como medir | Origem |
|---|---|---|---|---|
| RNF-01 | Qualidade do reconhecimento | O sistema deve aceitar no máximo 1% das pessoas não cadastradas (FAR ≤ 1%), com o threshold fixado na validação | FAR nos pares de validação do CelebA, onde o threshold é escolhido; depois reportada no LFW e no dataset privado (agregado) | grupo ([ADR-003](decisions/ADR-003-recognition-threshold.md)) |
| RNF-02 | Qualidade do reconhecimento | Com o threshold do RNF-01, o sistema deve rejeitar por engano no máximo 5% das tentativas de pessoas cadastradas (FRR ≤ 5%) | FRR nos pares de validação do CelebA no mesmo threshold. Medido na Parte 1: 3,73% (E06); no dataset privado, 0% (E04, agregado) | grupo (ADR-003) |
| RNF-03 | Desempenho | _Tempo por imagem ou FPS na câmera: definido na Parte 2. Referência medida na Parte 1: 2,9 ms por rosto já recortado para gerar o embedding na RTX A1000 (E06), sem contar a detecção_ | | grupo |
| RNF-04 | Privacidade | O sistema deve processar fotos e embeddings privados só na máquina local, sem enviá-los a serviços externos; embeddings privados ficam só em memória na avaliação | revisão de código + `tests/test_evaluate_private.py` (saída só agregada, recusa de dado individual) | item 23 |
| RNF-05 | Privacidade | O repositório não deve conter fotos, embeddings, bancos, checkpoints derivados de dados privados nem termos | `.gitignore` + hooks do `.pre-commit-config.yaml` bloqueando o commit; checklist do Anexo B em [privacy.md](privacy.md) | itens 18–20 |
| RNF-06 | Reprodutibilidade | Cada experimento deve registrar seed, versões das bibliotecas, hiperparâmetros e hardware, e repetir as métricas com tolerância de 0,5 p.p. | colunas de `reports/results.csv`; `tests/test_reproducibility.py` (CA-7, CA-8). Verificado em 2026-10-25: a reexecução das avaliações de E01, E02, E03 e E05 reproduziu exatamente as mesmas métricas, e as épocas 1–5 do E06 repetiram as do E05 | item 34 |
| RNF-07 | Manutenibilidade | Nenhum caminho ou hiperparâmetro fixo no código: tudo em `configs/*.yaml`; testes automatizados com dados sintéticos | `pytest`: 37 testes passando em 2026-10-25 | item 41 |
| RNF-08 | Usabilidade | _Definido na Parte 2, junto com a interface (pergunta em aberto abaixo)_ | | grupo |

## 6. Perguntas em aberto

Respostas que ainda faltam para fechar os requisitos acima.

- [ ] Entrada: foto enviada, câmera em tempo real ou as duas?
- [x] Qual erro é pior no cenário: aceitar um desconhecido (FAR) ou rejeitar um cadastrado (FRR)?
  **Aceitar um desconhecido** — por isso FAR ≤ 1% (RNF-01, ADR-003).
- [ ] Quantas pessoas cadastradas e quantas fotos por pessoa no cadastro? _Na Parte 1 o protocolo usou 5 pessoas
  e 2 fotos de cadastro por pessoa (E04); o número da aplicação é decidido na Parte 2._
- [ ] Onde a aplicação roda: CPU ou GPU?
- [ ] Interface: web, desktop ou outra?
