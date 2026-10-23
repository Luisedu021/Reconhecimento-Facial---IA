# CLAUDE.md — instruções para agentes de IA neste repositório

Projeto acadêmico (disciplina de IA, UFMS): sistema de reconhecimento facial em duas partes —
Parte 1 (experimentação, treinamento e fine-tuning) e Parte 2 (aplicação).
Desenvolvido com Spec-Driven Development: requisitos e decisões ficam em `docs/`.

## 1. Privacidade e LGPD — regras inegociáveis

Dados biométricos são dados pessoais sensíveis (LGPD, art. 5º, II). Violar estas regras zera o trabalho.

- **Nunca** ler, abrir, listar o conteúdo, copiar ou processar arquivos em `data/private/`, `faces/`,
  `embeddings/`, `database/`, `checkpoints/`, `consent/`, nem fotos, vídeos, embeddings (`.npy`, `.pkl`),
  bancos (`.db`, `.sqlite`) ou termos (`.pdf`, `.docx`). O `.claude/settings.json` bloqueia boa parte
  disso, mas a regra vale também para comandos de shell (`cat`, `python -c`, etc.).
- **Nunca** pedir aos integrantes fotos, embeddings, nomes de participantes ou resultados individuais.
- Testes e exemplos usam **dados sintéticos** (imagens geradas, vetores aleatórios) ou dados públicos.
- **Nunca** escrever código que envie fotos, embeddings ou resultados individuais para serviços externos
  (APIs, trackers em nuvem, notebooks remotos). Processamento de dados privados é sempre local.
- **Nunca** remover entradas do `.gitignore` ou hooks do `.pre-commit-config.yaml`, nem usar
  `git commit --no-verify` ou `git add -f` em arquivos ignorados.
- Logs não podem conter nomes, identificadores reais ou embeddings. Use IDs pseudônimos (`P01`, `P02`, …).
- Resultados do dataset privado só aparecem **agregados** (médias, taxas, curvas), nunca por pessoa.
- Modelos, classificadores (SVM, k-NN) ou índices treinados com dados privados são privados.
- Todo dado ligado a uma pessoa precisa poder ser excluído (cadastro, embeddings, fotos, derivados).

## 2. Fluxo de trabalho (Spec-Driven Development)

0. Os requisitos do sistema estão em `docs/requirements.md` e o cronograma em `docs/plano.md`.
1. Toda implementação parte de uma spec em `docs/specs/NNN-nome.md` (modelo: `000-template.md`).
   Se não houver spec para o pedido, proponha uma antes de escrever código.
2. Antes de implementar, apresente um plano curto e espere aprovação.
3. Testes são derivados dos **critérios de aceitação** da spec.
4. Decisões técnicas relevantes viram ADR em `docs/decisions/` (modelo: `ADR-000-template.md`),
   sempre citando a evidência (experimento, métrica, referência).
5. Experimentos seguem `experiments/E00-template.md` (hipótese, configuração, resultado, interpretação).
6. Mensagens de commit citam a spec/ADR/experimento quando aplicável, ex.: `feat(enrollment): trata múltiplas faces [spec-002]`.

## 3. Avaliação experimental — cuidados

- O conjunto de **teste** nunca é usado para treinar, escolher hiperparâmetros, threshold ou modelo.
- Separação por **identidade** para treinar/avaliar o backbone; por **sessão de captura** no dataset privado.
- Checar data leakage: imagens repetidas ou quase idênticas entre splits, augmentations atravessando
  splits, identidades em comum entre datasets (ex.: CelebA × LFW).
- Registrar seed, versões das bibliotecas, hiperparâmetros e hardware de cada experimento.

## 4. Código

- Código-fonte em `src/`, testes em `tests/`, configurações em `configs/` (nada de caminho fixo no código).
- O **mesmo** pré-processamento/alinhamento é usado no treino, no cadastro e no reconhecimento.
- Ao gerar código, explique as partes não óbvias: todos os integrantes precisam saber explicar
  arquitetura, perdas, treinamento, métricas e decisões.
