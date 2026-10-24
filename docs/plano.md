# Plano da Parte 1 — 10 dias

- **Início:** __/__ · **Entrega:** __/__
- **Trilha A — dados e avaliação:** _nome_
- **Trilha B — treinamento:** _nome_

As duas trilhas revisam o trabalho uma da outra (PR revisado pelo colega). Os dois integrantes
precisam saber explicar tudo, não só a própria trilha.

## Escopo reduzido (decisões para caber no prazo)

**Dataset privado:** 5 adultos com termo assinado, 5 selfies cada (25 fotos no total), enviadas pelo
participante diretamente ao aluno responsável ([convite-participantes.md](convite-participantes.md)).
O teste de "desconhecido" usa rodízio: cada pessoa fica uma vez fora do cadastro.

| Sessão | Fotos por pessoa | Uso |
|---|---|---|
| S1 (dia 2) | 2: selfie padrão com fundo branco + 1 sorrindo | cadastro |
| S2 (dia 5, outro dia, qualquer fundo) | 3 variações: cabeça virada, luz diferente, mais longe ou de óculos | teste |

Modelo, hiperparâmetros e threshold são escolhidos na **validação do CelebA** (identidades fora do treino).
LFW e dataset privado são só **teste** (item 29). Detalhes em [SPEC-001](specs/001-protocolo-experimental.md).

**Experimentos** (proposta — modelo final decidido no ADR-001):

| ID | Configuração | Teste |
|---|---|---|
| E01 | Pré-treinado, sem fine-tuning, cosseno (baseline) | LFW |
| E02 | Fine-tuning parcial + ArcFace loss, subconjunto do CelebA | LFW |
| E03 | Fine-tuning completo, mesma configuração do E02 | LFW |
| E04 | Melhor modelo: protótipo/cosseno × k-NN, threshold da validação, "desconhecido" | dataset privado (agregado) |
| E05 | _Opcional:_ E02 com Triplet loss no lugar da ArcFace | LFW |

## Cronograma

### Dia 1 — __/__
- [ ] **A:** convidar os participantes ([convite-participantes.md](convite-participantes.md))
- [ ] **A:** imprimir e coletar os termos de consentimento (Anexo A) de todos os participantes
- [ ] **A:** ler as licenças do LFW e do CelebA e preencher a tabela de datasets do README
- [ ] **A:** iniciar o download do LFW e do CelebA (fontes oficiais)
- [ ] **B:** instalar Python + PyTorch com CUDA; confirmar que a GPU é reconhecida
- [ ] **B:** testar no Windows a instalação das bibliotecas de detecção e reconhecimento facial
- [ ] **B:** `pre-commit install`, primeiro commit, repositório privado no GitHub
- [ ] **Ambos:** rascunho do `docs/requirements.md`

### Dia 2 — __/__
- [ ] **A:** fotos da sessão S1
- [ ] **A:** pipeline detecção → alinhamento → embedding funcionando em imagens do LFW
- [ ] **B:** spec do protocolo experimental (`docs/specs/001-protocolo-experimental.md`)
- [ ] **B:** loader do CelebA com divisão treino/validação **por identidade**

### Dia 3 — __/__
- [ ] **A:** avaliação no protocolo de pares do LFW: ROC, EER, threshold por fold → **E01**
- [ ] **B:** loop de fine-tuning (congelamento de camadas, AMP, checkpoints locais)
- [ ] **B:** checar identidades em comum entre CelebA e LFW (leakage)

### Dia 4 — __/__
- [ ] **A:** gráficos de distribuição de similaridade (mesma pessoa × pessoas diferentes)
- [ ] **B:** rodar **E02** e **E03** (aproveitar a noite)

### Dia 5 — __/__
- [ ] **A:** fotos da sessão S2
- [ ] **A:** script de avaliação no dataset privado (saída só agregada)
- [ ] **B:** rodar **E05** (opcional, Triplet loss)

### Dia 6 — __/__
- [ ] **A:** rodar **E04** (dataset privado)
- [ ] **B:** consolidar métricas, configs e seeds de todos os experimentos

### Dia 7 — __/__
- [ ] **Ambos:** análise: o que melhorou, o que piorou e por quê; custo × desempenho; erros e limitações
- [ ] **Ambos:** folga para o que atrasou

### Dia 8 — __/__
- [ ] **A:** README com a tabela de resultados e instruções de reprodução
- [ ] **B:** relatórios em `experiments/` e ADRs (modelo e threshold)

### Dia 9 — __/__
- [ ] **Ambos:** roteiro e gravação do vídeo (problema, datasets, arquitetura, hipóteses, experimentos, resultados, conclusão)

### Dia 10 — __/__
- [ ] **Ambos:** checklist do Anexo B e verificação do histórico do Git ([privacy.md](privacy.md))
- [ ] **Ambos:** folga + entrega (link do GitHub + link do vídeo)

## Riscos

| Risco | Mitigação |
|---|---|
| Download do CelebA travar (cota do Google Drive) | começar no dia 1; ter o LFW como plano mínimo |
| Biblioteca não instalar no Windows | testar no dia 1; trocar de biblioteca cedo |
| Treino lento na GPU de 4 GB | subconjunto do CelebA (1 a 2 mil identidades), poucas épocas, AMP; Kaggle/Colab **só com dados públicos** |
| Integrante não entender código gerado por IA | revisão cruzada e explicação do código antes de cada merge |
