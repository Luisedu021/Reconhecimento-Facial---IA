# E04 — Melhor modelo no dataset privado (resultados agregados)

- **Data:** 2026-10-25
- **Responsável:** Luís Eduardo Lopes dos Santos
- **Config:** [`configs/e04.yaml`](../configs/e04.yaml)
- **Modelo avaliado:** E06 (Triplet loss, 15 épocas), vencedor pela validação ([ADR-001](../docs/decisions/ADR-001-face-model.md)); threshold 0,4386

## Hipótese

O melhor modelo de E01–E03, E05 e E06, com o threshold escolhido na validação do CelebA, reconhece as pessoas
cadastradas e rejeita desconhecidos também nas selfies reais do grupo.

## Configuração

| Item | Valor |
|---|---|
| Cadastro | 2 fotos por pessoa (sessão S1) |
| Consulta | 3 fotos por pessoa (sessão S2, outro dia) |
| Comparação | protótipo (média do cadastro) + cosseno × k-NN (k = 1) |
| Threshold | 0,4386, o do E06, vindo de `reports/thresholds.json` (não é ajustado aqui, CA-5) |
| Desconhecidos | rodízio (cada pessoa fica uma vez fora do cadastro: 5 rodadas) + 500 rostos do LFW |
| Análise de sensibilidade | grade fixa de thresholds 0,30 a 0,80, passo 0,05, definida na config antes de rodar (CA-9) |
| Pré-processamento | o mesmo MTCNN 160×160 do treino e da avaliação pública |
| Seed / hardware | 42 / RTX A1000 4 GB |

## Dados

5 adultos com termo assinado, 25 fotos (10 de cadastro, 15 de consulta), IDs pseudônimos P01–P05.
O MTCNN encontrou rosto nas 25. Nada privado sai da máquina; embeddings só em memória.

| Item | Valor |
|---|---|
| Consultas de cadastrados | 15 (3 por pessoa, da sessão S2) |
| Consultas de desconhecidos (rodízio) | 25 (as 5 fotos, S1 e S2, de quem está fora do cadastro, em cada rodada) |
| Consultas de desconhecidos (LFW) | 500 rostos de pessoas públicas, nunca cadastrados |
| Checagem de leakage | cadastro e consulta de sessões diferentes; threshold e modelo escolhidos só no CelebA-val |

## Métricas

Definidas antes de rodar (SPEC-001): acerto rank-1, aceitação correta e FRR dos cadastrados, identificação errada
aceita, rejeição correta de desconhecidos (rodízio e LFW), ROC-AUC e EER das similaridades consulta × protótipo;
e, como análise (adendo da SPEC-001), FRR e FAR ao longo da grade de thresholds.

## Resultados (agregados)

| Métrica | Protótipo | k-NN |
|---|---|---|
| Acerto rank-1 | 100% | 100% |
| Aceitação correta (cadastrados) | 100% | 100% |
| Identificação errada aceita | 0% | 0% |
| FRR (cadastrado rejeitado) | 0% | 0% |
| Rejeição correta de desconhecidos (rodízio) | **56%** (FAR 44%) | **56%** (FAR 44%) |
| Rejeição correta de desconhecidos (LFW) | 97,0% | 95,6% |
| ROC-AUC / EER (consulta S2 × protótipos) | 1,00 / 0% | 1,00 / 0% |

Para comparação, a primeira execução com o E05 (threshold 0,450) deu rejeição de desconhecidos de 64% (protótipo)
e 60% (k-NN) no rodízio, com os demais números iguais. A diferença é de 1 a 2 fotos em 25.

**Análise de sensibilidade** (`reports/E04-sensibilidade.csv`, só análise; nenhum threshold foi escolhido daqui):

| Threshold | 0,30 | 0,35 | 0,40 | **0,45** | 0,50 | 0,55 | 0,60 | 0,65 | 0,70 | 0,75 | 0,80 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FRR (protótipo) | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 13,3% |
| FAR rodízio (protótipo) | 72% | 56% | 48% | 44% | 36% | 36% | 36% | 32% | 28% | 4% | 0% |
| FAR LFW (protótipo) | 18,0% | 8,8% | 4,4% | 2,6% | 0,8% | 0,6% | 0,2% | 0% | 0% | 0% | 0% |
| FAR rodízio (k-NN) | 76% | 60% | 52% | 44% | 36% | 36% | 36% | 28% | 28% | 0% | 0% |
| FAR LFW (k-NN) | 23,8% | 14,2% | 6,4% | 3,6% | 1,6% | 0,6% | 0,4% | 0,2% | 0% | 0% | 0% |

Gráficos: `reports/figures/E04-privado-similaridade.png` (distribuição mesma pessoa × pessoas diferentes) e
`reports/figures/E04-sensibilidade-threshold.png` (FRR e FAR em função do threshold, threshold oficial marcado).

## Interpretação

- **A hipótese se confirmou em parte.** O reconhecimento dos cadastrados foi perfeito: as 15 consultas, feitas com
  fotos de outro dia e com variações de pose, luz e distância, foram atribuídas à pessoa certa e aceitas. Mas a
  rejeição de desconhecidos ficou muito abaixo do esperado: no rodízio, 44% das fotos de quem **não** estava
  cadastrado passaram do threshold, associadas a outro participante.
- **O threshold é o principal problema, não a separação.** Com AUC = 1,0 e EER = 0% nas consultas S2 contra os
  protótipos, as similaridades da mesma pessoa (0,75 a 0,98) e de pessoas diferentes não se sobrepõem. A
  sensibilidade mostra o tamanho da mudança de domínio: o threshold escolhido em fotos de celebridades (0,439)
  fica longe da faixa em que as selfies funcionam. Nesta amostra, só perto de 0,75 o FAR do rodízio cai para
  0–4% sem rejeitar cadastrados; em 0,80 o FRR já sobe para 13%. Esse valor **não** é adotado: foi observado no
  teste (CA-5, item 29).
- **Os desconhecidos do LFW são bem rejeitados (95,6–97%)** e passam de 1% de FAR já em 0,50–0,55. O problema é
  específico de participantes do mesmo grupo, fotografados nas mesmas condições.
- **Erros concentrados:** a curva do rodízio tem um patamar (36% entre 0,50 e 0,60; 28% em 0,70) e cai de uma vez
  entre 0,70 e 0,75. No histograma, há um pequeno grupo isolado de pares de pessoas diferentes perto de 0,70, longe
  dos demais (abaixo de 0,50). Isso indica que poucos pares de participantes concentram os erros, com rostos muito
  parecidos entre si ou fotos em condições quase iguais (o rodízio também usa as fotos S1, de fundo branco e pose
  frontal padronizados). Por privacidade, a análise não identifica quais pares são.
- **Protótipo × k-NN:** mesma taxa no threshold oficial (56%); ao longo da grade, o protótipo aceita um pouco
  menos de desconhecidos nos thresholds baixos. A média das 2 fotos de cadastro é mais estável que a foto mais
  parecida.
- **E06 × E05 no privado:** o E06 rejeitou 1 a 2 fotos a menos que o E05, com threshold um pouco menor (0,439
  contra 0,450). Com 25 consultas, a diferença não é conclusiva.
- **Correção proposta para a Parte 2:** escolher o threshold de novo **na validação**, com critério mais rígido
  (por exemplo FAR ≤ 0,1%), e registrar no ADR-003. A sensibilidade sugere que só isso pode não bastar; uma
  validação mais parecida com selfies (por exemplo, um dataset público de fotos de celular) ajudaria.
- **Limitações:** amostra muito pequena (5 pessoas, 15 consultas de cadastrados, 25 de desconhecidos). Uma foto a
  mais ou a menos muda as taxas em 4 a 7 pontos percentuais; os números indicam tendência, não precisão estatística.
