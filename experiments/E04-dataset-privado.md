# E04 — Melhor modelo no dataset privado (resultados agregados)

- **Data:** 2026-10-25
- **Responsável:** Luís Eduardo Lopes dos Santos
- **Config:** [`configs/e04.yaml`](../configs/e04.yaml)
- **Modelo avaliado:** E05 (fine-tuning parcial com Triplet loss), vencedor pela validação ([ADR-001](../docs/decisions/ADR-001-face-model.md)); threshold 0,450

## Hipótese

O melhor modelo de E01–E03 e E05, com o threshold escolhido na validação do CelebA, reconhece as pessoas
cadastradas e rejeita desconhecidos também nas selfies reais do grupo.

## Configuração

| Item | Valor |
|---|---|
| Cadastro | 2 fotos por pessoa (sessão S1) |
| Consulta | 3 fotos por pessoa (sessão S2, outro dia) |
| Comparação | protótipo (média do cadastro) + cosseno × k-NN (k = 1) |
| Threshold | 0,4496, o do E05, vindo de `reports/thresholds.json` (não é ajustado aqui, CA-5) |
| Desconhecidos | rodízio (cada pessoa fica uma vez fora do cadastro: 5 rodadas) + 500 rostos do LFW |
| Pré-processamento | o mesmo MTCNN 160×160 do treino e da avaliação pública |
| Seed / hardware | 42 / RTX A1000 4 GB |

## Dados

5 adultos com termo assinado, 25 fotos (10 de cadastro, 15 de consulta), IDs pseudônimos P01–P05.
O MTCNN encontrou rosto nas 25. Nada privado sai da máquina; embeddings só em memória.

| Item | Valor |
|---|---|
| Consultas de cadastrados | 15 (3 por pessoa, da sessão S2) |
| Consultas de desconhecidos (rodízio) | 25 (as 5 fotos de quem está fora do cadastro, em cada rodada) |
| Consultas de desconhecidos (LFW) | 500 rostos de pessoas públicas, nunca cadastrados |
| Checagem de leakage | cadastro e consulta de sessões diferentes; threshold e modelo escolhidos só no CelebA-val |

## Métricas

Definidas antes de rodar (SPEC-001): acerto rank-1, aceitação correta e FRR dos cadastrados, identificação errada
aceita, rejeição correta de desconhecidos (rodízio e LFW), ROC-AUC e EER das similaridades consulta × protótipo.

## Resultados (agregados)

| Métrica | Protótipo | k-NN |
|---|---|---|
| Acerto rank-1 | 100% | 100% |
| Aceitação correta (cadastrados) | 100% | 100% |
| Identificação errada aceita | 0% | 0% |
| FRR (cadastrado rejeitado) | 0% | 0% |
| Rejeição correta de desconhecidos (rodízio) | **64%** (FAR 36%) | **60%** (FAR 40%) |
| Rejeição correta de desconhecidos (LFW) | 97,0% | 95,6% |
| ROC-AUC / EER (consulta S2 × protótipos) | 1,00 / 0% | 1,00 / 0% |

Gráfico: `reports/figures/E04-privado-similaridade.png` (distribuição de similaridade mesma pessoa × pessoas
diferentes, só agregada).

## Interpretação

- **A hipótese se confirmou em parte.** O reconhecimento dos cadastrados foi perfeito: as 15 consultas,
  feitas com fotos de outro dia e com variações de pose, luz e distância, foram atribuídas à pessoa certa e
  aceitas. Mas a rejeição de desconhecidos ficou muito abaixo do esperado: no rodízio, 36% (protótipo) a 40%
  (k-NN) das fotos de quem **não** estava cadastrado passaram do threshold, associadas a outro participante.
- **O problema é o threshold, não o modelo.** Com AUC = 1,0 e EER = 0%, as similaridades "mesma pessoa" e
  "pessoas diferentes" (consultas S2 contra os protótipos) não se sobrepõem: existe um threshold que separa
  esses pares. O rodízio também compara as fotos de cadastro (S1) de quem saiu, então a separação ali pode ser
  um pouco menor, mas o padrão é o mesmo. O valor 0,450 foi escolhido para
  FAR = 1% em fotos de celebridades (CelebA-val) e fica baixo demais para selfies. É a **mudança de domínio**
  prevista como risco no [ADR-003](../docs/decisions/ADR-003-recognition-threshold.md). O sinal já tinha
  aparecido no LFW (FAR 1,40%) e ficou bem maior aqui.
- **Por que os desconhecidos do LFW são mais rejeitados (95,6–97%) que os do rodízio (60–64%):** os rostos
  do LFW vêm de outro domínio (fotos de imprensa). Já um participante fora do cadastro tira selfies nas mesmas
  condições dos cadastrados (câmera de celular, distância, iluminação parecidas), o que aumenta a similaridade
  entre pessoas diferentes.
- **Protótipo × k-NN:** o protótipo rejeitou um pouco mais de desconhecidos (64% contra 60%) com o mesmo acerto
  nos cadastrados. A média das 2 fotos de cadastro é mais estável que a foto mais parecida, que pode coincidir
  por acaso com o desconhecido. Com 25 consultas, a diferença é de 1 foto e não é conclusiva.
- **O threshold não foi reajustado com estas fotos** (CA-5, item 29): elas são teste. A correção proposta para a
  Parte 2 é escolher o threshold de novo **na validação**, com um critério mais rígido (por exemplo FAR ≤ 0,1%),
  e registrar no ADR-003.
- **Limitações:** amostra muito pequena (5 pessoas, 15 consultas de cadastrados, 25 de desconhecidos). Uma foto a
  mais ou a menos muda as taxas em 4 a 7 pontos percentuais; os números indicam tendência, não precisão estatística.
