# E04 — Melhor modelo no dataset privado (resultados agregados)

- **Data:** _AAAA-MM-DD_
- **Responsável:** _nome_
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
| Threshold | o do modelo escolhido, vindo de `reports/thresholds.json` (não é ajustado aqui) |
| Desconhecidos | rodízio (cada pessoa fica uma vez fora do cadastro) + 500 rostos do LFW |

## Dados

6 adultos com termo assinado, 30 fotos (12 de cadastro, 18 de consulta). Nada privado sai da máquina;
embeddings só em memória.

## Resultados (agregados)

| Métrica | Protótipo | k-NN |
|---|---|---|
| Acerto rank-1 | | |
| Aceitação correta (cadastrados) | | |
| Identificação errada aceita | | |
| FRR (cadastrado rejeitado) | | |
| Rejeição correta de desconhecidos (rodízio) | | |
| Rejeição correta de desconhecidos (LFW) | | |

Gráfico: `reports/figures/E04-privado-similaridade.png`.

## Interpretação

_A preencher. Limitação: amostra pequena (18 consultas); os números indicam tendência, não precisão estatística._
