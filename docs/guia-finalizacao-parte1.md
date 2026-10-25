# Guia de finalização da Parte 1

Passo a passo para quem vai terminar a Parte 1 a partir de um `git clone`. Detalhes do protocolo em
[SPEC-001](specs/001-protocolo-experimental.md); regras de privacidade em [privacy.md](privacy.md).

## Antes de começar: privacidade

- **Termos assinados:** só usar fotos de quem assinou o Termo de Consentimento (item 14). Concordar de boca
  não basta. Os termos ficam em local privado, fora do repositório e da nuvem.
- **Nada de nuvem:** as fotos não ficam no Google Drive, no WhatsApp nem em outro serviço de compartilhamento
  (item 22). Depois de baixá-las para o computador, apague a pasta do Drive, inclusive da lixeira. Para passar as
  fotos entre integrantes, use pendrive ou outra transferência direta.
- **Nada para a IA:** fotos, nomes e resultados por pessoa não vão para o Claude nem para outro agente (item 23).

## 1. Preparar o ambiente (uma vez)

Clone **fora** do OneDrive ou de qualquer pasta sincronizada:

```bash
git clone https://github.com/Luisedu021/Reconhecimento-Facial---IA.git C:\dev\reconhecimento-facial
```

```bash
cd C:\dev\reconhecimento-facial; py -3.10 -m venv .venv; .venv\Scripts\pip install -r requirements.txt; .venv\Scripts\pip install -e .; .venv\Scripts\pre-commit install
```

- O `pre-commit install` é obrigatório: é ele que bloqueia fotos e dados privados no commit.
- Para conferir se está tudo certo: `.venv\Scripts\pytest`. Todos os testes devem passar.

## 2. Baixar os datasets públicos

Baixe pelo navegador, das fontes oficiais:

| Dataset | Arquivos | Onde colocar |
|---|---|---|
| [LFW](http://vis-www.cs.umass.edu/lfw/) (fora do ar; usar o espelho: [`lfw.tgz`](https://ndownloader.figshare.com/files/5976018) e [`pairs.txt`](https://ndownloader.figshare.com/files/5976006)) | `lfw.tgz` e `pairs.txt` | extrair em `data/public/lfw/` (fica `data/public/lfw/lfw/<Nome>/...`) e `data/public/lfw/pairs.txt` |
| [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) | `img_align_celeba.zip` e `identity_CelebA.txt` | extrair em `data/public/celeba/img_align_celeba/` e `data/public/celeba/identity_CelebA.txt` |

Leia os termos de uso dos dois e preencha a tabela de datasets do [README](../README.md) (item 10).

## 3. Rodar os experimentos públicos

Nesta ordem, cerca de 1 a 2 horas no total:

```bash
.venv\Scripts\python -m facerec.preprocess --dataset lfw; .venv\Scripts\python -m facerec.preprocess --dataset celeba; .venv\Scripts\python -m facerec.overlap
```

```bash
.venv\Scripts\python -m facerec.evaluate --config configs/e01.yaml; .venv\Scripts\python -m facerec.train --config configs/e02.yaml; .venv\Scripts\python -m facerec.evaluate --config configs/e02.yaml; .venv\Scripts\python -m facerec.train --config configs/e03.yaml; .venv\Scripts\python -m facerec.evaluate --config configs/e03.yaml
```

E05 (opcional na SPEC-001) e E06 (adendo; foi o vencedor):

```bash
.venv\Scripts\python -m facerec.train --config configs/e05.yaml; .venv\Scripts\python -m facerec.evaluate --config configs/e05.yaml; .venv\Scripts\python -m facerec.train --config configs/e06.yaml; .venv\Scripts\python -m facerec.evaluate --config configs/e06.yaml
```

Os resultados vão para `reports/results.csv`, `reports/thresholds.json` e `reports/figures/`.

## 4. Escolher o melhor modelo

- O vencedor é o de **menor EER na linha `celeba-val`** do `results.csv`. Não escolha pelo LFW: ele é teste (item 29).
- Registre o vencedor no [ADR-001](decisions/ADR-001-face-model.md) e troque o `source_config` em
  `configs/e04.yaml` para o YAML dele.
- Confirme a licença do `facenet-pytorch` e dos pesos VGGFace2, e anote no ADR-001.
- Commit e push de `reports/` e `docs/`. A pasta `checkpoints/` fica fora do Git automaticamente.

## 5. Dataset privado (5 pessoas)

1. Baixe as fotos para dentro de `C:\dev\reconhecimento-facial\dataset\`, com **uma subpasta por pessoa**
   (`pessoa1` a `pessoa6`, 5 fotos em cada). Depois apague a cópia do Drive.
2. Organize, gravando o mapa pessoa → ID fora do repositório:

   ```bash
   .venv\Scripts\python -m facerec.organize_private --origem dataset --mapa C:\dados-privados\mapa-participantes.csv
   ```

3. Abra `data/private/P01` a `P05` e confira: `S1` com 2 fotos (selfie padrão + sorrindo) e `S2` com 3 (as variações).
4. Rode a avaliação (depois do passo 4):

   ```bash
   .venv\Scripts\python -m facerec.preprocess --dataset private; .venv\Scripts\python -m facerec.evaluate_private --config configs/e04.yaml
   ```

5. A saída é só agregada (médias e taxas, nada por pessoa) e pode ir para o README.

## 6. Documentação

- `experiments/E01` a `E04`: preencher os resultados e escrever a interpretação: o que melhorou, o que piorou
  e por quê, custo computacional e limitações (item 33).
- [README](../README.md): integrantes, tabela de resultados e link do vídeo.
- [requirements.md](requirements.md): preencher o que falta.
- [plano.md](plano.md): confirmado com 5 participantes (25 fotos).

## 7. Vídeo

Mostrar: problema, datasets, arquitetura, hipóteses, experimentos, treinamento, resultados, comparação,
conclusão e escolha do modelo (item 35). Nenhuma foto ou rosto de participante, a não ser de quem autorizou no termo.

## 8. Antes de entregar

- [ ] Checklist do Anexo B em [privacy.md](privacy.md), incluindo a verificação do histórico do Git.
- [ ] Os dois integrantes sabem explicar o código: ArcFace, fine-tuning, métricas e threshold (item 5).
- [ ] Entregar o link do GitHub e o link do vídeo. Cada dia de atraso custa 1 ponto.
