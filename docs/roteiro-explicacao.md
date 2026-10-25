# Roteiro de explicação da Parte 1

Material de estudo para os dois integrantes e base do vídeo (item 35). A ordem segue os tópicos exigidos:
problema → datasets → arquitetura → hipóteses → experimentos → treinamento → resultados → comparação →
conclusão → escolha do modelo. Cada seção traz **o que dizer**, **onde está no código** e **perguntas
prováveis**. Os números vêm de `reports/results.csv` (execução de 2026-10-25); os do E04 entram depois.

---

## 1. O problema

**O que dizer**

- Reconhecimento facial tem duas tarefas:
  - **Verificação (1:1):** "estas duas fotos são da mesma pessoa?". Exemplo: desbloquear o celular.
  - **Identificação (1:N):** "quem é esta pessoa, entre as cadastradas, ou é um desconhecido?". Exemplo: controle de acesso.
- A Parte 2 vai fazer identificação, mas as duas se resolvem do mesmo jeito: comparar rostos por similaridade.
- A pergunta da Parte 1 (enunciado, item 4): **qual configuração de modelo, treinamento e/ou fine-tuning dá o
  melhor desempenho?** A resposta tem que vir de experimentos, não de opinião.

**A ideia central: embeddings, não classificação**

- Uma rede comum de classificação teria uma saída por pessoa. Para cadastrar alguém novo, seria preciso retreinar.
- Por isso usamos **metric learning**: a rede transforma cada rosto em um vetor de **512 números** (o *embedding*).
  - Fotos da mesma pessoa → vetores que apontam para a mesma direção.
  - Pessoas diferentes → vetores em direções diferentes.
- Para comparar, usamos a **similaridade de cosseno**: o cosseno do ângulo entre os dois vetores.
  Vai de -1 a 1; quanto maior, mais parecido.
- Decisão: se `similaridade ≥ threshold`, é a mesma pessoa.
- Cadastrar alguém novo = guardar o embedding dessa pessoa. **Nada de retreinar.**

**Perguntas prováveis**

- *Por que cosseno e não distância euclidiana?* Os embeddings são normalizados (comprimento 1). Nesse caso as duas
  medidas são equivalentes (‖a−b‖² = 2 − 2·cos), e o cosseno é mais fácil de interpretar como "ângulo".
- *Por que 512 dimensões?* É o tamanho da saída da rede pré-treinada que usamos; não escolhemos.

---

## 2. Os datasets

**O que dizer:** cada conjunto tem um papel, e nenhum deles é usado para duas coisas.

| Dataset | Papel | Detalhes |
|---|---|---|
| **CelebA** | Treino e **validação** | 202 mil fotos de ~10 mil famosos. Usamos um subconjunto: **1.000 identidades × 20 fotos**. ~900 identidades treinam; ~100, **pessoas diferentes**, só validam. |
| **LFW** | **Teste** final | 13.233 fotos de 5.749 pessoas. O `pairs.txt` define 6.000 pares (3.000 iguais, 3.000 diferentes) em 10 folds. É o benchmark padrão da área. |
| **Privado** | Teste em condições reais | Selfies dos 5 participantes com termo assinado, em duas sessões: **S1** (cadastro, 2 fotos) e **S2** (consulta, 3 fotos com variações, outro dia). |

**Por que separar assim (o ponto mais importante do trabalho)**

- Se usássemos o mesmo conjunto para **escolher** (modelo, época, threshold) e para **medir**, o número sairia
  otimista. É como estudar com a prova na mão. Isso se chama **data leakage** (vazamento de dados).
- Regras que seguimos (item 29):
  1. A validação (CelebA-val) escolhe **tudo**: a melhor época, o melhor experimento e o threshold.
  2. O LFW só **mede**, no fim. Nunca escolhe nada.
  3. A separação do CelebA é **por identidade**: ninguém da validação aparece no treino. Assim medimos se o
     modelo generaliza para **pessoas novas**, que é o que acontece na prática.
  4. **Overlap CelebA × LFW** (`overlap.py`): os dois têm famosos, então pode haver a mesma foto (ou quase) nos
     dois. Calculamos o cosseno entre todas as fotos do CelebA e do LFW; acima de 0,9 é quase-duplicata e sai do
     treino. Resultado: **6 de 19.974** imagens removidas.
  5. No privado, a separação é **por sessão**: cadastro com S1 e consulta com S2, de outro dia. Fotos da mesma
     sessão seriam parecidas demais e inflariam o resultado.

**Privacidade (LGPD):** foto de rosto é **dado biométrico, dado pessoal sensível** (art. 5º, II). Por isso:
termo de consentimento assinado, IDs pseudônimos (P01–P05), nenhuma foto no Git nem na nuvem, embeddings só em
memória e resultados só agregados (médias, taxas), nunca por pessoa.

**Perguntas prováveis**

- *Por que não treinar no LFW?* Porque ele é o teste. E ele é pequeno: muitas pessoas têm só 1 foto.
- *Por que só 1.000 identidades do CelebA?* Custo: a GPU tem 4 GB e o tempo é limitado. 20 fotos por pessoa dão
  variedade suficiente para o fine-tuning.
- *Por que 20 fotos por pessoa e não todas?* Para equilibrar as classes: ninguém domina o treino por ter mais fotos.

---

## 3. A arquitetura (o pipeline)

```
foto → [MTCNN: detecta e recorta] → rosto 160×160 → [InceptionResnetV1] → embedding 512-d → cosseno → threshold → decisão
```

**3.1 Pré-processamento: MTCNN** (`preprocess.py`)

- **MTCNN** (*Multi-task Cascaded CNN*) são 3 redes pequenas em cascata:
  - **P-Net** propõe muitas regiões candidatas;
  - **R-Net** descarta as falsas;
  - **O-Net** refina a caixa e acha 5 pontos (olhos, nariz, cantos da boca).
- Recortamos o rosto com **14 px de margem** e redimensionamos para **160×160**, o tamanho de entrada da rede.
- Se houver várias faces: no CelebA fica a **maior**; no LFW, a maior **e mais central** (`center_weighted_size`),
  porque o LFW foi montado com a pessoa-alvo no centro.
- Regra do projeto: **o mesmo pré-processamento** em treino, teste, cadastro e reconhecimento. Se o treino visse
  rostos recortados de um jeito e o cadastro de outro, o modelo erraria por causa do recorte, não do rosto.
- O recorte é salvo como PNG novo, **sem EXIF**: a localização GPS das fotos originais não é copiada.

**3.2 O modelo: InceptionResnetV1** (`model.py`)

- É uma CNN profunda que combina dois ingredientes:
  - **Inception:** cada bloco aplica filtros de tamanhos diferentes em paralelo (1×1, 3×3, …) e junta os
    resultados. Assim captura detalhes pequenos e formas maiores ao mesmo tempo.
  - **Conexões residuais (ResNet):** a entrada do bloco é somada à saída (`x + F(x)`). Isso deixa o gradiente
    passar direto por atalhos e torna possível treinar redes profundas.
- Vem **pré-treinada no VGGFace2** (~3,3 milhões de fotos, ~9 mil pessoas), pela biblioteca `facenet-pytorch`.
  Ou seja, ela já "sabe" o que diferencia rostos. Isso é **transfer learning**.
- Partes que importam para o fine-tuning, da entrada para a saída:
  `conv iniciais → blocos Inception-ResNet (repeat_1, 2, 3) → block8 → avgpool → dropout → last_linear (→512) → last_bn`.
- A camada de classificação original (8.631 pessoas do VGGFace2) **não é usada**. A saída é o embedding.

**Perguntas prováveis**

- *Por que um modelo pré-treinado?* Treinar do zero exigiria milhões de fotos e dias de GPU. Partir do VGGFace2 e
  ajustar é o padrão quando há pouco dado e pouco hardware.
- *O que é o BatchNorm (`last_bn`)?* Normaliza as ativações com média e variância aprendidas. Deixa o treino mais
  estável.

---

## 4. As hipóteses e os experimentos

| Exp. | Configuração | Hipótese |
|---|---|---|
| **E01** | Modelo pré-treinado, **sem treinar** | Já separa bem as pessoas e serve de **linha de base**. |
| **E02** | **Fine-tuning parcial:** treina só `block8` + `last_linear` + `last_bn`, com ArcFace | Melhora os embeddings sem "esquecer" o que já sabe. |
| **E03** | **Fine-tuning completo:** a rede inteira, com LR 10× menor no backbone | Adapta mais, mas custa mais e arrisca overfitting. |
| **E05** | Igual ao E02, mas com **Triplet loss** no lugar da ArcFace (opcional na spec) | A triplet gera embeddings **piores** que a ArcFace. |
| **E04** | O vencedor pela validação, no **dataset privado** | Funciona nas selfies reais e rejeita desconhecidos. |

Tudo o que muda entre os experimentos está nos YAML (`configs/e01.yaml` … `e05.yaml`). O que é comum fica em
`configs/base.yaml`. Seed fixa (42) para tudo ser reproduzível.

**Por que essa progressão?** É a pergunta clássica de transfer learning: *quanto* da rede ajustar?

- Nada (E01): zero risco, zero adaptação.
- Só o topo (E02): as camadas finais são as mais específicas da tarefa; as iniciais detectam bordas e texturas,
  que servem para qualquer imagem.
- Tudo (E03): máxima adaptação, mas com 900 pessoas a rede pode **decorar** o treino (overfitting) ou sofrer
  **esquecimento catastrófico** (perder o que aprendeu no VGGFace2).

---

## 5. O treinamento (`train.py`, `losses.py`)

**5.1 A função de perda: ArcFace**

- O problema: para aprender embeddings, precisamos de uma perda que **junte** as fotos da mesma pessoa e
  **afaste** as de pessoas diferentes.
- Como funciona:
  1. Cada pessoa do treino tem um vetor "representante" **W_j**, aprendido junto com a rede.
  2. Para um embedding **x** da pessoa **y**, calculamos o ângulo **θ** entre x e cada W_j (via cosseno, com os
     dois vetores normalizados).
  3. **O truque:** antes da softmax, somamos uma **margem m = 0,5 rad (~29°)** ao ângulo da classe correta.
     O logit vira **s · cos(θ_y + m)**.
  4. Como a rede é "penalizada" na classe certa, para acertar ela precisa deixar x **muito** mais perto de W_y do
     que dos outros. Resultado: grupos compactos e bem separados.
  5. **s = 64** é uma escala: os cossenos ficam entre -1 e 1, e sem escala a softmax ficaria "mole" demais.
- No código (`ArcFaceHead.forward`): `cos(θ + m) = cos θ · cos m − sen θ · sen m`. Quando θ + m passaria de π, o
  cosseno deixaria de diminuir; aí usamos uma penalidade linear (`cos − mm`).
- Depois do treino, a cabeça ArcFace (os W_j) **é descartada**: ela só serve para treinar. Usamos só o embedding.
- Referência: Deng et al., *ArcFace: Additive Angular Margin Loss for Deep Face Recognition*, CVPR 2019.

**Comparação com a Triplet Loss** (também implementada, item 28)

- A triplet usa trios (âncora, positivo, negativo) e exige `d(a,p) + margem < d(a,n)`.
- Problema: a maioria dos trios é fácil demais e não ensina nada, então é preciso **minerar** trios difíceis.
  Implementamos a mineração *semi-hard* (FaceNet, Schroff et al., 2015).
- A ArcFace compara cada exemplo com **todas** as classes de uma vez, sem mineração. Por isso costuma ser mais
  estável em treinos grandes e foi a escolhida nos experimentos principais (E02, E03).
- **O que aconteceu no nosso caso (E05):** a triplet ganhou. Ela só gera gradiente quando um negativo invade a
  margem; como o modelo pré-treinado já separa quase todos os trios, a perda começa baixa (0,015) e o ajuste é
  **suave**, corrigindo só os casos difíceis. A ArcFace, com s = 64 e m = 0,5, puxa cada rosto para um de 900
  centros novos e aleatórios: um sinal forte que reorganiza o espaço aprendido no VGGFace2.
- Referência: Schroff et al., *FaceNet: A Unified Embedding for Face Recognition and Clustering*, CVPR 2015.

**5.2 Detalhes do treino** (pergunta comum!)

- **Aquecimento (época 1):** a cabeça ArcFace começa com pesos **aleatórios**. Se o backbone treinasse junto,
  esses gradientes "lixo" estragariam os pesos pré-treinados. Então, na 1ª época, **só a cabeça treina**; da 2ª
  em diante, a rede também.
- **Learning rates diferentes:**
  - cabeça nova: `1e-3` (precisa aprender do zero);
  - camadas finais: `1e-4`;
  - resto do backbone, no E03: `1e-5`, 10× menor, para mexer pouco no que já funciona.
- **BatchNorm congelado no parcial** (`set_train_mode`): camadas congeladas ficam em `eval()`. Senão, o BatchNorm
  atualizaria suas médias com os dados novos e mudaria a rede mesmo sem gradiente. É um bug sutil e comum.
- **Otimizador AdamW** com `weight_decay = 5e-4`: o Adam com regularização L2 feita do jeito certo.
  Ajuda contra overfitting.
- **Data augmentation** (só no treino): espelhamento horizontal e variação de brilho, contraste e saturação.
  Simula condições diferentes sem precisar de mais fotos.
- **Mixed precision (AMP):** a rede roda em float16 na GPU (mais rápido, menos memória, cabe nos 4 GB); a perda
  é calculada em float32, para não perder precisão perto de cos = ±1.
- **Seleção da melhor época:** a cada época medimos o **EER na validação** e salvamos o checkpoint só se melhorou.
  Isso é *early stopping* por checkpoint: se a época 5 for pior que a 3, ficamos com a 3.
- 5 épocas, batch 64.

**Perguntas prováveis**

- *O que é uma época?* Uma passada por todas as imagens de treino.
- *O que é overfitting?* O modelo decora o treino e piora em dados novos. Detectamos quando a perda de treino
  cai, mas o EER de validação sobe.
- *Por que não usar a acurácia de classificação no treino para escolher?* Porque o que importa é separar
  **pessoas novas**. Por isso a validação usa identidades que o treino nunca viu.

---

## 6. As métricas (`metrics.py`)

Todas partem de pares com uma **similaridade** e um rótulo (mesma pessoa ou não), mais um threshold.

| Métrica | Pergunta que responde | Bom = |
|---|---|---|
| **FAR** (*False Accept Rate*) | De todos os pares de pessoas **diferentes**, quantos o sistema aceitou como iguais? | baixo |
| **FRR** (*False Reject Rate*) | De todos os pares da **mesma** pessoa, quantos o sistema rejeitou? | baixo |
| **TAR@FAR=1%** | Com o FAR travado em 1%, quantos pares corretos são aceitos? | alto |
| **EER** (*Equal Error Rate*) | Em que ponto FAR = FRR? Um número só para comparar modelos. | baixo |
| **ROC / AUC** | A curva mostra o FAR × TAR para todos os thresholds; a AUC é a área (1 = perfeito). | alto |
| **Acurácia 10 folds (LFW)** | Protocolo oficial: em cada fold, o threshold é escolhido nos outros 9. Comparável com artigos. | alto |

- **Trade-off:** subir o threshold derruba o FAR e aumenta o FRR (fica mais exigente), e vice-versa. Não dá para
  zerar os dois; é preciso escolher.
- **Por que o EER para escolher o modelo?** Ele não depende de um threshold específico e resume a curva inteira
  num número.

**Escolha do threshold (ADR-003)**

- Critério: **FAR ≤ 1% na validação do CelebA**. Escolhemos o menor threshold que respeita isso, que é o que
  aceita mais pessoas certas.
- Por quê? Num sistema de acesso, **aceitar um estranho é pior** do que pedir para a pessoa tentar de novo.
- O threshold é escolhido **uma vez**, na validação, e levado sem mudança para o LFW e para o privado. Se o FAR no
  LFW ficar perto de 1%, o threshold **generaliza**.
- Fica salvo em `reports/thresholds.json`.

**Métricas do privado (E04, identificação aberta)**

- **Rank-1:** a pessoa mais parecida na galeria é a pessoa certa?
- **Taxa de rejeição de desconhecidos:** em rodízio, cada participante sai uma vez do cadastro e vira
  "desconhecido". O sistema deve rejeitá-lo. Também testamos 500 rostos do LFW como desconhecidos.
- **Dois métodos de comparação com a galeria:**
  - `prototype`: compara com a **média** dos embeddings de cadastro da pessoa;
  - `knn1`: compara com cada foto de cadastro e fica com a **mais parecida**.

---

## 7. Resultados, comparação e conclusão

> Gráficos em `reports/figures/` (ROC e distribuição de similaridade de cada experimento). Não basta mostrar a
> tabela (item 33): é preciso **interpretar**.

| Exp. | EER CelebA-val | TAR@FAR1% val | Acurácia LFW (10 folds) | FAR / FRR no LFW (threshold da val.) | Parâmetros treinados | Tempo de treino |
|---|---|---|---|---|---|---|
| E01 | 2,82% | 95,17% | 99,27% ± 0,32% | 0,67% / 0,67% (thr 0,400) | 0 | — |
| E02 | 2,82%* | 95,17%* | 99,27%* | 0,67% / 0,67%* | 2,98 M | 3,5 min |
| E03 | 2,82%* | 95,17%* | 99,27%* | 0,67% / 0,67%* | 23,9 M | 4,6 min |
| **E05** | **2,57%** | 95,27% | 99,17% ± 0,38% | 1,40% / 0,50% (thr 0,450) | 2,52 M | 0,9 min |

\* Melhor época = aquecimento (backbone intacto), por isso igual ao E01.

EER na validação por época (a curva mais importante do vídeo):

| Época | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| E02 (ArcFace, parcial) | 2,82% (aquec.) | 2,93% | 3,00% | 3,18% | 3,07% |
| E03 (ArcFace, completo) | 2,82% (aquec.) | 3,12% | 3,12% | 3,12% | 3,37% |
| E05 (Triplet, parcial) | 2,93% | 2,83% | 2,65% | 2,72% | **2,57%** |

**Interpretação** (o que falar):

1. **Baseline forte:** sem treinar nada, 99,27% no LFW. Pesos treinados em 3,3 milhões de fotos já resolvem
   quase tudo.
2. **ArcFace piorou (E02, E03):** a perda de treino caiu (27 → 4–5), mas o EER de validação subiu. É overfitting
   e esquecimento: a rede aprendeu as 900 pessoas do treino e perdeu generalização. O E03, com 8× mais
   parâmetros livres, piorou mais (3,37%), como a hipótese dele previa.
3. **A seleção por validação nos protegeu:** o checkpoint só é salvo quando o EER melhora, então o modelo ruim
   nunca virou o modelo final.
4. **Triplet melhorou (E05):** a única configuração abaixo do baseline na validação (2,57%), e a mais barata
   (55 s). A hipótese do E05 ("triplet pior que ArcFace") foi **refutada**, e isso é um resultado legítimo.
5. **Mas o ganho é pequeno:** 0,25 p.p., da ordem do erro amostral (≈ 0,3 p.p.), e não apareceu no LFW
   (99,17% contra 99,27%, dentro do desvio entre folds; FAR de 1,40%, acima do alvo de 1%).
6. **Limitações:** uma seed, 5 épocas, 1.000 identidades, um backbone; LFW perto da saturação; dataset privado
   pequeno (5 pessoas, 15 consultas: as taxas têm muita incerteza).

**E04: dataset privado** (o que falar)

| Métrica (agregada) | Protótipo | k-NN |
|---|---|---|
| Rank-1 / FRR | 100% / 0% | 100% / 0% |
| Rejeição de desconhecidos (rodízio) | 64% | 60% |
| Rejeição de desconhecidos (LFW) | 97,0% | 95,6% |
| AUC / EER | 1,00 / 0% | 1,00 / 0% |

- **Reconhecer funcionou:** as 15 consultas (fotos de outro dia, com variações) foram para a pessoa certa.
- **Rejeitar desconhecidos não funcionou bem:** 36–40% dos participantes fora do cadastro passaram do threshold.
- **A causa é o threshold, não o modelo:** AUC 1,0 quer dizer que existe um valor que separa tudo; o 0,450,
  escolhido em fotos de celebridades, fica baixo demais para selfies (mudança de domínio, ADR-003).
- **Por que não ajustamos com essas fotos?** São teste (item 29, CA-5). A correção vai para a Parte 2: escolher de
  novo na validação, com FAR ≤ 0,1%.
- **Ressalva obrigatória:** 5 pessoas e 15 consultas; uma foto muda a taxa em 4 a 7 pontos.

**Escolha do modelo**

- Vencedor = **menor EER em CelebA-val** → **E05** (Triplet, fine-tuning parcial, threshold 0,450).
  Registrado no ADR-001 e usado no E04 e na Parte 2.
- **Por que não escolher o E01, já que ele foi um pouco melhor no LFW?** Porque isso seria usar o teste para
  escolher o modelo (item 29). O critério foi definido **antes** dos experimentos, e trocá-lo depois de ver o
  resultado do teste seria data leakage. O resultado do LFW é reportado como limitação, não usado na escolha.
- Frase-chave para o vídeo: *"escolhemos pela validação, não pelo teste; o LFW só confirma — e, neste caso,
  ele mostrou que a vantagem é pequena"*.

---

## 8. Mapa do código

| Arquivo | O que faz |
|---|---|
| `configs/base.yaml`, `e0x.yaml` | Todos os caminhos e hiperparâmetros. Nada fixo no código. |
| `src/facerec/preprocess.py` | MTCNN: detecta, recorta e salva 160×160 (LFW, CelebA, privado). |
| `src/facerec/data.py` | Leitura dos datasets, seleção do subconjunto, split por identidade, pares, augmentations. |
| `src/facerec/overlap.py` | Remove do treino as quase-duplicatas CelebA × LFW. |
| `src/facerec/model.py` | InceptionResnetV1, estratégias none/partial/full, embeddings, cosseno. |
| `src/facerec/losses.py` | ArcFace e Triplet semi-hard. |
| `src/facerec/train.py` | Laço de treino, aquecimento, AMP, melhor época pela validação. |
| `src/facerec/metrics.py` | FAR, FRR, EER, ROC, TAR@FAR, threshold, 10 folds. |
| `src/facerec/evaluate.py` | Validação → threshold → teste no LFW; grava `results.csv` e gráficos. |
| `src/facerec/evaluate_private.py` | E04: cadastro S1, consulta S2, desconhecidos; saída só agregada. |
| `src/facerec/organize_private.py` | Organiza as fotos privadas com IDs pseudônimos. |
| `tests/` | Testes com dados sintéticos (métricas, perdas, splits, reprodutibilidade, privacidade). |

## 9. Perguntas rápidas para treinar

1. Qual a diferença entre verificação e identificação?
2. O que é um embedding e por que não usamos uma classificação comum?
3. Por que o LFW não pode ser usado para escolher o threshold?
4. O que a margem *m* da ArcFace faz, intuitivamente?
5. Por que a 1ª época treina só a cabeça?
6. Por que o BatchNorm das camadas congeladas fica em `eval()`?
7. O que acontece com o FAR e o FRR quando o threshold sobe?
8. Por que o critério de threshold é FAR ≤ 1%, e não o EER?
9. O que é data leakage e quais três cuidados tomamos contra ele?
10. Por que o dataset privado é separado por sessão?
11. Por que o fine-tuning com ArcFace piorou e o com Triplet melhorou?
12. Por que o E05 foi escolhido mesmo tendo um resultado um pouco pior no LFW?
13. No E04, como o AUC pode ser 1,0 se 36% dos desconhecidos foram aceitos?
