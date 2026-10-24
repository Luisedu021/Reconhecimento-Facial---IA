# Dados

**Nenhum dado é versionado.** Tudo dentro de `data/` (exceto este arquivo) é ignorado pelo Git
e bloqueado pelo hook de pre-commit.

## Estrutura local esperada

```
data/
├── README.md        (único arquivo versionado)
├── public/          datasets públicos, baixados da fonte oficial
│   ├── lfw/
│   └── celeba/
└── private/         dataset privado do grupo (LGPD — ver docs/privacy.md)
```

## Datasets públicos

Baixe sempre da **fonte oficial** e leia os termos de uso antes. Não redistribua os arquivos.

| Dataset | Fonte oficial | Arquivos necessários | Onde colocar |
|---|---|---|---|
| LFW | http://vis-www.cs.umass.edu/lfw/ | `lfw.tgz` (todas as imagens) e `pairs.txt` (View 2) | extrair em `data/public/lfw/` → `data/public/lfw/lfw/<Nome>/<Nome>_0001.jpg`; `data/public/lfw/pairs.txt` |
| CelebA | https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html (link para o Google Drive oficial) | `Img/img_align_celeba.zip` e `Anno/identity_CelebA.txt` | extrair em `data/public/celeba/img_align_celeba/`; `data/public/celeba/identity_CelebA.txt` |

Baixe pelo navegador: o download automático do CelebA costuma falhar por limite de cota do Google Drive.
Os caminhos podem ser trocados em [configs/base.yaml](../configs/base.yaml).

O pré-processamento (`python -m facerec.preprocess`) grava os rostos recortados em `data/public/cache/`
e a divisão treino/validação do CelebA em `data/public/cache/celeba/split.csv`.

## Dataset privado (coleta local)

Só entram pessoas **adultas** que assinaram o Termo de Consentimento (Anexo A do enunciado).
Modelo de convite e cuidados na coleta: [docs/convite-participantes.md](../docs/convite-participantes.md).
Os termos ficam guardados **fora** deste repositório e fora de serviços de nuvem.

Organização sugerida, com IDs pseudônimos (sem nomes nos caminhos):

```
data/private/
└── P01/
    ├── S1/          sessão de captura 1 (ex.: um dia/local)
    │   └── 0001.jpg
    └── S2/          sessão de captura 2 (outro dia/local)
```

- **Organização automática:** com as fotos numa pasta por pessoa (ex.: `dataset/<pessoa>/`), o comando abaixo
  cria os IDs `P01`, `P02`, ..., divide as fotos em S1 (as 2 mais antigas) e S2 (as demais) e renomeia os arquivos.
  A tabela pessoa → ID vai para um CSV **fora** do repositório e da nuvem. Rode você mesmo: a saída só mostra contagens.

  ```
  python -m facerec.organize_private --origem dataset --mapa C:\dados-privados\mapa-participantes.csv
  ```

  A divisão S1/S2 é feita pela data das fotos: confira as pastas depois e corrija à mão se algo estiver trocado.
- Formatos aceitos: JPG ou PNG (HEIC do iPhone precisa ser convertido antes).
- O pré-processamento grava os recortes em `data/private/cache/` (também fora do Git). Os embeddings
  das fotos privadas **não** são gravados em disco: são calculados em memória a cada avaliação.
- Para excluir uma pessoa: apagar `data/private/<ID>/` e `data/private/cache/<ID>/`.
- A tabela que liga `P01` ao nome real fica **junto dos termos**, nunca no repositório.
- Separar treino/validação/teste **por sessão** evita fotos quase idênticas em conjuntos diferentes.
- Variar iluminação, pose, distância, expressão, fundo e uso de óculos.
- Coletar só o necessário (minimização de dados).
- Dados pseudonimizados continuam sendo dados pessoais: as mesmas regras se aplicam.
