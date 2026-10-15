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

| Dataset | Fonte oficial | Destino local |
|---|---|---|
| LFW | http://vis-www.cs.umass.edu/lfw/ | `data/public/lfw/` |
| CelebA | https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html | `data/public/celeba/` |

> _Instruções detalhadas de download (arquivos necessários, script, checagem de integridade) a preencher._

## Dataset privado (coleta local)

Só entram pessoas **adultas** que assinaram o Termo de Consentimento (Anexo A do enunciado).
Os termos ficam guardados **fora** deste repositório e fora de serviços de nuvem.

Organização sugerida, com IDs pseudônimos (sem nomes nos caminhos):

```
data/private/
└── P01/
    ├── S1/          sessão de captura 1 (ex.: um dia/local)
    │   └── 0001.jpg
    └── S2/          sessão de captura 2 (outro dia/local)
```

- A tabela que liga `P01` ao nome real fica **junto dos termos**, nunca no repositório.
- Separar treino/validação/teste **por sessão** evita fotos quase idênticas em conjuntos diferentes.
- Variar iluminação, pose, distância, expressão, fundo e uso de óculos.
- Coletar só o necessário (minimização de dados).
- Dados pseudonimizados continuam sendo dados pessoais: as mesmas regras se aplicam.
