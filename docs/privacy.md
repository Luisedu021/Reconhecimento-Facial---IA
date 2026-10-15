# Privacidade e LGPD

Reconhecimento facial trata dados biométricos, que a LGPD (Lei nº 13.709/2018, art. 5º, II) classifica
como dados pessoais sensíveis. Neste projeto, privacidade é pré-requisito: violar estas regras zera o trabalho.

Base legal: consentimento específico e destacado do titular (art. 11, I), para finalidade exclusivamente acadêmica.

## Inventário de dados

| Dado | Classificação | Onde fica | Pode ir para o GitHub? |
|---|---|---|---|
| Código, specs, ADRs, configs, testes | público | repositório | sim |
| Métricas e gráficos **agregados** | público | `reports/`, README | sim |
| Datasets públicos (LFW, CelebA) | terceiros, com termos de uso | `data/public/` | não (só instruções de download) |
| Fotos e vídeos dos participantes | **privado / sensível** | `data/private/` | **não** |
| Tabela ID pseudônimo ↔ nome | **privado** | junto dos termos, fora do repositório | **não** |
| Embeddings, banco vetorial, cadastro | **privado / sensível** | `data/private/`, `*.db` | **não** |
| Modelos/classificadores treinados ou calibrados com dados privados | **privado** | `checkpoints/` | **não** |
| Logs e resultados individualizados | **privado** | local | **não** |
| Termos de consentimento assinados | **privado** | local seguro, fora de nuvem | **não** |

## Camadas de proteção

1. **Localização:** repositório e dados fora de pastas sincronizadas com nuvem (OneDrive, Google Drive, Dropbox).
2. **`.gitignore`:** ignora `data/`, diretórios privados e formatos de foto, vídeo, embedding, modelo e banco.
3. **Pre-commit (`.pre-commit-config.yaml`):** bloqueia o commit desses arquivos mesmo com `git add -f`
   e remove saídas de notebooks (que podem conter imagens).
4. **Agentes de IA (`.claude/settings.json` + `CLAUDE.md`):** o Claude Code é impedido de ler dados privados
   e instruído a usar apenas código, specs e dados sintéticos.
5. **Serviços externos:** dados privados não vão para APIs, trackers em nuvem, Colab/Kaggle nem chats de IA.
   Experimentos só com datasets públicos podem rodar em nuvem.

## Exclusão de dados

- **Por pessoa (retirada de consentimento):** remover `data/private/<ID>/`, embeddings e cadastro da pessoa,
  e descartar ou retreinar qualquer modelo/classificador que tenha usado seus dados. Na Parte 2, a aplicação
  oferece a função de exclusão (ver spec correspondente).
- **Fim da disciplina:** apagar fotos, embeddings, bancos, checkpoints privados, logs e arquivos temporários
  derivados, nas máquinas de **todos** os integrantes.

## Verificação do histórico do Git

Adicionar algo ao `.gitignore` depois não remove versões já commitadas. Antes de cada entrega, listar todos
os arquivos que já passaram pelo histórico e revisar o resultado:

```bash
git log --all --name-only --pretty=format: | sort -u | grep -Ei '\.(jpe?g|png|heic|webp|mp4|mov|npy|npz|pkl|pt|pth|ckpt|onnx|safetensors|db|sqlite3?|pdf|docx?)$|^data/|faces/|embeddings/|checkpoints/|consent'
```

PowerShell:

```powershell
git log --all --name-only --pretty=format: | Sort-Object -Unique | Select-String -Pattern '\.(jpe?g|png|heic|webp|mp4|mov|npy|npz|pkl|pt|pth|ckpt|onnx|safetensors|db|sqlite3?|pdf|docx?)$|^data/|faces/|embeddings/|checkpoints/|consent'
```

O único resultado esperado são PNGs em `reports/figures/` (gráficos agregados).

## Se algo privado vazar

1. Tornar o repositório privado imediatamente.
2. Remover o arquivo de **todo** o histórico (`git filter-repo`) e forçar o push; pedir ao suporte do GitHub
   a remoção de caches e referências antigas.
3. Considerar o dado como exposto: comunicar o professor e as pessoas afetadas.

## Checklist de entrega (Anexo B do enunciado)

- [ ] Todas as pessoas participantes deram consentimento.
- [ ] Os termos de consentimento estão armazenados de forma privada.
- [ ] Nenhum termo de consentimento está no GitHub.
- [ ] Nenhuma fotografia privada está no GitHub.
- [ ] Nenhum embedding privado está no GitHub.
- [ ] Nenhum banco biométrico está no GitHub.
- [ ] Nenhum checkpoint derivado dos dados privados está no GitHub.
- [ ] O `.gitignore` protege os diretórios privados.
- [ ] O histórico do Git foi verificado.
- [ ] As licenças dos datasets públicos foram verificadas.
- [ ] O README informa a origem dos datasets públicos.
- [ ] Os resultados publicados não identificam os participantes.
- [ ] O vídeo respeita as autorizações concedidas.
- [ ] Os dados privados não foram enviados inadvertidamente para serviços externos.
- [ ] Existe uma forma de excluir uma pessoa cadastrada.
- [ ] O grupo conhece as regras de proteção de dados aplicáveis ao projeto.
