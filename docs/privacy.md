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

**Última verificação: 2026-10-25** (histórico completo, todos os ramos, até o commit `30e2637`):

- O comando acima listou só `data/README.md` e os 9 PNGs de `reports/figures/`. O PNG do E04 foi aberto e
  contém apenas histogramas agregados (15 pares mesma pessoa, 60 pares pessoas diferentes), sem IDs nem imagens.
- Nenhum arquivo em `dataset/`, `data/private/`, `faces/`, `embeddings/`, `database/`, `checkpoints/` ou `consent/`
  em nenhum commit; nenhum arquivo acima de 500 KB.
- Busca no conteúdo de todos os commits por IDs com resultado individual ou caminhos de fotos privadas: só
  referências genéricas em instruções e configs.
- As linhas do E04 em `reports/results.csv` têm só as chaves agregadas do CA-6.
- `git check-ignore` confirma que `data/private/`, `dataset/`, `checkpoints/`, `data/public/` e `consent/` estão
  ignorados; o hook do pre-commit está instalado. O `.pre-commit-config.yaml` só foi alterado para ampliar o
  bloqueio (`dataset/` e `datasets/`, commit `afe3d79`); nenhuma regra foi removida do `.gitignore`.

## Se algo privado vazar

1. Tornar o repositório privado imediatamente.
2. Remover o arquivo de **todo** o histórico (`git filter-repo`) e forçar o push; pedir ao suporte do GitHub
   a remoção de caches e referências antigas.
3. Considerar o dado como exposto: comunicar o professor e as pessoas afetadas.

## Checklist de entrega (Anexo B do enunciado)

- [x] Todas as pessoas participantes deram consentimento.
- [x] Os termos de consentimento estão armazenados de forma privada.
- [x] Nenhum termo de consentimento está no GitHub.
- [x] Nenhuma fotografia privada está no GitHub.
- [x] Nenhum embedding privado está no GitHub.
- [x] Nenhum banco biométrico está no GitHub.
- [x] Nenhum checkpoint derivado dos dados privados está no GitHub.
- [x] O `.gitignore` protege os diretórios privados.
- [x] O histórico do Git foi verificado.
- [x] As licenças dos datasets públicos foram verificadas.
- [x] O README informa a origem dos datasets públicos.
- [x] Os resultados publicados não identificam os participantes.
- [ ] O vídeo respeita as autorizações concedidas.
- [ ] Os dados privados não foram enviados inadvertidamente para serviços externos.
- [ ] Existe uma forma de excluir uma pessoa cadastrada.
- [ ] O grupo conhece as regras de proteção de dados aplicáveis ao projeto.
