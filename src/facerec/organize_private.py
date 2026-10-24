"""Organiza as fotos privadas em data/private/<ID>/<S1|S2>/ com IDs pseudônimos.

Uso (rode VOCÊ MESMO, localmente; a saída nunca mostra nomes de pessoas nem de arquivos):
    python -m facerec.organize_private --origem dataset --mapa C:\\dados-privados\\mapa-participantes.csv

Espera uma subpasta por pessoa em --origem (ex.: dataset/<pessoa>/*.jpg). Para cada pessoa:
- gera o ID P01, P02, ... na ordem alfabética das pastas;
- ordena as fotos pela data de captura (EXIF) ou, sem EXIF, pela data do arquivo e depois pelo nome;
- as 2 primeiras vão para S1 (cadastro) e as demais para S2 (consulta);
- renomeia para 1.jpg, 2.jpg, ... (nenhum nome de pessoa nos caminhos).

A tabela "pasta original -> ID" vai para --mapa, que precisa ficar FORA do repositório,
junto dos termos de consentimento. A divisão S1/S2 é uma estimativa pela data: confira
abrindo as pastas e corrija à mão se algo estiver trocado.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from datetime import datetime
from pathlib import Path

from PIL import Image

from .config import ROOT, load_config, resolve
from .data import IMAGE_SUFFIXES

CLOUD_HINTS = ("onedrive", "google drive", "googledrive", "dropbox", "icloud")
SYSTEM_FILES = {"desktop.ini", "thumbs.db", ".ds_store"}
EXIF_IFD, DATETIME_ORIGINAL, DATETIME = 0x8769, 0x9003, 0x0132


def capture_key(path: Path) -> tuple[str, str]:
    """Chave de ordenação: data de captura do EXIF; sem EXIF, a data de modificação do arquivo."""
    stamp = ""
    try:
        exif = Image.open(path).getexif()
        stamp = exif.get_ifd(EXIF_IFD).get(DATETIME_ORIGINAL) or exif.get(DATETIME) or ""
    except Exception:  # arquivo sem EXIF ou ilegível: cai para a data do arquivo
        stamp = ""
    if not stamp:
        stamp = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y:%m:%d %H:%M:%S")
    return str(stamp), path.name.lower()


def check_map_location(mapa: Path) -> None:
    """O mapa liga IDs a pessoas: não pode ficar no repositório nem em pasta sincronizada com nuvem."""
    resolved = mapa.resolve()
    if resolved == ROOT or ROOT in resolved.parents:
        raise ValueError("o mapa de participantes não pode ficar dentro do repositório")
    if any(hint in str(resolved).lower() for hint in CLOUD_HINTS):
        raise ValueError("o mapa de participantes não pode ficar numa pasta sincronizada com nuvem")


def _remove_empty_dirs(folder: Path) -> None:
    """Apaga as pastas que ficaram vazias depois da mudança (de baixo para cima)."""
    for d in sorted((p for p in folder.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        if not any(d.iterdir()):
            d.rmdir()
    if not any(folder.iterdir()):
        folder.rmdir()


def organize(origem: Path, destino: Path, mapa: Path, n_s1: int = 2) -> dict[str, dict[str, int]]:
    """Move as fotos e devolve só contagens por ID pseudônimo."""
    check_map_location(mapa)
    people = sorted(p for p in origem.iterdir() if p.is_dir())
    if not people:
        loose = sum(1 for f in origem.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES)
        raise SystemExit(
            f"Nada foi movido: {loose} foto(s) soltas em {origem.name}/ e nenhuma subpasta por pessoa.\n"
            f"Crie uma subpasta por pessoa dentro de {origem.name}/ (ex.: pessoa1, pessoa2, ...), "
            "arraste as fotos de cada uma para a sua pasta e rode o comando de novo."
        )
    existing = [d for d in destino.glob("P*") if d.is_dir() and any(d.rglob("*"))] if destino.exists() else []
    if existing:
        raise ValueError(f"{destino} já tem pastas de participantes; nada foi movido para não sobrescrever")

    summary, rows = {}, []
    for i, person in enumerate(people, start=1):
        pid = f"P{i:02d}"
        # rglob: aceita fotos soltas na pasta da pessoa ou em subpastas (ex.: uma por dia).
        files = [f for f in person.rglob("*") if f.is_file() and f.name.lower() not in SYSTEM_FILES]
        photos = sorted((f for f in files if f.suffix.lower() in IMAGE_SUFFIXES), key=capture_key)
        skipped = sum(1 for f in files if f.suffix.lower() not in IMAGE_SUFFIXES)
        counts = {"S1": 0, "S2": 0, "nao_suportadas": skipped}
        for n, photo in enumerate(photos, start=1):
            session = "S1" if n <= n_s1 else "S2"
            target = destino / pid / session / f"{counts[session] + 1}{photo.suffix.lower()}"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(photo), str(target))
            counts[session] += 1
        summary[pid] = counts
        rows.append((person.name, pid))
        _remove_empty_dirs(person)

    mapa.parent.mkdir(parents=True, exist_ok=True)
    with open(mapa, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pasta_original", "id"])
        writer.writerows(rows)
    if origem.exists() and not any(origem.iterdir()):
        origem.rmdir()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--origem", default="dataset", help="pasta com uma subpasta por pessoa")
    parser.add_argument("--mapa", required=True, help="CSV pasta -> ID, FORA do repositório e da nuvem")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--fotos-s1", type=int, default=2, help="quantas fotos de cada pessoa vão para o cadastro")
    args = parser.parse_args()
    cfg = load_config(args.config)
    summary = organize(resolve(args.origem), resolve(cfg["paths"]["private_images"]), Path(args.mapa), args.fotos_s1)

    total = sum(c["S1"] + c["S2"] for c in summary.values())
    print(f"{len(summary)} pessoas, {total} fotos movidas para {cfg['paths']['private_images']}/ (só IDs pseudônimos):")
    for pid, c in summary.items():
        aviso = f"  <- {c['nao_suportadas']} arquivo(s) não suportado(s) ficaram na origem (HEIC? converta para JPG)" if c["nao_suportadas"] else ""
        print(f"  {pid}: S1 = {c['S1']}, S2 = {c['S2']}{aviso}")
    print("Mapa pasta -> ID gravado no caminho indicado em --mapa. Guarde-o junto dos termos.")
    print("Confira as pastas S1 e S2 de cada ID: a divisão foi feita pela data das fotos.")


if __name__ == "__main__":
    main()
