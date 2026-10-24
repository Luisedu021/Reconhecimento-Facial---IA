"""Detecção, alinhamento e recorte das faces, feitos uma única vez por imagem (RF-1, RF-2).

Uso:
    python -m facerec.preprocess --config configs/base.yaml --dataset lfw
    python -m facerec.preprocess --config configs/base.yaml --dataset celeba
    python -m facerec.preprocess --config configs/base.yaml --dataset private

O MTCNN encontra o rosto e 5 pontos de referência (olhos, nariz, cantos da boca), recorta
a região com uma margem e redimensiona para 160x160. Todos os conjuntos passam por este
mesmo código, para que treino, teste e cadastro vejam rostos preparados do mesmo jeito.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from tqdm import tqdm

from .config import load_config, resolve, set_seed
from .data import (
    list_private_images,
    read_celeba_identities,
    select_identity_subset,
    split_by_identity,
    write_split,
)


def build_detector(cfg: dict, device: torch.device, selection: str):
    from facenet_pytorch import MTCNN

    p = cfg["preprocess"]
    return MTCNN(
        image_size=p["image_size"],
        margin=p["margin"],
        min_face_size=p["min_face_size"],
        post_process=False,  # mantém 0-255; a normalização é feita no carregamento (data.Standardize)
        select_largest=selection == "largest",
        selection_method=selection,
        keep_all=False,
        device=device,
    )


def save_face(face: torch.Tensor, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = face.permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()
    # PNG novo, sem EXIF: a localização GPS da foto original não é copiada para o cache.
    Image.fromarray(arr).save(out)


def center_crop(img: Image.Image, size: int) -> torch.Tensor:
    """Plano B quando o MTCNN não acha rosto: recorte central (só no LFW, onde o rosto está centralizado)."""
    w, h = img.size
    side = min(w, h) * 0.6
    box = ((w - side) / 2, (h - side) / 2, (w + side) / 2, (h + side) / 2)
    arr = np.asarray(img.crop(box).resize((size, size), Image.BILINEAR), dtype=np.float32)
    return torch.from_numpy(arr).permute(2, 0, 1)


def run_batched(detector, sources: list[Path], outputs: list[Path], batch_size: int, fallback_size: int | None):
    """Imagens do mesmo tamanho (LFW, CelebA) são detectadas em lotes na GPU."""
    status = []
    for start in tqdm(range(0, len(sources), batch_size), desc="faces"):
        src, dst = sources[start : start + batch_size], outputs[start : start + batch_size]
        imgs = [Image.open(s).convert("RGB") for s in src]
        faces = detector(imgs)
        for img, face, out in zip(imgs, faces, dst):
            if face is None and fallback_size:
                face, ok = center_crop(img, fallback_size), "fallback"
            else:
                ok = "ok" if face is not None else "sem_face"
            if face is not None:
                save_face(face, out)
            status.append(ok)
    return status


def write_manifest(path: Path, rows: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "status"])
        writer.writerows(rows)


def preprocess_lfw(cfg: dict, device: torch.device) -> None:
    src_root = resolve(cfg["paths"]["lfw_images"])
    out_root = resolve(cfg["paths"]["public_cache"]) / "lfw"
    sources = sorted(src_root.glob("*/*.jpg"))
    rel = [s.relative_to(src_root).as_posix() for s in sources]
    outputs = [out_root / (r[:-4] + ".png") for r in rel]
    detector = build_detector(cfg, device, selection="center_weighted_size")
    status = run_batched(detector, sources, outputs, cfg["preprocess"]["batch_size"], cfg["preprocess"]["image_size"])
    write_manifest(out_root / "manifest.csv", list(zip(rel, status)))
    print(f"LFW: {len(rel)} imagens; {status.count('fallback')} sem rosto detectado (recorte central)")


def preprocess_celeba(cfg: dict, device: torch.device) -> None:
    c = cfg["celeba"]
    src_root = resolve(cfg["paths"]["celeba_images"])
    out_root = resolve(cfg["paths"]["public_cache"]) / "celeba"
    identities = read_celeba_identities(resolve(cfg["paths"]["celeba_identities"]))
    subset = select_identity_subset(identities, c["num_identities"], c["min_images"], c["max_images"], cfg["seed"])
    train_ids, val_ids = split_by_identity(list(subset), c["val_fraction"], cfg["seed"])
    split_of = {i: "train" for i in train_ids} | {i: "val" for i in val_ids}

    images = [(img, ident) for ident, imgs in subset.items() for img in imgs]
    sources = [src_root / img for img, _ in images]
    outputs = [out_root / (img[:-4] + ".png") for img, _ in images]
    detector = build_detector(cfg, device, selection="largest")
    status = run_batched(detector, sources, outputs, cfg["preprocess"]["batch_size"], fallback_size=None)

    # Imagens sem rosto detectado ficam fora do split.
    rows = [(img[:-4] + ".png", ident, split_of[ident]) for (img, ident), s in zip(images, status) if s == "ok"]
    write_split(out_root / "split.csv", rows)
    write_manifest(out_root / "manifest.csv", [(img, s) for (img, _), s in zip(images, status)])
    print(
        f"CelebA: {len(subset)} identidades ({len(train_ids)} treino, {len(val_ids)} validação); "
        f"{len(rows)} imagens com rosto; {status.count('sem_face')} descartadas"
    )


def preprocess_private(cfg: dict, device: torch.device) -> None:
    """Fotos privadas têm tamanhos diferentes: uma por vez. Saída em data/private/cache (fora do Git)."""
    src_root = resolve(cfg["paths"]["private_images"])
    out_root = resolve(cfg["paths"]["private_cache"])
    detector = build_detector(cfg, device, selection="largest")
    people = list_private_images(src_root)
    rows, failed = [], []
    for person, sessions in people.items():
        for session, files in sessions.items():
            for f in files:
                # Fotos de celular guardam a rotação no EXIF; sem isso a selfie pode chegar deitada.
                img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
                face = detector(img)
                rel = f"{person}/{session}/{f.stem}.png"
                if face is None:
                    failed.append(rel)
                    continue
                save_face(face, out_root / rel)
                rows.append((rel, "ok"))
    write_manifest(out_root / "manifest.csv", rows)
    print(f"Privado: {len(people)} pessoas, {len(rows)} rostos recortados, {len(failed)} sem rosto detectado")
    # Só IDs pseudônimos, para saber qual foto refazer. Não cole esta saída em serviços externos.
    for rel in failed:
        print(f"  sem rosto: {rel}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--dataset", choices=["lfw", "celeba", "private"], required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    {"lfw": preprocess_lfw, "celeba": preprocess_celeba, "private": preprocess_private}[args.dataset](cfg, device)


if __name__ == "__main__":
    main()
