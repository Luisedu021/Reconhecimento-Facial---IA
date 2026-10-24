"""Datasets, divisões e checagens de data leakage (SPEC-001: Dados e divisão, CA-1 a CA-4)."""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms

# ---------------------------------------------------------------------------
# Pré-processamento das imagens já recortadas (160x160) para a rede
# ---------------------------------------------------------------------------


class Standardize:
    """Converte PIL (0-255) para tensor e aplica a normalização usada no treino do FaceNet: (x - 127.5) / 128."""

    def __call__(self, img: Image.Image) -> torch.Tensor:
        t = torch.from_numpy(np.asarray(img, dtype=np.float32).copy()).permute(2, 0, 1)
        return (t - 127.5) / 128.0


def build_transform(train: bool) -> transforms.Compose:
    """Augmentations só no treino (CA-3). Validação, teste e cadastro usam exatamente o mesmo caminho."""
    steps = []
    if train:
        steps += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ]
    steps.append(Standardize())
    return transforms.Compose(steps)


class ImageListDataset(Dataset):
    """Lista de imagens recortadas com rótulo inteiro (identidade) opcional."""

    def __init__(self, paths: list[Path], labels: list[int] | None = None, train: bool = False):
        self.paths = [Path(p) for p in paths]
        self.labels = labels if labels is not None else [-1] * len(self.paths)
        self.transform = build_transform(train)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.labels[i]


# ---------------------------------------------------------------------------
# CelebA: subconjunto de identidades e divisão por identidade
# ---------------------------------------------------------------------------


def read_celeba_identities(path: Path) -> dict[str, int]:
    """identity_CelebA.txt: uma linha por imagem, "000001.jpg 2880"."""
    mapping = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                mapping[parts[0]] = int(parts[1])
    return mapping


def select_identity_subset(
    image_to_id: dict[str, int], num_identities: int, min_images: int, max_images: int, seed: int
) -> dict[int, list[str]]:
    """Sorteia `num_identities` identidades com pelo menos `min_images` fotos e guarda até `max_images` de cada."""
    by_id: dict[int, list[str]] = defaultdict(list)
    for image, identity in image_to_id.items():
        by_id[identity].append(image)
    eligible = sorted(i for i, imgs in by_id.items() if len(imgs) >= min_images)
    if len(eligible) < num_identities:
        raise ValueError(f"só {len(eligible)} identidades têm >= {min_images} imagens")
    rng = random.Random(seed)
    chosen = rng.sample(eligible, num_identities)
    return {i: sorted(by_id[i])[:max_images] for i in sorted(chosen)}


def split_by_identity(identities: list[int], val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Divide por IDENTIDADE, não por imagem: quem está na validação nunca foi visto no treino (CA-1)."""
    ids = sorted(identities)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = max(1, round(len(ids) * val_fraction))
    return sorted(ids[n_val:]), sorted(ids[:n_val])


def make_pairs(groups: dict[int, list[str]], n_per_class: int, seed: int) -> list[tuple[str, str, int]]:
    """Gera `n_per_class` pares da mesma pessoa (1) e de pessoas diferentes (0), sem repetir pares."""
    rng = random.Random(seed)
    ids = sorted(i for i, imgs in groups.items() if len(imgs) >= 2)
    all_ids = sorted(groups)
    genuine: set[tuple[str, str]] = set()
    impostor: set[tuple[str, str]] = set()
    max_tries = n_per_class * 50
    tries = 0
    while len(genuine) < n_per_class and tries < max_tries:
        a, b = sorted(rng.sample(groups[rng.choice(ids)], 2))
        genuine.add((a, b))
        tries += 1
    tries = 0
    while len(impostor) < n_per_class and tries < max_tries:
        i, j = rng.sample(all_ids, 2)
        a, b = sorted((rng.choice(groups[i]), rng.choice(groups[j])))
        impostor.add((a, b))
        tries += 1
    pairs = [(a, b, 1) for a, b in sorted(genuine)] + [(a, b, 0) for a, b in sorted(impostor)]
    return pairs


def write_split(path: Path, rows: list[tuple[str, int, str]]) -> None:
    """Grava (imagem, identidade, split) para que a divisão seja auditável e reproduzível."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "identity", "split"])
        writer.writerows(rows)


def read_split(path: Path) -> list[tuple[str, int, str]]:
    with open(path, encoding="utf-8") as f:
        return [(r["image"], int(r["identity"]), r["split"]) for r in csv.DictReader(f)]


def assert_disjoint(split_rows: list[tuple[str, int, str]]) -> None:
    """Falha se alguma identidade ou imagem aparecer em mais de um split (CA-1, CA-2)."""
    ids_per_split: dict[str, set[int]] = defaultdict(set)
    imgs_per_split: dict[str, set[str]] = defaultdict(set)
    for image, identity, split in split_rows:
        ids_per_split[split].add(identity)
        imgs_per_split[split].add(image)
    splits = sorted(ids_per_split)
    for i, a in enumerate(splits):
        for b in splits[i + 1 :]:
            if ids_per_split[a] & ids_per_split[b]:
                raise AssertionError(f"identidades em comum entre {a} e {b}")
            if imgs_per_split[a] & imgs_per_split[b]:
                raise AssertionError(f"imagens em comum entre {a} e {b}")


# ---------------------------------------------------------------------------
# LFW: protocolo de pares (View 2)
# ---------------------------------------------------------------------------


def read_lfw_pairs(pairs_file: Path) -> list[tuple[str, str, int, int]]:
    """Lê pairs.txt (10 folds x 300 pares iguais + 300 diferentes).

    Retorna (imagem_a, imagem_b, label, fold) com caminhos relativos "Nome/Nome_0001.jpg".
    """
    with open(pairs_file, encoding="utf-8") as f:
        lines = [line.split() for line in f if line.strip()]
    n_folds, n_per = int(lines[0][0]), int(lines[0][1])
    pairs = []
    body = lines[1:]
    for fold in range(n_folds):
        block = body[fold * 2 * n_per : (fold + 1) * 2 * n_per]
        for parts in block:
            if len(parts) == 3:
                name, i, j = parts
                a, b, label = (name, int(i)), (name, int(j)), 1
            else:
                n1, i, n2, j = parts
                a, b, label = (n1, int(i)), (n2, int(j)), 0
            pairs.append((f"{a[0]}/{a[0]}_{a[1]:04d}.jpg", f"{b[0]}/{b[0]}_{b[1]:04d}.jpg", label, fold))
    return pairs


# ---------------------------------------------------------------------------
# Dataset privado: data/private/<ID>/<S1|S2>/*.jpg
# ---------------------------------------------------------------------------

PRIVATE_SESSIONS = ("S1", "S2")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def list_private_images(root: Path) -> dict[str, dict[str, list[Path]]]:
    """{ID pseudônimo: {sessão: [imagens]}}. Só pastas no formato P01, P02, ..."""
    people: dict[str, dict[str, list[Path]]] = {}
    for person in sorted(p for p in root.iterdir() if p.is_dir() and p.name[:1] == "P" and p.name[1:].isdigit()):
        people[person.name] = {
            s: sorted(f for f in (person / s).glob("*") if f.suffix.lower() in IMAGE_SUFFIXES)
            for s in PRIVATE_SESSIONS
            if (person / s).is_dir()
        }
    return people


# ---------------------------------------------------------------------------
# Amostragem P x K (Triplet loss precisa de positivos em cada batch)
# ---------------------------------------------------------------------------


class PKSampler(Sampler[list[int]]):
    """Cada batch tem P identidades com K imagens cada, para sempre existir par positivo."""

    def __init__(self, labels: list[int], p: int, k: int, seed: int):
        self.p, self.k, self.seed = p, k, seed
        self.by_label: dict[int, list[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            self.by_label[label].append(idx)
        self.labels = sorted(lbl for lbl, idxs in self.by_label.items() if len(idxs) >= k)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.labels) // self.p

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        labels = self.labels[:]
        rng.shuffle(labels)
        for b in range(len(self)):
            batch = []
            for lbl in labels[b * self.p : (b + 1) * self.p]:
                batch += rng.sample(self.by_label[lbl], self.k)
            yield batch


# ---------------------------------------------------------------------------
# Quase-duplicatas entre datasets (CA-4)
# ---------------------------------------------------------------------------


def near_duplicate_mask(a: torch.Tensor, b: torch.Tensor, threshold: float, chunk: int = 4096) -> np.ndarray:
    """Para cada linha de `a`, True se algum embedding de `b` tiver cosseno > threshold.

    `a` e `b` devem estar normalizados (L2). O cálculo é feito em blocos para não estourar a memória.
    """
    hits = np.zeros(len(a), dtype=bool)
    for start in range(0, len(a), chunk):
        sims = a[start : start + chunk] @ b.T
        hits[start : start + chunk] = (sims.max(dim=1).values > threshold).cpu().numpy()
    return hits
