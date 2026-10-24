"""Fine-tuning no subconjunto de treino do CelebA (E02, E03, E05).

Uso:
    python -m facerec.train --config configs/e02.yaml

A cada época mede a validação (identidades do CelebA nunca vistas no treino) e guarda o
checkpoint da melhor época em checkpoints/<experimento>.pt. O LFW não é tocado aqui.
"""

from __future__ import annotations

import argparse
import csv
import functools
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import load_config, resolve, set_seed
from .data import ImageListDataset, PKSampler
from .evaluate import celeba_val_pairs, checkpoint_path, load_celeba, validate
from .losses import ArcFaceHead, semi_hard_triplet_loss
from .model import apply_strategy, build_backbone, count_trainable, param_groups, set_train_mode


def seed_worker(worker_id: int) -> None:
    """Cada worker do DataLoader recebe uma seed derivada da principal (augmentations reproduzíveis)."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def train_one_epoch(model, loss_fn, loader, optimizer, scaler, device, backbone_frozen: bool = False) -> float:
    """Uma passada pelos dados de treino. Retorna a perda média.

    `backbone_frozen` = aquecimento: os embeddings saem sem gradiente e só a cabeça
    ArcFace (pesos novos, aleatórios) aprende. Evita que gradientes de uma cabeça ainda
    aleatória estraguem o backbone pré-treinado logo no começo.
    """
    losses = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            if backbone_frozen:
                with torch.no_grad():
                    emb = model(x)
            else:
                emb = model(x)
        loss = loss_fn(emb, y)  # as perdas calculam em float32 (ver losses.py)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())
    return float(np.mean(losses))


def build_loader(cfg: dict, paths, labels) -> DataLoader:
    t = cfg["train"]
    dataset = ImageListDataset(paths, labels, train=True)
    common = {"num_workers": t["num_workers"], "worker_init_fn": seed_worker, "pin_memory": torch.cuda.is_available(),
              "persistent_workers": t["num_workers"] > 0}
    if t["loss"] == "triplet":
        sampler = PKSampler(labels, p=t["pk"]["p"], k=t["pk"]["k"], seed=cfg["seed"])
        return DataLoader(dataset, batch_sampler=sampler, **common)
    generator = torch.Generator().manual_seed(cfg["seed"])
    return DataLoader(dataset, batch_size=t["batch_size"], shuffle=True, drop_last=True, generator=generator, **common)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    t = cfg.get("train")
    if not t:
        raise SystemExit(f"{cfg['experiment']} não tem treino (baseline); use: python -m facerec.evaluate --config {args.config}")
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_paths, train_labels, val_groups = load_celeba(cfg)
    val_pairs = celeba_val_pairs(cfg, val_groups)
    loader = build_loader(cfg, train_paths, train_labels)
    n_classes = max(train_labels) + 1
    print(f"{cfg['experiment']}: {len(train_paths)} imagens de treino, {n_classes} identidades, "
          f"{len(val_pairs)} pares de validação")

    model = build_backbone(cfg["model"]["pretrained"]).to(device)
    apply_strategy(model, t["strategy"])
    groups = param_groups(model, t["strategy"], t["lr"], t["lr_backbone"])

    if t["loss"] == "arcface":
        head = ArcFaceHead(512, n_classes, s=t["arcface"]["s"], m=t["arcface"]["m"]).to(device)
        groups.append({"params": head.parameters(), "lr": t["lr_head"]})
        loss_fn = head
    elif t["loss"] == "triplet":
        head = None
        loss_fn = functools.partial(semi_hard_triplet_loss, margin=t["triplet"]["margin"])
    else:
        raise ValueError(f"perda desconhecida: {t['loss']}")

    optimizer = torch.optim.AdamW(groups, weight_decay=t["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    trainable = count_trainable(model) + (count_trainable(head) if head else 0)

    history_path = resolve(cfg["paths"]["reports"]) / f"{cfg['experiment']}-historico.csv"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    selection = t["selection_metric"]  # "eer" (menor é melhor) ou "tar_at_far" (maior é melhor)
    better = (lambda new, old: new < old) if selection == "eer" else (lambda new, old: new > old)
    best, history = None, []
    ckpt = checkpoint_path(cfg)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    for epoch in range(1, t["epochs"] + 1):
        warmup = head is not None and epoch <= t["warmup_epochs"]
        if warmup:
            model.eval()  # backbone congelado inteiro durante o aquecimento
        else:
            set_train_mode(model, t["strategy"])
        loss = train_one_epoch(model, loss_fn, loader, optimizer, scaler, device, backbone_frozen=warmup)
        val = validate(model, cfg, val_pairs, device)
        history.append({"epoch": epoch, "warmup": warmup, "loss": loss, "val_eer": val["eer"], "val_tar_at_far": val["tar_at_far"]})
        print(f"  época {epoch}{' (aquecimento)' if warmup else ''}: perda {loss:.4f} | "
              f"val EER {val['eer']:.2%} | val TAR@FAR {val['tar_at_far']:.2%}")

        if best is None or better(val[selection], best):
            best = val[selection]
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "val": val, "trainable_params": trainable, "config": cfg},
                ckpt,
            )

    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    # Custo computacional (item 33): tempo total do treino, gravado junto do melhor checkpoint.
    total = time.perf_counter() - start
    state = torch.load(ckpt, map_location="cpu")
    state["train_time_s"] = total
    torch.save(state, ckpt)
    print(f"{cfg['experiment']}: treino em {total / 60:.1f} min; melhor {selection} na validação = {best:.4f}; "
          f"checkpoint em {ckpt.relative_to(resolve('.'))}")


if __name__ == "__main__":
    main()
