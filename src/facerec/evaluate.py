"""Avaliação pública de um experimento: validação (CelebA-val) e teste (LFW).

Uso:
    python -m facerec.evaluate --config configs/e01.yaml

1. Escolhe o threshold SÓ na validação do CelebA (critério do YAML: FAR-alvo ou EER).
2. Mede o LFW no protocolo oficial de 10 folds e também com esse threshold fixo.
3. Grava as linhas em reports/results.csv, o threshold em reports/thresholds.json e os gráficos.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import metrics
from .config import environment_info, load_config, resolve, set_seed
from .data import assert_disjoint, make_pairs, read_lfw_pairs, read_split
from .model import build_backbone, cosine_scores, embed_paths, load_checkpoint
from .plots import plot_roc, plot_similarity
from .results import append_result

# ---------------------------------------------------------------------------
# CelebA: carregamento do split (usado também pelo treino)
# ---------------------------------------------------------------------------


def load_celeba(cfg: dict) -> tuple[list[Path], list[int], dict[int, list[str]]]:
    """Treino (caminhos + rótulos 0..N-1) e grupos de validação por identidade, sem as quase-duplicatas do LFW."""
    root = resolve(cfg["paths"]["public_cache"]) / "celeba"
    rows = read_split(root / "split.csv")
    assert_disjoint(rows)  # CA-1, CA-2: falha alto se o split estiver contaminado
    overlap_file = root / "overlap_lfw.txt"
    excluded = set(overlap_file.read_text(encoding="utf-8").split()) if overlap_file.exists() else set()
    rows = [r for r in rows if r[0] not in excluded]

    train_ids = sorted({ident for _, ident, split in rows if split == "train"})
    label_of = {ident: i for i, ident in enumerate(train_ids)}
    train_paths = [root / img for img, _, split in rows if split == "train"]
    train_labels = [label_of[ident] for _, ident, split in rows if split == "train"]
    val_groups: dict[int, list[str]] = {}
    for img, ident, split in rows:
        if split == "val":
            val_groups.setdefault(ident, []).append(img)
    return train_paths, train_labels, val_groups


def celeba_val_pairs(cfg: dict, val_groups: dict[int, list[str]]) -> list[tuple[str, str, int]]:
    return make_pairs(val_groups, cfg["celeba"]["val_pairs_per_class"], cfg["seed"])


def score_pairs(model, pairs, root: Path, device, batch_size: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Embeda cada imagem uma vez e devolve (scores, labels, ms por imagem)."""
    images = sorted({p[0] for p in pairs} | {p[1] for p in pairs})
    start = time.perf_counter()
    emb = embed_paths(model, [root / i for i in images], device, batch_size)
    ms_per_image = (time.perf_counter() - start) * 1000 / len(images)
    index = {img: i for i, img in enumerate(images)}
    scores = cosine_scores(emb, index, pairs)
    labels = np.array([p[2] for p in pairs])
    return scores, labels, ms_per_image


def choose_threshold(cfg: dict, scores: np.ndarray, labels: np.ndarray) -> float:
    """Threshold do sistema, escolhido na validação (item 32; ADR-003 define o critério)."""
    t = cfg["threshold"]
    if t["criterion"] == "far":
        return metrics.threshold_at_far(scores, labels, t["far_target"])
    if t["criterion"] == "eer":
        return metrics.eer(scores, labels)[1]
    raise ValueError(f"critério de threshold desconhecido: {t['criterion']}")


def validate(model, cfg: dict, val_pairs, device) -> dict[str, float]:
    """Métricas na validação; usada a cada época do treino para escolher o melhor checkpoint."""
    root = resolve(cfg["paths"]["public_cache"]) / "celeba"
    scores, labels, _ = score_pairs(model, val_pairs, root, device, cfg["eval"]["batch_size"])
    thr = choose_threshold(cfg, scores, labels)
    return metrics.verification_report(scores, labels, thr, cfg["threshold"]["far_target"])


# ---------------------------------------------------------------------------
# Modelo do experimento
# ---------------------------------------------------------------------------


def checkpoint_path(cfg: dict) -> Path:
    return resolve(cfg["paths"]["checkpoints"]) / f"{cfg['experiment']}.pt"


def load_experiment_model(cfg: dict, device) -> tuple[torch.nn.Module, dict]:
    """Pré-treinado (E01) ou o melhor checkpoint salvo pelo treino (E02, E03, E05)."""
    model = build_backbone(cfg["model"]["pretrained"])
    info = {"train_time_s": 0.0, "trainable_params": 0}
    if cfg.get("train"):
        path = checkpoint_path(cfg)
        if not path.exists():
            raise FileNotFoundError(f"{path} não existe; rode antes: python -m facerec.train --config ...")
        load_checkpoint(model, path)
        state = torch.load(path, map_location="cpu")
        info = {"train_time_s": state.get("train_time_s", 0.0), "trainable_params": state["trainable_params"]}
    return model.to(device).eval(), info


def base_row(cfg: dict, env: dict) -> dict:
    return {
        "experiment": cfg["experiment"],
        "seed": cfg["seed"],
        "versions": env["versions"],
        "hardware": env["hardware"],
        "config": cfg,
    }


def save_threshold(cfg: dict, threshold: float) -> None:
    path = resolve(cfg["paths"]["reports"]) / "thresholds.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[cfg["experiment"]] = {
        "threshold": threshold,
        "criterion": cfg["threshold"]["criterion"],
        "far_target": cfg["threshold"]["far_target"],
        "chosen_on": "celeba-val",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = environment_info()
    exp = cfg["experiment"]
    reports = resolve(cfg["paths"]["reports"])
    cache = resolve(cfg["paths"]["public_cache"])
    bs = cfg["eval"]["batch_size"]
    far_target = cfg["threshold"]["far_target"]

    model, info = load_experiment_model(cfg, device)

    # 1. Validação: o threshold nasce aqui.
    _, _, val_groups = load_celeba(cfg)
    val_pairs = celeba_val_pairs(cfg, val_groups)
    val_scores, val_labels, _ = score_pairs(model, val_pairs, cache / "celeba", device, bs)
    thr = choose_threshold(cfg, val_scores, val_labels)
    val = metrics.verification_report(val_scores, val_labels, thr, far_target)

    # 2. Teste no LFW: caminhos do pairs.txt apontam para os recortes .png do cache.
    lfw_pairs = [(a[:-4] + ".png", b[:-4] + ".png", y, fold) for a, b, y, fold in read_lfw_pairs(resolve(cfg["paths"]["lfw_pairs"]))]
    lfw_scores, lfw_labels, ms = score_pairs(model, lfw_pairs, cache / "lfw", device, bs)
    lfw = metrics.verification_report(lfw_scores, lfw_labels, thr, far_target)
    lfw["lfw_10fold_accuracy"], lfw["lfw_10fold_std"] = metrics.kfold_accuracy(
        lfw_scores, lfw_labels, [p[3] for p in lfw_pairs]
    )

    cost = {"ms_per_image": ms, "train_time_s": info["train_time_s"], "trainable_params": info["trainable_params"]}
    results_csv = reports / "results.csv"
    append_result(results_csv, {**base_row(cfg, env), "dataset": "celeba-val", **val, **cost})
    append_result(results_csv, {**base_row(cfg, env), "dataset": "lfw", **lfw, **cost})
    save_threshold(cfg, thr)

    # 3. Gráficos agregados.
    fig = reports / "figures"
    curves, points = {}, {}
    for label, (s, y, rep) in {"CelebA-val": (val_scores, val_labels, val), "LFW": (lfw_scores, lfw_labels, lfw)}.items():
        fpr, tpr, _ = metrics.roc(s, y)
        curves[label] = (fpr, tpr)
        points[label] = (rep["far"], 1 - rep["frr"])
    plot_roc(curves, points, f"{exp} — curva ROC", fig / f"{exp}-roc.png")
    plot_similarity(lfw_scores[lfw_labels == 1], lfw_scores[lfw_labels == 0], thr, f"{exp} — similaridade no LFW", fig / f"{exp}-lfw-similaridade.png")

    print(f"{exp}: threshold {thr:.3f} (escolhido em CelebA-val)")
    print(f"  CelebA-val: EER {val['eer']:.2%} | TAR@FAR{far_target:.0%} {val['tar_at_far']:.2%}")
    print(f"  LFW: acurácia 10 folds {lfw['lfw_10fold_accuracy']:.2%} ± {lfw['lfw_10fold_std']:.2%} | "
          f"EER {lfw['eer']:.2%} | com o threshold da validação: FAR {lfw['far']:.2%}, FRR {lfw['frr']:.2%}")


if __name__ == "__main__":
    main()
