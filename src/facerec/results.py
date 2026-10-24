"""Registro dos resultados em reports/results.csv (RF-4, CA-8)."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

FIELDS = [
    "timestamp",
    "experiment",
    "dataset",
    # verificação (CelebA-val, LFW)
    "accuracy",
    "f1",
    "roc_auc",
    "eer",
    "tar_at_far",
    "far",
    "frr",
    "threshold",
    "lfw_10fold_accuracy",
    "lfw_10fold_std",
    # identificação (dataset privado, sempre agregado)
    "rank1_accuracy",
    "correct_accept_rate",
    "misidentification_rate",
    "unknown_reject_rate",
    "unknown_reject_rate_lfw",
    "n_probes",
    # custo
    "train_time_s",
    "ms_per_image",
    "trainable_params",
    # reprodutibilidade
    "seed",
    "versions",
    "hardware",
    "config",
]

REQUIRED = ("experiment", "dataset", "seed", "versions", "hardware", "config")


def append_result(path: Path, row: dict) -> None:
    """Acrescenta uma linha. Recusa linhas sem os campos de reprodutibilidade (CA-8)."""
    missing = [k for k in REQUIRED if row.get(k) in (None, "", {})]
    if missing:
        raise ValueError(f"resultado sem campos obrigatórios: {missing}")
    unknown = set(row) - set(FIELDS)
    if unknown:
        raise ValueError(f"campos fora do formato: {sorted(unknown)}")
    out = {k: row.get(k, "") for k in FIELDS}
    out["timestamp"] = out["timestamp"] or datetime.now().isoformat(timespec="seconds")
    for k in ("versions", "config"):
        if not isinstance(out[k], str):
            out[k] = json.dumps(out[k], ensure_ascii=False, sort_keys=True)
    for k, v in out.items():
        if isinstance(v, float):
            out[k] = f"{v:.6f}"
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(out)
