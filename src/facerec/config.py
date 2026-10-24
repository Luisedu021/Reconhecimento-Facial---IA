"""Leitura das configurações YAML e utilidades de reprodutibilidade (RF-3, RNF-1)."""

from __future__ import annotations

import copy
import os
import platform
import random
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Raiz do repositório: src/facerec/config.py -> ../../
ROOT = Path(__file__).resolve().parents[2]


def _deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | os.PathLike) -> dict[str, Any]:
    """Carrega um YAML de experimento.

    Se o arquivo tiver a chave `base`, o YAML base é carregado primeiro e o do
    experimento sobrescreve só o que declarar. Assim cada experimento mostra
    apenas o que muda em relação aos demais.
    """
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base_name = cfg.pop("base", None)
    if base_name:
        cfg = _deep_merge(load_config(path.parent / base_name), cfg)
    return cfg


def resolve(path: str | os.PathLike) -> Path:
    """Caminhos da config são relativos à raiz do repositório."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def set_seed(seed: int) -> None:
    """Fixa todas as fontes de aleatoriedade usadas no projeto."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Convoluções determinísticas: um pouco mais lentas, mas o mesmo resultado a cada execução.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def environment_info() -> dict[str, Any]:
    """Versões e hardware, registrados junto de cada resultado (item 34)."""
    import torch

    versions = {"python": platform.python_version()}
    for pkg in ("torch", "torchvision", "facenet-pytorch", "numpy", "scikit-learn"):
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = "ausente"
    versions["cuda"] = torch.version.cuda or "cpu"
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "sem GPU"
    hardware = f"{gpu} | {platform.processor() or platform.machine()} | {platform.system()} {platform.release()}"
    return {"versions": versions, "hardware": hardware}
