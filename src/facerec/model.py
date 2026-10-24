"""Backbone InceptionResnetV1 (FaceNet) e estratégias de fine-tuning (SPEC-001: E01-E03)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import ImageListDataset

# Camadas finais da rede. No fine-tuning parcial só elas treinam.
# avgpool e dropout não têm pesos, mas ficam aqui para seguirem o modo de treino do bloco.
TOP_MODULES = ("block8", "avgpool_1a", "dropout", "last_linear", "last_bn")
TOP_WITH_WEIGHTS = ("block8", "last_linear", "last_bn")


def build_backbone(pretrained: str = "vggface2") -> nn.Module:
    """InceptionResnetV1 pré-treinado. Sem `classify`, a saída já é o embedding de 512 dimensões normalizado (L2)."""
    from facenet_pytorch import InceptionResnetV1

    model = InceptionResnetV1(pretrained=pretrained, classify=False)
    # A camada de classificação original (8631 pessoas do VGGFace2) não é usada.
    if hasattr(model, "logits"):
        for p in model.logits.parameters():
            p.requires_grad = False
    return model


def apply_strategy(model: nn.Module, strategy: str) -> None:
    """Define quais parâmetros treinam: none (E01), partial (E02) ou full (E03)."""
    if strategy not in {"none", "partial", "full"}:
        raise ValueError(f"estratégia desconhecida: {strategy}")
    for name, module in model.named_children():
        if name == "logits":
            continue
        trainable = strategy == "full" or (strategy == "partial" and name in TOP_WITH_WEIGHTS)
        for p in module.parameters():
            p.requires_grad = trainable


def set_train_mode(model: nn.Module, strategy: str) -> None:
    """model.train() só nas partes que treinam.

    Detalhe importante: camadas congeladas ficam em eval(). Senão o BatchNorm atualizaria
    as médias e variâncias com os dados novos, mudando a rede mesmo sem gradiente.
    """
    model.train()
    if strategy == "full":
        return
    for name, module in model.named_children():
        if strategy == "none" or name not in TOP_MODULES:
            module.eval()


def param_groups(model: nn.Module, strategy: str, lr: float, lr_backbone: float) -> list[dict]:
    """Learning rate maior nas camadas finais e menor no restante do backbone (E03)."""
    top, rest = [], []
    for name, module in model.named_children():
        params = [p for p in module.parameters() if p.requires_grad]
        (top if name in TOP_WITH_WEIGHTS else rest).extend(params)
    groups = [{"params": top, "lr": lr}]
    if strategy == "full" and rest:
        groups.append({"params": rest, "lr": lr_backbone})
    return groups


def count_trainable(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def load_checkpoint(model: nn.Module, path: Path) -> None:
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"])


@torch.no_grad()
def embed_paths(model: nn.Module, paths: list[Path], device: torch.device, batch_size: int = 128) -> torch.Tensor:
    """Embeddings normalizados de uma lista de recortes, na ordem dada. Ficam só em memória."""
    model.eval()
    loader = DataLoader(ImageListDataset(paths, train=False), batch_size=batch_size, shuffle=False, num_workers=0)
    chunks = []
    for x, _ in loader:
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            chunks.append(model(x.to(device)).float())
    emb = torch.cat(chunks)
    return nn.functional.normalize(emb, dim=1)


def cosine_scores(emb: torch.Tensor, index: dict, pairs) -> np.ndarray:
    """Similaridade de cosseno de cada par (a, b, ...), usando `index` imagem -> linha de `emb`."""
    ia = torch.tensor([index[p[0]] for p in pairs])
    ib = torch.tensor([index[p[1]] for p in pairs])
    return (emb[ia] * emb[ib]).sum(dim=1).cpu().numpy()
