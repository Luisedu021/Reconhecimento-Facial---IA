"""CA-7 e CA-8: mesmo YAML e mesma seed dão o mesmo resultado; cada linha registra seed, versões e hardware."""

import csv

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from facerec.config import load_config, set_seed
from facerec.losses import ArcFaceHead
from facerec.results import append_result
from facerec.train import train_one_epoch


def tiny_run(seed: int) -> list[float]:
    set_seed(seed)
    x = torch.randn(64, 12)
    y = torch.randint(0, 4, (64,))
    model = nn.Sequential(nn.Linear(12, 8))
    head = ArcFaceHead(8, 4, s=10.0, m=0.3)
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=1e-2)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True, generator=torch.Generator().manual_seed(seed))
    return [train_one_epoch(model, head, loader, opt, scaler, torch.device("cpu")) for _ in range(3)]


def test_mesma_seed_mesmo_resultado():
    assert tiny_run(42) == tiny_run(42)  # CA-7


def test_resultado_registra_reprodutibilidade(tmp_path):
    path = tmp_path / "results.csv"
    row = {"experiment": "E01", "dataset": "lfw", "eer": 0.05, "seed": 42,
           "versions": {"torch": "2.2.2"}, "hardware": "GPU X", "config": {"seed": 42}}
    append_result(path, row)
    with open(path, encoding="utf-8") as f:
        saved = next(csv.DictReader(f))
    for key in ("seed", "versions", "hardware", "config"):  # CA-8
        assert saved[key]
    assert saved["eer"] == "0.050000"


def test_resultado_sem_seed_e_recusado(tmp_path):
    with pytest.raises(ValueError):
        append_result(tmp_path / "r.csv", {"experiment": "E01", "dataset": "lfw", "versions": {"a": 1},
                                           "hardware": "x", "config": {"a": 1}})


def test_configs_herdam_do_base():
    e03 = load_config("configs/e03.yaml")
    assert e03["experiment"] == "E03" and e03["seed"] == 42
    assert e03["train"]["strategy"] == "full" and e03["train"]["loss"] == "arcface"
    assert e03["train"]["lr"] == 1.0e-4  # herdado do e02
    e05 = load_config("configs/e05.yaml")
    assert e05["train"]["loss"] == "triplet" and e05["train"]["strategy"] == "partial"
