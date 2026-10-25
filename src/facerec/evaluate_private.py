"""Avaliação no dataset privado (E04) — saída SEMPRE agregada (RF-5, RF-6, CA-5, CA-6).

Uso (rodar localmente, pelos integrantes; nunca enviar a saída por pessoa a serviços externos):
    python -m facerec.preprocess --config configs/base.yaml --dataset private
    python -m facerec.evaluate_private --config configs/e04.yaml

- O threshold vem de fora (reports/thresholds.json, escolhido na validação do CelebA, ou --threshold).
  Este script nunca escolhe threshold com dados privados (CA-5).
- Embeddings das fotos privadas ficam só em memória: nada biométrico é gravado em disco.
- Cadastro = fotos da sessão S1; consulta = fotos da sessão S2 (outro dia).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import metrics
from .config import environment_info, load_config, resolve, set_seed
from .evaluate import base_row, load_experiment_model
from .model import embed_paths
from .plots import plot_similarity
from .results import append_result

# Únicas chaves que podem sair deste script (CA-6). Todas numéricas e agregadas.
AGGREGATE_KEYS = {
    "rank1_accuracy",
    "correct_accept_rate",
    "misidentification_rate",
    "frr",
    "unknown_reject_rate",
    "far",
    "unknown_reject_rate_lfw",
    "roc_auc",
    "eer",
    "n_probes",
    "threshold",
}

Gallery = dict[str, torch.Tensor]  # ID pseudônimo -> embeddings de cadastro (n, 512)


def best_match(gallery: Gallery, probe: torch.Tensor, method: str) -> tuple[str, float]:
    """Pessoa mais parecida e a similaridade.

    prototype: compara com a média dos embeddings de cadastro de cada pessoa.
    knn1: compara com cada foto de cadastro e fica com a mais parecida (1 vizinho mais próximo).
    """
    best_id, best_score = None, -2.0
    for pid, embs in gallery.items():
        if method == "prototype":
            score = float(F.normalize(embs.mean(dim=0), dim=0) @ probe)
        elif method == "knn1":
            score = float((embs @ probe).max())
        else:
            raise ValueError(f"método desconhecido: {method}")
        if score > best_score:
            best_id, best_score = pid, score
    return best_id, best_score


def identify(gallery: Gallery, probes: list[tuple[str, torch.Tensor]], threshold: float, method: str) -> dict[str, int]:
    """Contagens de identificação aberta: quem é cadastrado deve ser reconhecido; quem não é, rejeitado."""
    c = dict(known=0, rank1=0, correct_accept=0, misidentified=0, false_reject=0, unknown=0, unknown_rejected=0)
    for true_id, emb in probes:
        pred, score = best_match(gallery, emb, method)
        accepted = score >= threshold
        if true_id in gallery:
            c["known"] += 1
            c["rank1"] += pred == true_id
            c["correct_accept"] += pred == true_id and accepted
            c["misidentified"] += pred != true_id and accepted
            c["false_reject"] += not accepted
        else:
            c["unknown"] += 1
            c["unknown_rejected"] += not accepted
    return c


def evaluate_private(
    people: dict[str, dict[str, torch.Tensor]], lfw_unknown: torch.Tensor, threshold: float, method: str
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Calcula as métricas agregadas. `people` = {ID: {"S1": (n,512), "S2": (m,512)}}.

    Retorna (métricas agregadas, similaridades mesma pessoa, similaridades pessoas diferentes).
    """
    ids = sorted(people)
    full_gallery = {pid: people[pid]["S1"] for pid in ids}

    # 1. Todos cadastrados: as fotos S2 de cada um são consultadas.
    known_probes = [(pid, e) for pid in ids for e in people[pid]["S2"]]
    closed = identify(full_gallery, known_probes, threshold, method)

    # 2. Rodízio: cada pessoa fica uma vez de fora do cadastro e vira "desconhecida" (RF-6).
    open_counts = dict(unknown=0, unknown_rejected=0)
    for out in ids:
        gallery = {pid: full_gallery[pid] for pid in ids if pid != out}
        unknown = [(out, e) for s in ("S1", "S2") for e in people[out][s]]
        c = identify(gallery, unknown, threshold, method)
        open_counts["unknown"] += c["unknown"]
        open_counts["unknown_rejected"] += c["unknown_rejected"]

    # 3. Rostos do LFW como desconhecidos (pessoas públicas, nunca cadastradas).
    lfw = identify(full_gallery, [("lfw", e) for e in lfw_unknown], threshold, method)

    # Distribuições para o gráfico: cada consulta S2 contra o protótipo de cada pessoa.
    protos = torch.stack([F.normalize(full_gallery[pid].mean(dim=0), dim=0) for pid in ids])
    genuine, impostor = [], []
    for i, pid in enumerate(ids):
        sims = people[pid]["S2"] @ protos.T  # (m, n_pessoas)
        genuine += sims[:, i].tolist()
        impostor += torch.cat([sims[:, :i], sims[:, i + 1 :]], dim=1).flatten().tolist()
    genuine, impostor = np.array(genuine), np.array(impostor)
    scores = np.concatenate([genuine, impostor])
    labels = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])

    result = {
        "rank1_accuracy": closed["rank1"] / closed["known"],
        "correct_accept_rate": closed["correct_accept"] / closed["known"],
        "misidentification_rate": closed["misidentified"] / closed["known"],
        "frr": closed["false_reject"] / closed["known"],
        "unknown_reject_rate": open_counts["unknown_rejected"] / open_counts["unknown"],
        "far": 1 - open_counts["unknown_rejected"] / open_counts["unknown"],
        "unknown_reject_rate_lfw": lfw["unknown_rejected"] / max(lfw["unknown"], 1),
        "roc_auc": float(metrics.roc_auc_score(labels, scores)),
        "eer": metrics.eer(scores, labels)[0],
        "n_probes": closed["known"],
        "threshold": float(threshold),
    }
    check_aggregate_only(result)
    return result, genuine, impostor


def check_aggregate_only(result: dict) -> None:
    """Garante que a saída não carrega IDs, nomes, caminhos nem embeddings (CA-6)."""
    extra = set(result) - AGGREGATE_KEYS
    if extra:
        raise ValueError(f"chaves não permitidas na saída privada: {sorted(extra)}")
    for key, value in result.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{key} precisa ser um número agregado, não {type(value).__name__}")


def load_private_embeddings(cfg: dict, model, device) -> dict[str, dict[str, torch.Tensor]]:
    root = resolve(cfg["paths"]["private_cache"])
    people: dict[str, dict[str, torch.Tensor]] = {}
    for person in sorted(p for p in root.iterdir() if p.is_dir()):
        sessions = {s: sorted((person / s).glob("*.png")) for s in ("S1", "S2")}
        if not sessions["S1"] or not sessions["S2"]:
            continue  # sem cadastro ou sem consulta: fica fora da avaliação
        people[person.name] = {s: embed_paths(model, files, device) for s, files in sessions.items()}
    if len(people) < 2:
        raise SystemExit("são necessárias pelo menos 2 pessoas com fotos S1 e S2 em data/private/cache")
    return people


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--threshold", type=float, help="padrão: o da validação em reports/thresholds.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = load_config(cfg["source_config"])  # experimento vencedor na validação (ADR-001)
    reports = resolve(cfg["paths"]["reports"])

    threshold = args.threshold
    if threshold is None:
        thresholds = json.loads((reports / "thresholds.json").read_text(encoding="utf-8"))
        threshold = thresholds[source["experiment"]]["threshold"]

    model, _ = load_experiment_model(source, device)
    people = load_private_embeddings(cfg, model, device)
    lfw_files = sorted((resolve(cfg["paths"]["public_cache"]) / "lfw").glob("*/*.png"))
    lfw_sample = random.Random(cfg["seed"]).sample(lfw_files, min(cfg["eval"]["lfw_unknown_probes"], len(lfw_files)))
    lfw_unknown = embed_paths(model, lfw_sample, device)

    env = environment_info()
    for method in cfg["methods"]:
        result, genuine, impostor = evaluate_private(people, lfw_unknown, threshold, method)
        append_result(reports / "results.csv", {**base_row(cfg, env), "dataset": f"privado-{method}", **result})
        print(f"{cfg['experiment']} ({method}): " + json.dumps({k: round(v, 4) for k, v in result.items()}))
    plot_similarity(genuine, impostor, threshold, f"{cfg['experiment']} — similaridade no dataset privado (agregado)",
                    reports / "figures" / f"{cfg['experiment']}-privado-similaridade.png")
    print(f"{len(people)} pessoas avaliadas. Métricas agregadas gravadas em {Path('reports') / 'results.csv'}.")


if __name__ == "__main__":
    main()
