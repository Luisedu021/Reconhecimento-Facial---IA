"""Funções de perda de metric learning (item 28): ArcFace e Triplet com mineração semi-hard."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class ArcFaceHead(nn.Module):
    """ArcFace (Deng et al., 2019): margem angular aditiva.

    Cada identidade de treino tem um vetor W_j. Para um embedding x da identidade y,
    o ângulo θ_y entre x e W_y é aumentado em m antes da softmax: o logit vira s·cos(θ_y + m).
    Para acertar mesmo com essa penalidade, a rede precisa deixar cada pessoa mais
    concentrada e mais longe das outras — exatamente o que melhora a comparação por cosseno.
    """

    def __init__(self, embedding_size: int, num_classes: int, s: float = 64.0, m: float = 0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        self.s, self.m = s, m
        self.cos_m, self.sin_m = math.cos(m), math.sin(m)
        # Quando θ + m passaria de π, cos(θ + m) deixa de ser decrescente; usa-se uma penalidade linear.
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Em float32: com AMP, cos/sqrt em meia precisão perdem exatidão perto de ±1.
        cos = F.linear(F.normalize(emb.float()), F.normalize(self.weight.float())).clamp(-1 + 1e-7, 1 - 1e-7)
        sin = torch.sqrt(1.0 - cos**2)
        cos_theta_m = cos * self.cos_m - sin * self.sin_m  # cos(θ + m)
        cos_theta_m = torch.where(cos > self.th, cos_theta_m, cos - self.mm)
        one_hot = F.one_hot(labels, num_classes=cos.shape[1]).bool()
        logits = torch.where(one_hot, cos_theta_m, cos) * self.s
        return F.cross_entropy(logits, labels)


def semi_hard_triplet_loss(emb: torch.Tensor, labels: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    """Triplet loss (Schroff et al., 2015) com mineração semi-hard dentro do batch.

    Para cada par âncora-positivo (a, p), escolhe o negativo n mais próximo que ainda esteja
    MAIS LONGE que o positivo: d(a,p) < d(a,n). São os "semi-difíceis": informativos sem
    serem tão difíceis a ponto de travar o treino no começo. Se não houver nenhum, usa o
    negativo mais distante. Perda: max(d(a,p) - d(a,n) + margin, 0), com d = distância euclidiana ao quadrado.
    """
    emb = F.normalize(emb.float())
    dist = torch.cdist(emb, emb).pow(2)  # (B, B)
    same = labels[:, None] == labels[None, :]
    eye = torch.eye(len(labels), dtype=torch.bool, device=emb.device)
    neg_mask = ~same  # pares (a, n)
    # Âncoras sem nenhum negativo no batch não formam triplet.
    pos_mask = same & ~eye & neg_mask.any(dim=1, keepdim=True)  # pares (a, p)

    d_ap = dist[:, :, None]  # (a, p, 1)
    d_an = dist[:, None, :]  # (a, 1, n)
    valid_neg = neg_mask[:, None, :]  # (a, 1, n)

    outside = valid_neg & (d_an > d_ap)  # negativos mais longe que o positivo
    inf = torch.tensor(float("inf"), device=emb.device)
    semi_hard = torch.where(outside, d_an.expand_as(outside), inf).min(dim=2).values  # (a, p)
    farthest = torch.where(valid_neg, d_an, -inf).max(dim=2).values.expand_as(semi_hard)  # (a, p)
    chosen = torch.where(torch.isinf(semi_hard), farthest, semi_hard)

    losses = F.relu(dist - chosen + margin)[pos_mask]
    return losses.mean() if losses.numel() else emb.sum() * 0.0
