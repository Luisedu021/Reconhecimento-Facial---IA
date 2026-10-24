import torch
from torch.nn import functional as F

from facerec.losses import ArcFaceHead, semi_hard_triplet_loss


def test_arcface_sem_margem_vira_softmax_com_escala():
    torch.manual_seed(0)
    head = ArcFaceHead(8, 5, s=10.0, m=0.0)
    emb, y = torch.randn(16, 8), torch.randint(0, 5, (16,))
    cos = F.linear(F.normalize(emb), F.normalize(head.weight))
    assert torch.allclose(head(emb, y), F.cross_entropy(10.0 * cos, y), atol=1e-5)


def test_arcface_margem_aumenta_a_perda():
    torch.manual_seed(0)
    emb, y = torch.randn(16, 8), torch.randint(0, 5, (16,))
    a, b = ArcFaceHead(8, 5, m=0.0), ArcFaceHead(8, 5, m=0.5)
    b.weight.data = a.weight.data.clone()
    assert b(emb, y) > a(emb, y)


def clusters(spread: float):
    torch.manual_seed(0)
    centers = F.normalize(torch.eye(4, 16), dim=1)
    labels = torch.arange(4).repeat_interleave(4)
    return centers[labels] + spread * torch.randn(16, 16), labels


def test_triplet_zero_com_grupos_bem_separados():
    emb, labels = clusters(spread=0.01)
    assert semi_hard_triplet_loss(emb, labels, margin=0.2).item() == 0.0


def test_triplet_positiva_com_grupos_misturados():
    emb, labels = clusters(spread=1.0)
    assert semi_hard_triplet_loss(emb, labels, margin=0.2).item() > 0.0


def test_triplet_sem_negativos_nao_explode():
    loss = semi_hard_triplet_loss(torch.randn(4, 8), torch.zeros(4, dtype=torch.long))
    assert torch.isfinite(loss) and loss.item() == 0.0
