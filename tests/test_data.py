"""CA-1 a CA-4: divisões sem sobreposição, sem augmentation fora do treino e remoção de quase-duplicatas."""

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms

from facerec.data import (
    PKSampler,
    Standardize,
    assert_disjoint,
    build_transform,
    make_pairs,
    near_duplicate_mask,
    read_lfw_pairs,
    select_identity_subset,
    split_by_identity,
)


def fake_celeba(n_ids=50, per_id=25):
    return {f"{i * 100 + j:06d}.jpg": i for i in range(n_ids) for j in range(per_id)}


def test_split_sem_sobreposicao():
    subset = select_identity_subset(fake_celeba(), num_identities=40, min_images=20, max_images=20, seed=42)
    train_ids, val_ids = split_by_identity(list(subset), val_fraction=0.1, seed=42)
    assert not set(train_ids) & set(val_ids)  # CA-1
    rows = [(img, i, "train" if i in train_ids else "val") for i, imgs in subset.items() for img in imgs]
    assert_disjoint(rows)  # CA-1 e CA-2
    assert len(val_ids) == 4 and all(len(v) == 20 for v in subset.values())


def test_split_contaminado_e_detectado():
    with pytest.raises(AssertionError):
        assert_disjoint([("a.png", 1, "train"), ("b.png", 1, "val")])  # mesma identidade nos dois
    with pytest.raises(AssertionError):
        assert_disjoint([("a.png", 1, "train"), ("a.png", 2, "val")])  # mesma imagem nos dois


def test_split_e_reproduzivel():
    ids = list(range(100))
    assert split_by_identity(ids, 0.1, seed=7) == split_by_identity(ids, 0.1, seed=7)


def test_sem_augmentation_fora_do_treino():
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (160, 160, 3), dtype=np.uint8))
    t = build_transform(train=False)
    assert all(isinstance(step, Standardize) for step in t.transforms)  # CA-3
    assert torch.equal(t(img), t(img))
    kinds = {type(step) for step in build_transform(train=True).transforms}
    assert transforms.RandomHorizontalFlip in kinds and transforms.ColorJitter in kinds


def test_remove_sobreposicao_celeba_lfw():
    g = torch.Generator().manual_seed(0)
    lfw = torch.nn.functional.normalize(torch.randn(200, 512, generator=g), dim=1)
    celeba = torch.nn.functional.normalize(torch.randn(300, 512, generator=g), dim=1)
    planted = [5, 77, 250]
    for k, i in enumerate(planted):  # quase-cópias de imagens do LFW
        celeba[i] = torch.nn.functional.normalize(lfw[k] + 0.01 * torch.randn(512, generator=g), dim=0)
    mask = near_duplicate_mask(celeba, lfw, threshold=0.9, chunk=64)
    assert sorted(np.where(mask)[0].tolist()) == planted  # CA-4


def test_make_pairs_balanceado_e_sem_repeticao():
    groups = {i: [f"{i}_{j}.png" for j in range(10)] for i in range(20)}
    pairs = make_pairs(groups, n_per_class=100, seed=1)
    assert sum(p[2] for p in pairs) == 100 and len(pairs) == 200
    assert len({(a, b) for a, b, _ in pairs}) == 200
    for a, b, y in pairs:
        assert (a.split("_")[0] == b.split("_")[0]) == bool(y)


def test_read_lfw_pairs(tmp_path):
    f = tmp_path / "pairs.txt"
    f.write_text("2\t1\nAna\t1\t2\nAna\t1\tBeto\t3\nCaio\t2\t4\nCaio\t1\tDani\t1\n", encoding="utf-8")
    pairs = read_lfw_pairs(f)
    assert pairs[0] == ("Ana/Ana_0001.jpg", "Ana/Ana_0002.jpg", 1, 0)
    assert pairs[1] == ("Ana/Ana_0001.jpg", "Beto/Beto_0003.jpg", 0, 0)
    assert [p[3] for p in pairs] == [0, 0, 1, 1]


def test_pk_sampler_tem_p_pessoas_com_k_fotos():
    labels = [i // 6 for i in range(60)]  # 10 pessoas x 6 fotos
    sampler = PKSampler(labels, p=4, k=3, seed=0)
    for batch in sampler:
        people = [labels[i] for i in batch]
        assert len(set(people)) == 4 and all(people.count(p) == 3 for p in set(people))
