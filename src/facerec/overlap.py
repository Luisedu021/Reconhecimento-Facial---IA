"""Checagem de quase-duplicatas entre o subconjunto do CelebA e o LFW (CA-4).

Uso (depois de preprocess lfw e celeba):
    python -m facerec.overlap --config configs/base.yaml

Embeda as duas bases com o modelo pré-treinado e lista as imagens do CelebA cujo cosseno
com alguma imagem do LFW passa do limite. Essas imagens ficam fora do treino e da validação,
para que nenhuma foto do teste (LFW) tenha sido vista antes.
"""

from __future__ import annotations

import argparse

import torch

from .config import load_config, resolve, set_seed
from .data import near_duplicate_mask, read_split
from .model import build_backbone, embed_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = resolve(cfg["paths"]["public_cache"])

    celeba = [img for img, _, _ in read_split(cache / "celeba" / "split.csv")]
    lfw = sorted((cache / "lfw").glob("*/*.png"))
    model = build_backbone(cfg["model"]["pretrained"]).to(device)
    bs = cfg["eval"]["batch_size"]
    emb_celeba = embed_paths(model, [cache / "celeba" / i for i in celeba], device, bs)
    emb_lfw = embed_paths(model, lfw, device, bs)

    hits = near_duplicate_mask(emb_celeba, emb_lfw, cfg["celeba"]["overlap_threshold"])
    flagged = [img for img, hit in zip(celeba, hits) if hit]
    out = cache / "celeba" / "overlap_lfw.txt"
    out.write_text("\n".join(flagged), encoding="utf-8")
    print(f"{len(flagged)} de {len(celeba)} imagens do CelebA com cosseno > {cfg['celeba']['overlap_threshold']} "
          f"em relação ao LFW; listadas em {out.relative_to(resolve('.'))} e excluídas do treino")


if __name__ == "__main__":
    main()
