"""Pré-processamento em lote com dados sintéticos (sem MTCNN real)."""

import numpy as np
import torch
from PIL import Image

from facerec.preprocess import run_batched


class FakeDetector:
    """Imita o MTCNN: em lote quebra se alguma imagem não tem rosto (imagem preta)."""

    def __call__(self, imgs):
        if isinstance(imgs, list):
            if any(self._is_blank(i) for i in imgs):
                raise ValueError("inhomogeneous shape")
            return [self(i) for i in imgs]
        return None if self._is_blank(imgs) else torch.full((3, 8, 8), 128.0)

    @staticmethod
    def _is_blank(img):
        return np.asarray(img).max() == 0


def _write(path, value):
    Image.fromarray(np.full((16, 16, 3), value, dtype=np.uint8)).save(path)
    return path


def test_lote_com_imagem_sem_rosto_nao_quebra(tmp_path):
    sources = [_write(tmp_path / f"in{i}.png", v) for i, v in enumerate([200, 0, 200])]
    outputs = [tmp_path / "out" / f"out{i}.png" for i in range(3)]

    status = run_batched(FakeDetector(), sources, outputs, batch_size=3, fallback_size=None)

    assert status == ["ok", "sem_face", "ok"]
    assert outputs[0].exists() and not outputs[1].exists() and outputs[2].exists()
