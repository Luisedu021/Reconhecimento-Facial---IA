"""CA-5 e CA-6: o threshold vem de fora e a saída privada é só agregada. Dados 100% sintéticos."""

import inspect

import pytest
import torch
from torch.nn import functional as F

from facerec import evaluate_private as ep


def synthetic_people(n=5, noise=0.02):
    g = torch.Generator().manual_seed(0)
    centers = F.normalize(torch.randn(n, 512, generator=g), dim=1)
    people = {}
    for i in range(n):
        def shots(k):
            return F.normalize(centers[i] + noise * torch.randn(k, 512, generator=g), dim=1)
        people[f"P{i + 1:02d}"] = {"S1": shots(2), "S2": shots(3)}
    unknown = F.normalize(torch.randn(50, 512, generator=g), dim=1)
    return people, unknown


@pytest.mark.parametrize("method", ["prototype", "knn1"])
def test_saida_agregada(method):
    people, unknown = synthetic_people()
    result, genuine, impostor = ep.evaluate_private(people, unknown, threshold=0.5, method=method)
    assert set(result) <= ep.AGGREGATE_KEYS  # CA-6: nenhuma chave com ID, nome ou caminho
    assert all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in result.values())
    assert "P01" not in repr(result)
    # Pessoas sintéticas bem separadas: tudo certo.
    assert result["rank1_accuracy"] == 1.0
    assert result["unknown_reject_rate"] == 1.0 and result["unknown_reject_rate_lfw"] == 1.0
    assert result["n_probes"] == 15
    assert len(genuine) == 15 and len(impostor) == 60


def test_saida_com_dado_individual_e_recusada():
    with pytest.raises(ValueError):
        ep.check_aggregate_only({"rank1_accuracy": 1.0, "pessoa": 1.0})
    with pytest.raises(TypeError):
        ep.check_aggregate_only({"rank1_accuracy": "P01"})
    with pytest.raises(TypeError):
        ep.check_aggregate_only({"eer": [0.1, 0.2]})


def test_threshold_nao_e_escolhido_com_dados_privados():
    source = inspect.getsource(ep)  # CA-5
    for forbidden in ("threshold_at_far", "choose_threshold", "best_accuracy_threshold"):
        assert forbidden not in source


def test_rodizio_rejeita_quem_saiu_do_cadastro():
    people, unknown = synthetic_people()
    # Threshold impossível de atingir: ninguém é aceito, todo desconhecido é rejeitado.
    result, _, _ = ep.evaluate_private(people, unknown, threshold=1.1, method="prototype")
    assert result["unknown_reject_rate"] == 1.0 and result["frr"] == 1.0 and result["correct_accept_rate"] == 0.0
