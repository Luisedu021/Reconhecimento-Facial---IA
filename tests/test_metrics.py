import numpy as np
import pytest

from facerec import metrics


def separable():
    scores = np.array([0.9, 0.8, 0.85, 0.1, 0.2, 0.05])
    labels = np.array([1, 1, 1, 0, 0, 0])
    return scores, labels


def test_rates_at_threshold_conta_aceites_e_rejeicoes():
    scores = np.array([0.9, 0.4, 0.6, 0.1])
    labels = np.array([1, 1, 0, 0])
    r = metrics.rates_at_threshold(scores, labels, 0.5)
    assert r["far"] == 0.5  # 1 de 2 pares diferentes aceito
    assert r["frr"] == 0.5  # 1 de 2 pares iguais rejeitado
    assert r["accuracy"] == 0.5


def test_dados_separaveis_tem_eer_zero_e_auc_um():
    scores, labels = separable()
    assert metrics.eer(scores, labels)[0] == 0.0
    report = metrics.verification_report(scores, labels, threshold=0.5, far_target=0.01)
    assert report["roc_auc"] == 1.0
    assert report["tar_at_far"] == 1.0


def test_threshold_at_far_respeita_o_alvo():
    rng = np.random.default_rng(0)
    scores = np.concatenate([rng.normal(0.6, 0.15, 2000), rng.normal(0.2, 0.15, 2000)])
    labels = np.concatenate([np.ones(2000), np.zeros(2000)])
    thr = metrics.threshold_at_far(scores, labels, 0.01)
    assert metrics.rates_at_threshold(scores, labels, thr)["far"] <= 0.01


def test_kfold_escolhe_threshold_fora_do_fold():
    scores, labels = separable()
    folds = np.array([0, 1, 0, 1, 0, 1])
    mean, std = metrics.kfold_accuracy(scores, labels, folds)
    assert mean == 1.0 and std == 0.0


def test_sem_negativos_falha():
    with pytest.raises(ValueError):
        metrics.eer([0.5, 0.6], [1, 1])
