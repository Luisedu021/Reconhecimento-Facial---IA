"""Métricas de verificação e escolha de threshold (item 31 e 32).

Convenção: `scores` são similaridades de cosseno (maior = mais parecido) e
`labels` vale 1 para pares da mesma pessoa e 0 para pessoas diferentes.
Uma predição "mesma pessoa" acontece quando score >= threshold.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def _as_arrays(scores, labels) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if scores.shape != labels.shape:
        raise ValueError("scores e labels precisam ter o mesmo tamanho")
    if labels.min() == labels.max():
        raise ValueError("é preciso ter pares positivos e negativos")
    return scores, labels


def rates_at_threshold(scores, labels, threshold: float) -> dict[str, float]:
    """FAR, FRR, accuracy, precision, recall, F1 e a matriz de confusão (contagens de pares) para um threshold fixo."""
    scores, labels = _as_arrays(scores, labels)
    pred = scores >= threshold
    pos, neg = labels == 1, labels == 0
    tp = np.sum(pred & pos)
    fp = np.sum(pred & neg)
    fn = np.sum(~pred & pos)
    tn = np.sum(~pred & neg)
    far = fp / neg.sum()  # pessoas diferentes aceitas como a mesma
    frr = fn / pos.sum()  # mesma pessoa rejeitada
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / pos.sum()
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = np.mean(pred == pos)
    return {
        "far": float(far), "frr": float(frr), "accuracy": float(accuracy),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        # Matriz de confusão da verificação: "positivo" = par da mesma pessoa.
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def roc(scores, labels) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Curva ROC: fpr (= FAR), tpr (= 1 - FRR) e os thresholds correspondentes."""
    scores, labels = _as_arrays(scores, labels)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    return fpr, tpr, thresholds


def eer(scores, labels) -> tuple[float, float]:
    """Equal Error Rate: ponto em que FAR = FRR. Retorna (eer, threshold)."""
    fpr, tpr, thresholds = roc(scores, labels)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2), float(thresholds[i])


def threshold_at_far(scores, labels, far_target: float) -> float:
    """Menor threshold cuja FAR não passa de `far_target` (o que aceita mais pessoas certas)."""
    fpr, _, thresholds = roc(scores, labels)
    ok = np.where(fpr <= far_target)[0]
    # roc_curve ordena os thresholds do maior para o menor; o último índice válido é o mais permissivo.
    thr = thresholds[ok[-1]]
    # O primeiro threshold do sklearn é "infinito" (ninguém aceito); nesse caso usa o maior score + epsilon.
    return float(min(thr, np.max(scores) + 1e-6))


def tar_at_far(scores, labels, far_target: float) -> float:
    """TAR (taxa de aceitação correta) com FAR <= far_target."""
    fpr, tpr, _ = roc(scores, labels)
    return float(np.max(tpr[fpr <= far_target]))


def best_accuracy_threshold(scores, labels) -> float:
    """Threshold que maximiza a accuracy (usado no protocolo de 10 folds do LFW)."""
    scores, labels = _as_arrays(scores, labels)
    candidates = np.unique(scores)
    # Pontos médios entre scores vizinhos evitam empates exatamente no threshold.
    mids = np.concatenate([[candidates[0] - 1e-6], (candidates[:-1] + candidates[1:]) / 2, [candidates[-1] + 1e-6]])
    accs = [np.mean((scores >= t) == (labels == 1)) for t in mids]
    return float(mids[int(np.argmax(accs))])


def kfold_accuracy(scores, labels, folds) -> tuple[float, float]:
    """Protocolo oficial do LFW (View 2): para cada fold, o threshold é escolhido nos outros 9."""
    scores, labels = _as_arrays(scores, labels)
    folds = np.asarray(folds)
    accs = []
    for k in np.unique(folds):
        train, test = folds != k, folds == k
        thr = best_accuracy_threshold(scores[train], labels[train])
        accs.append(np.mean((scores[test] >= thr) == (labels[test] == 1)))
    return float(np.mean(accs)), float(np.std(accs))


def verification_report(scores, labels, threshold: float, far_target: float) -> dict[str, float]:
    """Todas as métricas de verificação de uma vez."""
    scores, labels = _as_arrays(scores, labels)
    report = rates_at_threshold(scores, labels, threshold)
    report["roc_auc"] = float(roc_auc_score(labels, scores))
    report["eer"], _ = eer(scores, labels)
    report["tar_at_far"] = tar_at_far(scores, labels, far_target)
    report["threshold"] = float(threshold)
    return report
