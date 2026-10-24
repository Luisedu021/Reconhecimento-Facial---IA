"""Gráficos agregados para reports/figures/ (os únicos PNG permitidos no repositório).

Cores da paleta de referência validada (séries 1-3: azul, laranja, verde-água), superfície clara.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # gera arquivos sem abrir janela
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]

plt.rcParams.update(
    {
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "font.size": 10,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
        "legend.labelcolor": INK_2,
    }
)


def pct(v: float) -> str:
    """Percentual com vírgula decimal: 0.962 -> "96,2%"."""
    return f"{v:.1%}".replace(".", ",")


def _legend_above(ax, ncol: int) -> None:
    """Legenda numa linha acima da área do gráfico, para não cobrir os dados."""
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=ncol, borderaxespad=0.3, handlelength=1.6, fontsize=9)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_roc(curves: dict[str, tuple[np.ndarray, np.ndarray]], points: dict[str, tuple[float, float]], title: str, path: Path) -> None:
    """TAR × FAR com FAR em escala log: em reconhecimento facial o que importa é a região de FAR baixa.

    `points` marca o ponto de operação do threshold escolhido em cada curva: {rótulo: (far, tar)}.
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    lowest = 1.0
    for color, (label, (fpr, tpr)) in zip(SERIES, curves.items()):
        mask = fpr > 0
        lowest = min(lowest, tpr[mask & (fpr >= 1e-4)].min(initial=1.0))
        if label in points:
            far, tar = points[label]
            label = f"{label} — no threshold: FAR {pct(far)}, TAR {pct(tar)}"
            ax.plot([max(far, 1e-4)], [tar], "o", ms=7, color=color, mec=SURFACE, mew=1.5, zorder=3)
        ax.plot(fpr[mask], tpr[mask], color=color, lw=1.6, label=label)
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1)
    # Zoom na faixa de TAR que existe: com modelos bons, a curva inteira fica acima de 80%.
    ax.set_ylim(max(0.0, lowest - 0.05), 1.005)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("FAR — pessoas diferentes aceitas (escala log)")
    ax.set_ylabel("TAR — mesma pessoa aceita")
    ax.set_title(title, pad=34)
    _legend_above(ax, ncol=1)
    _save(fig, path)


def plot_similarity(genuine: np.ndarray, impostor: np.ndarray, threshold: float, title: str, path: Path) -> None:
    """Distribuição da similaridade: mesma pessoa × pessoas diferentes, com o threshold marcado (item 31)."""
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    bins = np.linspace(-0.4, 1.0, 57)
    for values, color, label in ((genuine, SERIES[0], "Mesma pessoa"), (impostor, SERIES[1], "Pessoas diferentes")):
        weights = np.full(len(values), 1 / max(len(values), 1))
        ax.hist(values, bins=bins, weights=weights, color=color, alpha=0.25, edgecolor="none")
        ax.hist(values, bins=bins, weights=weights, histtype="step", color=color, lw=1.6, label=f"{label} (n = {len(values)})")
    ax.axvline(threshold, color=INK_2, lw=1.2, ls="--", label=f"Threshold = {threshold:.3f}".replace(".", ","))
    ax.set_xlabel("Similaridade de cosseno")
    ax.set_ylabel("Proporção dos pares")
    ax.set_title(title, pad=24)
    _legend_above(ax, ncol=3)
    _save(fig, path)


def plot_comparison(labels: list[str], values: list[float], title: str, xlabel: str, path: Path) -> None:
    """Uma métrica entre experimentos (barras horizontais a partir do zero, valor no fim de cada barra).

    Com acurácias muito próximas (ex.: 96% x 97%), prefira plotar a taxa de ERRO ou o EER:
    as barras começam no zero, e a diferença relativa entre erros é o que aparece.
    """
    fig, ax = plt.subplots(figsize=(6.4, 0.55 * len(labels) + 1.2))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=SERIES[0], height=0.55)
    for yi, v in zip(y, values):
        ax.annotate(pct(v), (v, yi), xytext=(4, 0), textcoords="offset points", va="center", color=INK, fontsize=9)
    ax.set_yticks(y, labels)
    ax.tick_params(axis="y", colors=INK_2, length=0)
    ax.set_xlim(0, max(values) * 1.15 if values else 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    _save(fig, path)
