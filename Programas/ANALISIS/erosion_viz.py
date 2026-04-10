"""Módulo de visualizaciones de erosión.

Genera gráficos de boxplot y heatmap para análisis de erosión por posición.
"""

import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from cluster_calc import compute_unique_ids_raw
from kinematics_calc import compute_heatmap_data
from constants import I_VALUE, COLORS


def plot_erosion_boxplot(df: pd.DataFrame, ff: float) -> Figure:
    """Genera gráfico de cajas y bigotes de IDs únicos por K/m.

    Muestra la distribución de partículas erosionadas a través de todos
    los movimientos para cada K/m.

    Args:
        df: DataFrame con datos.
        ff: Factor de fricción.

    Returns:
        Figura matplotlib.
    """
    data = compute_unique_ids_raw(df, ff)
    if not data:
        return Figure()

    fs = 1.8 * 12
    unique_vel0 = sorted(data.keys())

    km_values = [(v**2) / 2 for v in unique_vel0]
    box_data = [data[v] for v in unique_vel0]

    fig, ax = plt.subplots(figsize=(10, 6))

    bp = ax.boxplot(
        box_data,
        labels=[f"{km:.1f}" for km in km_values],
        patch_artist=True,
        medianprops={"color": COLORS["power_law"], "linewidth": 2},
        flierprops={"marker": "o", "markerfacecolor": COLORS["primary"],
                    "markersize": 6, "alpha": 0.5},
    )

    for patch in bp["boxes"]:
        patch.set_facecolor(COLORS["primary"])
        patch.set_alpha(0.3)

    ax.set_xlabel(r"$K$/$m$ [J/kg]", fontsize=fs)
    ax.set_ylabel("Y promedio", fontsize=fs)
    ax.set_title(f"Distribución de erosión por K/m (ff={ff}, i={I_VALUE})", fontsize=fs * 0.8)
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(labelsize=fs * 0.7)
    plt.tight_layout()
    return fig


def plot_erosion_heatmap(
    df: pd.DataFrame, ff: float, vel0: float, grid_pairs: list[tuple[float, float]] | None = None
) -> Figure:
    """Genera heatmap de partículas por posición.

    Args:
        df: DataFrame con datos.
        ff: Factor de fricción.
        vel0: Velocidad inicial.
        grid_pairs: Lista de pares (posx, posy) válidos.

    Returns:
        Figura matplotlib.
    """
    data = compute_heatmap_data(df, ff, vel0, grid_pairs)
    if data.empty:
        return Figure()

    fs = 1.8 * 12

    fig, ax = plt.subplots(figsize=(8, 6))

    data_sorted = data.sort_index(ascending=False)

    sns.heatmap(
        data_sorted,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        cbar_kws={"label": "Y", "shrink": 0.8},
        ax=ax,
        linewidths=0.5,
        linecolor="white",
        square=True,
    )

    km = (vel0**2) / 2
    ax.set_xlabel("posx (σ)", fontsize=fs)
    ax.set_ylabel("posy (σ)", fontsize=fs)
    ax.set_title(f"Erosión por posición (ff={ff}, K/m={km:.1f} J/kg)",
                 fontsize=fs * 0.8)
    ax.tick_params(labelsize=fs * 0.7)
    plt.tight_layout()
    return fig