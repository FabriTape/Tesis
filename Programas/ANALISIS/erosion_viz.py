"""Módulo de visualizaciones de erosión.

Genera gráficos de boxplot y heatmap para análisis de erosión por posición.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cluster_calc import compute_erosion_counts_with_zeros
from constants import COLORS, I_VALUE
from kinematics_calc import compute_heatmap_data, get_grid_pairs
from matplotlib.figure import Figure
from scipy.stats import t


def plot_erosion_boxplot(df: pd.DataFrame, ff: float) -> Figure:
    """Genera gráfico de cajas y bigotes de IDs únicos por K/m
    e histogramas de distribución (Yield) por cada velocidad con IC.

    Args:
        df: DataFrame con datos.
        ff: Factor de fricción.

    Returns:
        Figura matplotlib.
    """
    # 1. Obtenemos la grilla global
    grid_pairs = get_grid_pairs(df)

    # 2. Usamos la función que INCLUYE los ceros
    data = compute_erosion_counts_with_zeros(df, ff, grid_pairs)

    if not data:
        return Figure()

    fs = 1.8 * 12
    unique_vel0 = sorted(data.keys())
    n_vels = len(unique_vel0)

    km_values = [(v**2) / 2 for v in unique_vel0]
    box_data = [data[v] for v in unique_vel0]

    # Configurar GridSpec: 2 filas.
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(2, n_vels, height_ratios=[1.5, 1])

    # --- TOP: GRÁFICO BOXPLOT ---
    ax_box = fig.add_subplot(gs[0, :])

    bp = ax_box.boxplot(
        box_data,
        labels=[f"{km:.1f}" for km in km_values],
        patch_artist=True,
        medianprops={"color": COLORS["power_law"], "linewidth": 2},
        flierprops={
            "marker": "o",
            "markerfacecolor": COLORS["primary"],
            "markersize": 6,
            "alpha": 0.5,
        },
    )

    for patch in bp["boxes"]:
        patch.set_facecolor(COLORS["primary"])
        patch.set_alpha(0.3)

    ax_box.set_xlabel(r"$K$/$m$ [J/kg]", fontsize=fs)
    ax_box.set_ylabel("Y (Yield)", fontsize=fs)
    ax_box.set_title(
        f"Distribución de erosión general por K/m (ff={ff}, i={I_VALUE})",
        fontsize=fs * 0.8,
    )
    ax_box.grid(True, alpha=0.3, axis="y")
    ax_box.tick_params(labelsize=fs * 0.7)

    # --- BOTTOM: HISTOGRAMAS POR VELOCIDAD ---
    for i, vel in enumerate(unique_vel0):
        ax_hist = fig.add_subplot(gs[1, i])
        y_vals = np.array(data[vel])

        # Histograma con KDE para suavizar la curva
        sns.histplot(y_vals, color=COLORS["secondary"], ax=ax_hist, alpha=0.4)

        # Calcular Estadísticas (Media e Intervalo de Confianza 95%)
        n = len(y_vals)
        mu = np.mean(y_vals)

        if n > 1:
            se = np.std(y_vals, ddof=1) / np.sqrt(n)
            ic = t.ppf(0.975, n - 1) * se
        else:
            ic = 0

        # Dibujar Media y Límites del IC
        ax_hist.axvline(
            mu, color=COLORS["power_law"], linestyle="-", linewidth=2, label="Media"
        )

        if ic > 0:
            ax_hist.axvline(
                mu - ic,
                color=COLORS["power_law"],
                linestyle="--",
                linewidth=1.5,
                label="IC MSE",
            )
            ax_hist.axvline(
                mu + ic, color=COLORS["power_law"], linestyle="--", linewidth=1.5
            )

        # Formato de los subplots individuales
        ax_hist.set_title(f"vel0 = {vel}", fontsize=fs * 0.6)
        ax_hist.set_xlabel("Y", fontsize=fs * 0.6)

        if i == 0:
            ax_hist.set_ylabel("Frecuencia", fontsize=fs * 0.6)
            ax_hist.legend(fontsize=fs * 0.45)
        else:
            ax_hist.set_ylabel("")

        ax_hist.tick_params(labelsize=fs * 0.5)

    fig.tight_layout()
    return fig


def plot_erosion_heatmap(df: pd.DataFrame, ff: float, vel0: float) -> Figure:
    data = compute_heatmap_data(df, ff, vel0)
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
    ax.set_title(
        f"Erosión por posición (ff={ff}, K/m={km:.1f} J/kg)", fontsize=fs * 0.8
    )
    ax.tick_params(labelsize=fs * 0.7)
    plt.tight_layout()
    return fig


def plot_erosion_vs_energia(
    df: pd.DataFrame,
    ff: float,
    vel0: float,
    metodo: str = "MAD",
) -> Figure | None:
    df = df[df["Método"] == metodo].copy()
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 6))
    df["errorbar"] = df["v"].apply(lambda x: float(x.split("+/-")[1]))
    df["v"] = df["v"].apply(lambda x: float(x.split("+/-")[0]))
    X = df["vel0"].apply(lambda x: 0.5 * x**2)
    ax.errorbar(
        X,
        df["v"],
        yerr=df["errorbar"],
        fmt="o",
        capsize=3,
        label=f"ff={ff}, vel0={vel0}, {metodo}",
    )
    ax.set_xlabel("Energía de impacto (J/kg)", fontsize=12)
    ax.set_ylabel("Erosión (Y)", fontsize=12)
    ax.set_title(f"Erosión vs Energía ({metodo})", fontsize=14)
    ax.tick_params(labelsize=10)
    ax.grid(visible=True)
    ax.grid(which="minor", linestyle=":")
    ax.minorticks_on()
    ax.legend()
    plt.tight_layout()
    return fig


def plot_erosion_vs_ff(
    df: pd.DataFrame, vel0: float, metodo: str = "MAD"
) -> Figure | None:
    df = df[df["Método"] == metodo].copy()
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 6))
    df["errorbar"] = df["v"].apply(lambda x: float(x.split("+/-")[1]))
    df["v"] = df["v"].apply(lambda x: float(x.split("+/-")[0]))
    grouped = (
        df.groupby("ff")
        .agg(v=("v", "mean"), errorbar=("errorbar", "mean"))
        .reset_index()
    )
    ax.errorbar(
        grouped["ff"], grouped["v"], yerr=grouped["errorbar"], fmt="o-", capsize=3
    )
    ax.set_xlabel("Fracción de fase (ff)", fontsize=12)
    ax.set_ylabel("Erosión (Y)", fontsize=12)
    ax.set_title(f"Erosión vs Fracción de fase (vel0={vel0}, {metodo})", fontsize=14)
    ax.grid(visible=True)
    ax.grid(which="minor", linestyle=":")
    ax.minorticks_on()
    ax.tick_params(labelsize=10)
    plt.tight_layout()
    return fig


def plot_erosion_vs_np(
    df: pd.DataFrame, ff: float, metodo: str = "MAD"
) -> Figure | None:
    df = df[df["Método"] == metodo].copy()
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 6))
    df["errorbar"] = df["v"].apply(lambda x: float(x.split("+/-")[1]))
    df["v"] = df["v"].apply(lambda x: float(x.split("+/-")[0]))
    grouped = (
        df.groupby("np")
        .agg(v=("v", "mean"), errorbar=("errorbar", "mean"))
        .reset_index()
    )
    ax.errorbar(
        grouped["np"], grouped["v"], yerr=grouped["errorbar"], fmt="o-", capsize=3
    )
    ax.set_xlabel("Número de proyectiles", fontsize=12)
    ax.set_ylabel("Erosión (Y)", fontsize=12)
    ax.set_title(f"Erosión vs Número de proyectiles (ff={ff}, {metodo})", fontsize=14)
    ax.grid(visible=True)
    ax.grid(which="minor", linestyle=":")
    ax.minorticks_on()
    ax.tick_params(labelsize=10)
    plt.tight_layout()
    return fig


def plot_erosion_vs_angulo(
    df: pd.DataFrame, ff: float, metodo: str = "MAD"
) -> Figure | None:
    df = df[df["Método"] == metodo].copy()
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 6))
    df["errorbar"] = df["v"].apply(lambda x: float(x.split("+/-")[1]))
    df["v"] = df["v"].apply(lambda x: float(x.split("+/-")[0]))
    grouped = (
        df.groupby("angulo")
        .agg(v=("v", "mean"), errorbar=("errorbar", "mean"))
        .reset_index()
    )
    ax.errorbar(
        grouped["angulo"], grouped["v"], yerr=grouped["errorbar"], fmt="o-", capsize=3
    )
    ax.set_xlabel("Ángulo (°)", fontsize=12)
    ax.set_ylabel("Erosión (Y)", fontsize=12)
    ax.set_title(f"Erosión vs Ángulo (ff={ff}, {metodo})", fontsize=14)
    ax.grid(visible=True)
    ax.grid(which="minor", linestyle=":")
    ax.minorticks_on()
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    return fig
