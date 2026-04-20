"""Visualizaciones básicas para análisis de erosión.

Genera gráficos de boxplot y heatmap para análisis de erosión por posición.
Usa Polars para manejo de datos y Matplotlib/Seaborn para visualizaciones.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from scipy.stats import t

# Colores para gráficos
_COLORS = {
    "primary": "#3b82f6",
    "power_law": "#e74c3c",
    "truncated": "#2ecc71",
    "secondary": "#3498db",
}

# Valor I por defecto
I_VALUE = 80


def plot_erosion_boxplot(
    df: pl.DataFrame,
    ff: float,
    output_dir: Path | None = None,
) -> Figure:
    """Genera gráfico de cajas y bigotes de IDs únicos por K/m
    e histogramas de distribución (Yield) por cada velocidad con IC.

    Args:
        df: DataFrame con columnas: vel0, posx, posy, id.
        ff: Fraction fill.
        output_dir: Directorio de salida (si None, solo retorna figura).

    Returns:
        Figura matplotlib.
    """
    from matplotlib import pyplot as plt
    from stats.clusters import compute_erosion_counts_with_zeros
    from stats.kinematics import get_grid_pairs

    grid_pairs = get_grid_pairs(df)
    data = compute_erosion_counts_with_zeros(df, ff, grid_pairs)

    if not data:
        return Figure()

    fs = 1.8 * 12
    unique_vel0 = sorted(data.keys())
    n_vels = len(unique_vel0)

    km_values = [(v**2) / 2 for v in unique_vel0]
    box_data = [data[v] for v in unique_vel0]

    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(2, n_vels, height_ratios=[1.5, 1])

    ax_box = fig.add_subplot(gs[0, :])

    bp = ax_box.boxplot(
        box_data,
        labels=[f"{km:.1f}" for km in km_values],
        patch_artist=True,
        medianprops={"color": _COLORS["power_law"], "linewidth": 2},
        flierprops={
            "marker": "o",
            "markerfacecolor": _COLORS["primary"],
            "markersize": 6,
            "alpha": 0.5,
        },
    )

    for patch in bp["boxes"]:
        patch.set_facecolor(_COLORS["primary"])
        patch.set_alpha(0.3)

    ax_box.set_xlabel(r"$K$/$m$ [J/kg]", fontsize=fs)
    ax_box.set_ylabel("Y (Yield)", fontsize=fs)
    ax_box.set_title(
        f"Distribución de erosión general por K/m (ff={ff}, i={I_VALUE})",
        fontsize=fs * 0.8,
    )
    ax_box.grid(True, alpha=0.3, axis="y")
    ax_box.tick_params(labelsize=fs * 0.7)

    for i, vel in enumerate(unique_vel0):
        ax_hist = fig.add_subplot(gs[1, i])
        y_vals = np.array(data[vel])

        sns.histplot(y_vals, color=_COLORS["secondary"], ax=ax_hist, alpha=0.4)

        n = len(y_vals)
        mu = np.mean(y_vals)

        if n > 1:
            se = np.std(y_vals, ddof=1) / np.sqrt(n)
            ic = t.ppf(0.975, n - 1) * se
        else:
            ic = 0

        ax_hist.axvline(
            mu, color=_COLORS["power_law"], linestyle="-", linewidth=2
        )

        if ic > 0:
            ax_hist.axvline(
                mu - ic,
                color=_COLORS["power_law"],
                linestyle="--",
                linewidth=1.5,
            )
            ax_hist.axvline(
                mu + ic, color=_COLORS["power_law"], linestyle="--", linewidth=1.5
            )

        ax_hist.set_title(f"vel0 = {vel}", fontsize=fs * 0.6)
        ax_hist.set_xlabel("Y", fontsize=fs * 0.6)

        if i == 0:
            ax_hist.set_ylabel("Frecuencia", fontsize=fs * 0.6)
        else:
            ax_hist.set_ylabel("")

        ax_hist.tick_params(labelsize=fs * 0.5)

    fig.tight_layout()

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"boxplot_ff{ff}.png"
        fig.savefig(output_path, dpi=250, bbox_inches="tight")
        print(f"  Gráfico guardado en: {output_path}")

    return fig


def plot_erosion_heatmap(
    df: pl.DataFrame,
    ff: float,
    vel0: float,
    output_dir: Path | None = None,
) -> Figure:
    """Genera heatmap de erosión por posición.

    Args:
        df: DataFrame con columnas: posx, posy, type, z, vz.
        ff: Fraction fill.
        vel0: Velocidad inicial.
        output_dir: Directorio de salida.

    Returns:
        Figura matplotlib.
    """
    from matplotlib import pyplot as plt
    from stats.kinematics import compute_heatmap_data

    data = compute_heatmap_data(df, ff, vel0)
    if data.is_empty():
        return Figure()

    fs = 1.8 * 12

    fig, ax = plt.subplots(figsize=(8, 6))

    data_sorted = data.sort(by="posy", descending=True)

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

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"heatmap_ff{ff}_vel{vel0}.png"
        fig.savefig(output_path, dpi=250, bbox_inches="tight")
        print(f"  Gráfico guardado en: {output_path}")

    return fig


def plot_erosion_vs_energia(
    df: pl.DataFrame,
    ff: float,
    vel0: float,
    metodo: str = "MAD",
    output_dir: Path | None = None,
) -> Figure | None:
    """Genera gráfico de erosión vs energía de impacto.

    Args:
        df: DataFrame con columnas: vel0, metodo, v, errorbar.
        ff: Fraction fill.
        vel0: Velocidad inicial.
        metodo: Método de filtrado.
        output_dir: Directorio de salida.

    Returns:
        Figura matplotlib o None.
    """
    from matplotlib import pyplot as plt

    df_filtered = df.filter(pl.col("metodo") == metodo)
    if df_filtered.is_empty():
        return None

    fig, ax = plt.subplots(figsize=(8, 6))
    X = df_filtered["vel0"].map(lambda x: 0.5 * x**2)

    ax.errorbar(
        X,
        df_filtered["v"],
        yerr=df_filtered["errorbar"],
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

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"vs_energia_ff{ff}_{metodo}.png"
        fig.savefig(output_path, dpi=250, bbox_inches="tight")

    return fig


def plot_erosion_vs_ff(
    df: pl.DataFrame,
    vel0: float,
    metodo: str = "MAD",
    output_dir: Path | None = None,
) -> Figure | None:
    """Genera gráfico de erosión vs fracción de fase.

    Args:
        df: DataFrame con columnas: ff, metodo, v, errorbar.
        vel0: Velocidad inicial.
        metodo: Método de filtrado.
        output_dir: Directorio de salida.

    Returns:
        Figura matplotlib o None.
    """
    from matplotlib import pyplot as plt

    df_filtered = df.filter(pl.col("metodo") == metodo)
    if df_filtered.is_empty():
        return None

    fig, ax = plt.subplots(figsize=(8, 6))

    grouped = (
        df_filtered.group_by("ff")
        .agg(
            v=pl.col("v").mean(),
            errorbar=pl.col("errorbar").mean(),
        )
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

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"vs_ff_vel{vel0}_{metodo}.png"
        fig.savefig(output_path, dpi=250, bbox_inches="tight")

    return fig


def plot_erosion_vs_np(
    df: pl.DataFrame,
    ff: float,
    metodo: str = "MAD",
    output_dir: Path | None = None,
) -> Figure | None:
    """Genera gráfico de erosión vs número de proyectiles.

    Args:
        df: DataFrame con columnas: np, metodo, v, errorbar.
        ff: Fraction fill.
        metodo: Método de filtrado.
        output_dir: Directorio de salida.

    Returns:
        Figura matplotlib o None.
    """
    from matplotlib import pyplot as plt

    df_filtered = df.filter(pl.col("metodo") == metodo)
    if df_filtered.is_empty():
        return None

    fig, ax = plt.subplots(figsize=(8, 6))

    grouped = (
        df_filtered.group_by("np")
        .agg(
            v=pl.col("v").mean(),
            errorbar=pl.col("errorbar").mean(),
        )
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

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"vs_np_ff{ff}_{metodo}.png"
        fig.savefig(output_path, dpi=250, bbox_inches="tight")

    return fig


def plot_erosion_vs_angulo(
    df: pl.DataFrame,
    ff: float,
    metodo: str = "MAD",
    output_dir: Path | None = None,
) -> Figure | None:
    """Genera gráfico de erosión vs ángulo de impacto.

    Args:
        df: DataFrame con columnas: angulo, metodo, v, errorbar.
        ff: Fraction fill.
        metodo: Método de filtrado.
        output_dir: Directorio de salida.

    Returns:
        Figura matplotlib o None.
    """
    from matplotlib import pyplot as plt

    df_filtered = df.filter(pl.col("metodo") == metodo)
    if df_filtered.is_empty():
        return None

    fig, ax = plt.subplots(figsize=(8, 6))

    grouped = (
        df_filtered.group_by("angulo")
        .agg(
            v=pl.col("v").mean(),
            errorbar=pl.col("errorbar").mean(),
        )
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

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"vs_angulo_ff{ff}_{metodo}.png"
        fig.savefig(output_path, dpi=250, bbox_inches="tight")

    return fig


def save_all_figures_pdf(figures: list[Figure], output_path: Path) -> None:
    """Guarda una lista de figuras en un archivo PDF.

    Args:
        figures: Lista de figuras matplotlib.
        output_path: Ruta del archivo PDF de salida.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_path) as pdf:
        for fig in figures:
            if fig.get_axes():
                pdf.savefig(fig)
                import matplotlib.pyplot as plt

                plt.close(fig)