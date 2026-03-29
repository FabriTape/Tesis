"""
Visualización de resultados para CARBOXI_MINI.

Genera gráficos SVG de la evolución temporal de clusters balísticos
(partículas en vuelo balístico libre) a partir de los DataFrames
producidos por el pipeline.

Física: Cada punto representa la cantidad de clusters con fuerza Z neta
cercana a cero en un paso temporal, indicando eyección libre de la cama.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl


def plot_erosion(
    df: pl.DataFrame,
    sim_id: str,
    timestamp: str,
    dir_imagenes: Path,
) -> None:
    """Genera y guarda gráfico de evolución de clusters balísticos.

    Agrupa por ``step``, cuenta clusters únicos y produce un gráfico
    de línea SVG con metadatos de simulación.

    Args:
        df: DataFrame con columnas ``"step"``, ``"Cluster_ID"``,
            ``"ff"``, ``"np"``, ``"vel0"``, ``"angulo"``.
        sim_id: Identificador de la simulación.
        timestamp: Marca temporal ``"YYYY-MM-DD_HH-MM-SS"``.
        dir_imagenes: Directorio de destino para el SVG.

    Example:
        >>> plot_erosion(df, "ff_0.15_...", "2025-07-01_12-00-00",
        ...              Path("./resultados/Imagenes"))
    """
    dir_imagenes.mkdir(parents=True, exist_ok=True)

    df_plot = _aggregate_clusters(df)
    meta = _extract_plot_metadata(df)

    _draw_figure(df_plot, meta)

    filename = f"plot_{sim_id}_{timestamp}.svg"
    output_path = dir_imagenes / filename
    plt.savefig(output_path, format="svg")
    plt.close()

    print(f"Gráfico guardado en: {output_path}")


def _aggregate_clusters(df: pl.DataFrame) -> pl.DataFrame:
    """Cuenta clusters balísticos únicos por paso temporal.

    Args:
        df: DataFrame con ``"step"`` y ``"Cluster_ID"``.

    Returns:
        DataFrame con ``"step"`` y ``"n_clusters"`` ordenado por step.
    """
    return (
        df.group_by("step")
        .agg(pl.col("Cluster_ID").n_unique().alias("n_clusters"))
        .sort("step")
    )


def _extract_plot_metadata(df: pl.DataFrame) -> dict[str, str]:
    """Extrae metadatos de simulación del primer registro.

    Args:
        df: DataFrame con columnas de metadatos.

    Returns:
        Dict con claves ``"ff"``, ``"np"``, ``"vel"``, ``"angle"``.
    """
    first = df.row(0, named=True)
    return {
        "ff": str(first.get("ff", "N/A")),
        "np": str(first.get("np", "N/A")),
        "vel": str(first.get("vel0", "N/A")),
        "angle": str(first.get("angulo", "N/A")),
    }


def _draw_figure(
    df_plot: pl.DataFrame, meta: dict[str, str]
) -> None:
    """Dibuja la figura matplotlib de clusters por step.

    Args:
        df_plot: DataFrame con ``"step"`` y ``"n_clusters"``.
        meta: Parámetros físicos para título y leyenda.
    """
    plt.figure(figsize=(10, 6))

    plt.plot(
        df_plot["step"].to_list(),
        df_plot["n_clusters"].to_list(),
        marker="o",
        linestyle="-",
        color="steelblue",
        linewidth=1.5,
        markersize=4,
    )

    titulo = (
        "Clusters balísticos por paso\n"
        f"v₀={meta['vel']} m/s  |  ff={meta['ff']}  |  "
        f"N={meta['np']}  |  θ={meta['angle']}°"
    )
    plt.title(titulo, fontsize=12)
    plt.xlabel("Step (timestep)", fontsize=11)
    plt.ylabel("Clusters balísticos", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)

    param_text = (
        f"ff = {meta['ff']}\n"
        f"N = {meta['np']}\n"
        f"v₀ = {meta['vel']} m/s\n"
        f"θ = {meta['angle']}°"
    )
    plt.text(
        0.98,
        0.02,
        param_text,
        transform=plt.gca().transAxes,
        fontsize=9,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    plt.tight_layout()
