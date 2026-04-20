"""Visualizaciones de trayectorias de erosión.

Genera gráficos de partículas eyectadas vs tiempo, clusters, gaps acumulados,
y gaps por step.
Usa Polars para manejo de datos.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

_STEP_TO_US: float = 1.25
_AXIS_FONTSIZE: int = 14


def plot_erosion(
    df: pl.DataFrame,
    sim_id: str,
    timestamp: str,
    dir_imagenes: Path,
    filter_name: str = "F3C",
    step_to_us: float = _STEP_TO_US,
    total_timesteps: int | None = None,
) -> None:
    """Genera y guarda el gráfico comparativo de erosión con 4 subplots.

    Produce una figura de 2x2 con:
    (a) Partículas eyectadas vs tiempo
    (b) Número de clusters vs tiempo
    (c) Gaps acumulados vs tiempo
    (d) Gaps por step vs tiempo

    Args:
        df: DataFrame con datos de erosión.
        sim_id: Identificador de la simulación.
        timestamp: Marca temporal.
        dir_imagenes: Directorio de destino.
        filter_name: Nombre del filtro aplicado.
        step_to_us: Factor de conversión timestep a microsegundos.
        total_timesteps: Número total de timesteps.
    """
    from matplotlib import pyplot as plt

    dir_imagenes.mkdir(parents=True, exist_ok=True)

    df_plot = _aggregate_ejected(df, total_timesteps)
    df_clusters = _aggregate_clusters(df, total_timesteps)
    df_gaps_cum = _compute_gaps_cumulative(df, total_timesteps)
    df_gaps_raw = _compute_gaps_per_step_raw(df, total_timesteps)
    meta = _extract_plot_metadata(df)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.subplots_adjust(top=0.90, right=0.82, wspace=0.28)
    fig.suptitle(
        f"Erosión — {filter_name} — {sim_id.replace('_', ' ')}",
        fontsize=14,
    )

    _draw_ejected(axes[0, 0], df_plot, meta, step_to_us)
    _draw_clusters(axes[0, 1], df_clusters, meta, step_to_us)
    _draw_gaps_cumulative(axes[1, 0], df_gaps_cum, step_to_us)
    _draw_gaps_per_step(axes[1, 1], df_gaps_raw, step_to_us)

    output_path = dir_imagenes / f"plot_{sim_id}_{filter_name}_{timestamp}.svg"
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close()

    print(f"  Gráfico guardado en: {output_path}")


def _get_all_timesteps(
    df: pl.DataFrame, total_timesteps: int | None
) -> pl.DataFrame:
    """Genera lista de todos los timesteps posibles."""
    if total_timesteps is not None:
        return pl.DataFrame({"i": range(total_timesteps + 1)})
    return df.select("i").unique().sort("i")


def _aggregate_ejected(
    df: pl.DataFrame, total_timesteps: int | None = None
) -> pl.DataFrame:
    """Agrupa por iteración y cuenta partículas eyectadas."""
    if df.is_empty():
        all_steps = _get_all_timesteps(df, total_timesteps)
        return all_steps.with_columns(pl.lit(0).alias("cantidad_eyectados"))

    ejected = (
        df.group_by("i")
        .agg(pl.len().alias("cantidad_eyectados"))
        .sort("i")
    )

    all_steps = _get_all_timesteps(df, total_timesteps)
    return all_steps.join(ejected, on="i", how="left").fill_null(0).sort("i")


def _aggregate_clusters(
    df: pl.DataFrame, total_timesteps: int | None = None
) -> pl.DataFrame:
    """Agrupa por iteración y cuenta clusters únicos."""
    if df.is_empty():
        all_steps = _get_all_timesteps(df, total_timesteps)
        return all_steps.with_columns(pl.lit(0).alias("n_clusters"))

    clusters = (
        df.group_by("i")
        .agg(pl.col("cluster").n_unique().alias("n_clusters"))
        .sort("i")
    )

    all_steps = _get_all_timesteps(df, total_timesteps)
    return all_steps.join(clusters, on="i", how="left").fill_null(0).sort("i")


def _compute_gaps_cumulative(
    df: pl.DataFrame, total_timesteps: int | None = None
) -> pl.DataFrame:
    """Calcula gaps acumulados por timestep."""
    if df.is_empty():
        all_steps = _get_all_timesteps(df, total_timesteps)
        return all_steps.with_columns(pl.lit(0).alias("gaps_cum"))

    df_sorted = df.sort(["id", "i"])
    df_with_lag = df_sorted.with_columns(
        pl.col("i").shift(1).over("id").alias("i_prev"),
        pl.col("z").shift(1).over("id").alias("z_prev"),
    )
    gaps = df_with_lag.filter(
        pl.col("i_prev").is_not_null() & (pl.col("i") - pl.col("i_prev") > 1)
    )
    gaps_per_step = gaps.group_by("i").agg(pl.len().alias("gaps")).sort("i")

    all_steps = _get_all_timesteps(df, total_timesteps)

    return (
        all_steps.join(gaps_per_step, on="i", how="left")
        .fill_null(0)
        .sort("i")
        .with_columns(pl.col("gaps").cast(pl.Int64).cum_sum().alias("gaps_cum"))
    )


def _compute_gaps_per_step_raw(
    df: pl.DataFrame, total_timesteps: int | None = None
) -> pl.DataFrame:
    """Calcula gaps por step individual."""
    if df.is_empty():
        all_steps = _get_all_timesteps(df, total_timesteps)
        return all_steps.with_columns(pl.lit(0).alias("gaps"))

    df_sorted = df.sort(["id", "i"])
    df_with_lag = df_sorted.with_columns(
        pl.col("i").shift(1).over("id").alias("i_prev"),
    )
    gaps = df_with_lag.filter(
        pl.col("i_prev").is_not_null() & (pl.col("i") - pl.col("i_prev") > 1)
    )
    gaps_per_step = gaps.group_by("i").agg(pl.len().alias("gaps")).sort("i")

    all_steps = _get_all_timesteps(df, total_timesteps)
    return all_steps.join(gaps_per_step, on="i", how="left").fill_null(0).sort("i")


def _extract_plot_metadata(df: pl.DataFrame) -> dict:
    """Extrae metadatos del DataFrame para usar en los gráficos."""
    if df.is_empty():
        return {"ff": "N/A", "np": "N/A", "vel0": "N/A"}

    def _get_first(col: str) -> str:
        return df[col][0] if col in df.columns else "N/A"

    return {
        "ff": _get_first("ff"),
        "np": _get_first("np"),
        "vel0": _get_first("vel0"),
    }


def _draw_ejected(
    ax, df: pl.DataFrame, meta: dict, step_to_us: float
) -> None:
    """Dibuja el subplot de partículas eyectadas."""
    if not df.is_empty():
        ax.plot(df["i"] * step_to_us, df["cantidad_eyectados"], marker="o", markersize=3)
    ax.set_xlabel(r"tiempo [$\mu$s]", fontsize=_AXIS_FONTSIZE)
    ax.set_ylabel("Y", fontsize=_AXIS_FONTSIZE)
    ax.tick_params(labelsize=_AXIS_FONTSIZE)
    ax.text(
        0.02, 0.98, "(a)", transform=ax.transAxes,
        fontsize=14, fontweight="bold", va="top", ha="left",
    )
    ax.grid(alpha=0.3)


def _draw_clusters(
    ax, df: pl.DataFrame, meta: dict, step_to_us: float
) -> None:
    """Dibuja el subplot de clusters."""
    if not df.is_empty():
        ax.plot(df["i"] * step_to_us, df["n_clusters"], marker="s", markersize=3)
    ax.set_xlabel(r"tiempo [$\mu$s]", fontsize=_AXIS_FONTSIZE)
    ax.set_ylabel("N clusters", fontsize=_AXIS_FONTSIZE)
    ax.tick_params(labelsize=_AXIS_FONTSIZE)
    ax.text(
        0.02, 0.98, "(b)", transform=ax.transAxes,
        fontsize=14, fontweight="bold", va="top", ha="left",
    )
    ax.grid(alpha=0.3)


def _draw_gaps_cumulative(ax, df: pl.DataFrame, step_to_us: float) -> None:
    """Dibuja el subplot de gaps acumulados."""
    if not df.is_empty():
        ax.plot(df["i"] * step_to_us, df["gaps_cum"], marker="^", markersize=3)
    ax.set_xlabel(r"tiempo [$\mu$s]", fontsize=_AXIS_FONTSIZE)
    ax.set_ylabel("Gaps acumulados", fontsize=_AXIS_FONTSIZE)
    ax.tick_params(labelsize=_AXIS_FONTSIZE)
    ax.text(
        0.02, 0.98, "(c)", transform=ax.transAxes,
        fontsize=14, fontweight="bold", va="top", ha="left",
    )
    ax.grid(alpha=0.3)


def _draw_gaps_per_step(ax, df: pl.DataFrame, step_to_us: float) -> None:
    """Dibuja el subplot de gaps por step."""
    if not df.is_empty():
        ax.plot(df["i"] * step_to_us, df["gaps"], marker="v", markersize=3)
    ax.set_xlabel(r"tiempo [$\mu$s]", fontsize=_AXIS_FONTSIZE)
    ax.set_ylabel("Gaps por step", fontsize=_AXIS_FONTSIZE)
    ax.tick_params(labelsize=_AXIS_FONTSIZE)
    ax.text(
        0.02, 0.98, "(d)", transform=ax.transAxes,
        fontsize=14, fontweight="bold", va="top", ha="left",
    )
    ax.grid(alpha=0.3)