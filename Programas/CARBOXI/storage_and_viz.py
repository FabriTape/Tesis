"""
Almacenamiento de datos y visualización de resultados.

Este módulo combina:
- Lectura de archivos dump de LAMMPS
- Escritura de resultados (Parquet, marcadores)
- Generación de gráficos SVG

Física: Los archivos dump contienen posiciones y velocidades de partículas.
Los resultados almacenan el estado cinemático y topológico de las partículas
eyectadas de la cama de sílice.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

_ALL_COLUMNS: list[str] = ["id", "type", "x", "y", "z", "vx", "vy", "vz", "fz"]
_WAVE_COLUMNS: list[str] = ["id", "type", "x", "y", "z", "vx", "vy", "vz"]

_STEP_TO_US = 1.25
_AXIS_FONTSIZE = 14


def read_dump(path: Path, columns: list[str] | None = None) -> pl.LazyFrame:
    """Lee un archivo dump comprimido de LAMMPS como LazyFrame de Polars.

    Args:
        path: Ruta al archivo ``.gz`` a leer.
        columns: Lista de columnas a seleccionar.

    Returns:
        LazyFrame con las columnas solicitadas.
    """
    lf = pl.scan_csv(
        path,
        skip_rows=9,
        has_header=False,
        separator=" ",
        truncate_ragged_lines=True,
        new_columns=_ALL_COLUMNS,
        infer_schema_length=10_000,
        schema_overrides={"type": pl.UInt8},
    )
    return lf.select(columns or _ALL_COLUMNS)


def list_dump_files(directory: Path) -> list[Path]:
    """Lista y ordena cronológicamente los archivos dump de una simulación."""
    files = list(directory.glob("output.*.gz"))
    files.sort(key=lambda p: _extract_timestep(p.name))
    return files


def get_timestep_interval(files: list[Path]) -> int:
    """Extrae el intervalo de volcado desde la lista de archivos."""
    if len(files) < 2:
        return 0
    return _extract_timestep(files[1].name)


def _extract_timestep(filename: str) -> int:
    """Extrae el número de timestep del nombre de un archivo dump."""
    match = re.search(r"output\.(\d+)\.gz", filename)
    return int(match.group(1)) if match else 0


def save_erosion_results(
    results: list[pl.DataFrame],
    sim_id: str,
    timestamp: str,
    output_dir: Path,
    filter_thresholds: dict[str, float] | None = None,
    suffix: str = "",
    filter_name: str = "F3C",
) -> Path:
    """Consolida y guarda los resultados de erosión en formato Parquet."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pl.concat(results, how="vertical_relaxed")

    if filter_thresholds:
        df = df.with_columns(**{k: pl.lit(v) for k, v in filter_thresholds.items()})

    path = output_dir / f"{sim_id}_TOTAL_{filter_name}{suffix}_{timestamp}.parquet"
    df.write_parquet(path)
    return path


def save_no_erosion_flag(
    sim_id: str, timestamp: str, output_dir: Path, filter_name: str = "F3C"
) -> Path:
    """Crea un archivo de texto indicando que la simulación no produjo erosión."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{sim_id}_NADA_{filter_name}_{timestamp}.txt"
    path.write_text(
        "No se obtuvo erosión significativa en esta simulación.\n"
        f"Simulación: {sim_id}\n"
        f"Filtro: {filter_name}\n"
        f"Timestamp: {timestamp}\n"
    )
    return path


def save_wave_data(
    wave_frame: pl.LazyFrame,
    sim_id: str,
    step_index: int,
    timestamp: str,
    output_dir: Path,
    filter_name: str = "F3C",
) -> Path:
    """Guarda las partículas detectadas como parte de la onda de choque."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{sim_id}_ONDA_{filter_name}_{step_index}_{timestamp}.parquet"
    wave_frame.sink_parquet(path)
    return path


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
        total_timesteps: Número total de timesteps (len(results)-1).
            Si no se especifica, usa el máximo índice en df.
    """
    dir_imagenes.mkdir(parents=True, exist_ok=True)

    df_plot = _aggregate_ejected(df, total_timesteps)
    df_clusters = _aggregate_clusters(df, total_timesteps)
    df_gaps_cum = _compute_gaps_per_step(df, total_timesteps)
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


def _get_all_timesteps(df: pl.DataFrame, total_timesteps: int | None) -> pl.DataFrame:
    """Genera lista de todos los timesteps posibles.
    
    Args:
        df: DataFrame con datos.
        total_timesteps: Total de timesteps. Si None, usa el máximo en df.
    """
    if total_timesteps is not None:
        return pl.DataFrame({"i": range(total_timesteps + 1)})
    return df.select("i").unique().sort("i")


def _aggregate_ejected(df: pl.DataFrame, total_timesteps: int | None = None) -> pl.DataFrame:
    """Agrupa por iteración y cuenta partículas eyectadas.
    
    Rellena con 0 los timesteps sin eyectadas.
    """
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


def _aggregate_clusters(df: pl.DataFrame, total_timesteps: int | None = None) -> pl.DataFrame:
    """Agrupa por iteración y cuenta clusters únicos.
    
    Rellena con 0 los timesteps sin clusters (solo proyectil).
    """
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


def _compute_gaps_per_step(df: pl.DataFrame, total_timesteps: int | None = None) -> pl.DataFrame:
    """Calcula gaps acumulados por timestep.
    
    Rellena con 0 los timesteps sin gaps para mantener continuidad.
    """
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
    gaps_per_step = (
        gaps.group_by("i")
        .agg(pl.len().alias("gaps_cum"))
        .sort("i")
    )
    
    all_steps = _get_all_timesteps(df, total_timesteps)
    
    return all_steps.join(gaps_per_step, on="i", how="left").fill_null(0).sort("i").with_columns(
        pl.col("gaps_cum").cast(pl.Int64).cum_sum().alias("gaps_cum")
    )


def _compute_gaps_per_step_raw(df: pl.DataFrame, total_timesteps: int | None = None) -> pl.DataFrame:
    """Calcula gaps por step individual.
    
    Rellena con 0 los timesteps sin gaps.
    """
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
    
    cols = df.columns
    get_first = lambda c: df[c][0] if c in cols else "N/A"
    return {
        "ff": get_first("ff"),
        "np": get_first("np"),
        "vel0": get_first("vel0"),
    }


def _draw_ejected(ax, df: pl.DataFrame, meta: dict, step_to_us: float) -> None:
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


def _draw_clusters(ax, df: pl.DataFrame, meta: dict, step_to_us: float) -> None:
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


def _plot_empty(
    sim_id: str,
    timestamp: str,
    dir_imagenes: Path,
    filter_name: str = "F3C",
    step_to_us: float = _STEP_TO_US,
) -> None:
    """Genera un gráfico vacío cuando no hay erosión."""
    dir_imagenes.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.subplots_adjust(top=0.90, right=0.82, wspace=0.28)
    fig.suptitle(
        f"Erosión — {filter_name} — {sim_id.replace('_', ' ')} (sin datos)",
        fontsize=14,
    )

    for ax in axes.flat:
        ax.set_xlabel(r"tiempo [$\mu$s]", fontsize=_AXIS_FONTSIZE)
        ax.set_ylabel("0", fontsize=_AXIS_FONTSIZE)
        ax.tick_params(labelsize=_AXIS_FONTSIZE)
        ax.grid(alpha=0.3)
        ax.text(
            0.5, 0.5, "Sin erosión", transform=ax.transAxes,
            fontsize=12, ha="center", va="center",
            color="gray", style="italic",
        )

    axes[0, 0].set_ylabel("Y", fontsize=_AXIS_FONTSIZE)
    axes[0, 1].set_ylabel("N clusters", fontsize=_AXIS_FONTSIZE)
    axes[1, 0].set_ylabel("Gaps acumulados", fontsize=_AXIS_FONTSIZE)
    axes[1, 1].set_ylabel("Gaps por step", fontsize=_AXIS_FONTSIZE)

    output_path = dir_imagenes / f"plot_{sim_id}_{filter_name}_{timestamp}.svg"
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close()

    print(f"  Gráfico guardado en: {output_path}")
