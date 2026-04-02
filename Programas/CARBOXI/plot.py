"""
Módulo de visualización para el análisis de simulaciones LAMMPS.

Genera gráficos SVG comparativos de la evolución temporal de partículas
eyectadas a partir de los DataFrames producidos por el pipeline de análisis.
El nombre del archivo de salida incluye los parámetros de la simulación,
el filtro activo y un timestamp para garantizar trazabilidad y unicidad.

Física: Cada subplot representa una métrica diferente de la erosión
acumulada a lo largo del tiempo de simulación.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

_STEP_TO_US = 1.25
_AXIS_FONTSIZE = 14


def plot_erosion(
    df: pl.DataFrame,
    sim_id: str,
    timestamp: str,
    dir_imagenes: Path,
    filter_name: str = "F3C",
    step_to_us: float = _STEP_TO_US,
) -> None:
    """Genera y guarda el gráfico comparativo de erosión con 4 subplots.

    Produce una figura de 2x2 con:
    (a) Partículas eyectadas vs tiempo
    (b) Número de clusters vs tiempo
    (c) Gaps acumulados vs tiempo
    (d) Gaps por step vs tiempo

    Args:
        df: DataFrame con las columnas ``"i"`` (índice de iteración),
            ``"proyectil_erosion"`` (bool o None), ``"cluster"`` (UInt32 o None),
            ``"ff"``, ``"np"``, ``"vel0"``, ``"angulo"``, ``"posx"``,
            ``"posy"``, ``"posz"``.
        sim_id: Identificador de la simulación (nombre de carpeta sin prefijo
            ``erosion_bed_SI_``).
        timestamp: Marca temporal en formato ``"YYYY-MM-DD_HH-MM-SS"``.
        dir_imagenes: Directorio de destino para el archivo SVG.
        filter_name: Nombre del filtro de erosión aplicado (F1I, F1C, F2I,
            F2C, F3I, F3C).
        step_to_us: Factor de conversión de timestep a microsegundos.
            Por defecto 1.25.

    Returns:
        None

    Raises:
        KeyError: Si el DataFrame no contiene las columnas requeridas.

    Example:
        >>> plot_erosion(df, "ff_0.15_np_1_vel_-50.0_...", "2025-07-01_12-00-00",
        ...              Path("./resultados/Imagenes"), filter_name="F3C")
    """
    dir_imagenes.mkdir(parents=True, exist_ok=True)

    df_plot = _aggregate_ejected(df)
    df_clusters = _aggregate_clusters(df)
    df_gaps_cum = _compute_gaps_per_step(df)
    df_gaps_raw = _compute_gaps_per_step_raw(df)
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


def _aggregate_ejected(df: pl.DataFrame) -> pl.DataFrame:
    """Agrupa por iteración y cuenta partículas eyectadas en cada timestep.

    Args:
        df: DataFrame completo con columnas ``"i"`` y ``"proyectil_erosion"``.

    Returns:
        DataFrame ordenado con columnas ``"i"`` y ``"cantidad_eyectados"``.
    """
    return (
        df.group_by("i")
        .agg(
            (pl.col("proyectil_erosion").is_null() | pl.col("proyectil_erosion"))
            .sum()
            .alias("cantidad_eyectados")
        )
        .sort("i")
    )


def _aggregate_clusters(df: pl.DataFrame) -> pl.DataFrame:
    """Agrupa por iteración y cuenta clusters únicos en cada timestep.

    Args:
        df: DataFrame completo con columnas ``"i"`` y ``"cluster"``.

    Returns:
        DataFrame ordenado con columnas ``"i"`` y ``"n_clusters"``.
    """
    return (
        df.filter(pl.col("cluster").is_not_null())
        .group_by("i")
        .agg(pl.col("cluster").n_unique().alias("n_clusters"))
        .sort("i")
    )


def _compute_gaps_per_step(df: pl.DataFrame) -> pl.DataFrame:
    """Calcula gaps acumulados por step usando tracking por ID.

    Para cada step, cuenta cuántas IDs únicas desaparecieron en algún
    momento anterior. La curva es estrictamente creciente o plana.

    Args:
        df: DataFrame concatenado de todos los steps con columnas ``"i"`` e ``"id"``.

    Returns:
        DataFrame con columnas ``"i"`` y ``"gaps_acumulados"``.
    """
    if df.is_empty():
        return pl.DataFrame(schema={"i": pl.Int64, "gaps_acumulados": pl.Int64})

    max_step = int(df["i"].max())
    seen_ids: set[int] = set()
    all_missing: set[int] = set()
    result = []

    for step in range(max_step + 1):
        step_ids = set(df.filter(pl.col("i") == step)["id"].to_list())
        missing = seen_ids - step_ids
        all_missing.update(missing)
        result.append({"i": step, "gaps_acumulados": len(all_missing)})
        seen_ids.update(step_ids)

    return pl.DataFrame(result)


def _compute_gaps_per_step_raw(df: pl.DataFrame) -> pl.DataFrame:
    """Calcula gaps por step (no acumulados) usando tracking por ID.

    Para cada step, cuenta cuántos IDs NO están presentes en el step siguiente.

    Args:
        df: DataFrame concatenado de todos los steps con columnas ``"i"`` e ``"id"``.

    Returns:
        DataFrame con columnas ``"i"`` y ``"gaps"``.
    """
    if df.is_empty():
        return pl.DataFrame(schema={"i": pl.Int64, "gaps": pl.Int64})

    max_step = int(df["i"].max())
    result = []

    for step in range(max_step + 1):
        step_ids = set(df.filter(pl.col("i") == step)["id"].to_list())
        if step < max_step:
            next_ids = set(df.filter(pl.col("i") == step + 1)["id"].to_list())
            gaps = len(step_ids - next_ids)
        else:
            gaps = 0
        result.append({"i": step, "gaps": gaps})

    return pl.DataFrame(result)


def _extract_plot_metadata(df: pl.DataFrame) -> dict[str, str | float]:
    """Extrae los metadatos de simulación del primer registro del DataFrame.

    Args:
        df: DataFrame con columnas de metadatos de la simulación.

    Returns:
        Diccionario con claves ``"ff"``, ``"np"``, ``"vel"``, ``"angle"``,
        ``"move1"``, ``"move2"``, ``"move3"``.
    """
    first = df.row(0, named=True)
    return {
        "ff": first.get("ff", "N/A"),
        "np": first.get("np", "N/A"),
        "vel": first.get("vel0", "N/A"),
        "angle": first.get("angulo", "N/A"),
        "move1": first.get("posx", "0"),
        "move2": first.get("posy", "0"),
        "move3": first.get("posz", "0"),
    }


def _draw_ejected(
    ax: plt.Axes,
    df_plot: pl.DataFrame,
    meta: dict[str, str | float],
    step_to_us: float,
) -> None:
    """Dibuja el subplot (a): partículas eyectadas vs tiempo.

    Args:
        ax: Eje matplotlib donde dibujar.
        df_plot: DataFrame con columnas ``"i"`` y ``"cantidad_eyectados"``.
        meta: Metadatos de la simulación para la leyenda.
        step_to_us: Factor de conversión timestep a microsegundos.
    """
    time_us = [i * step_to_us for i in df_plot["i"].to_list()]
    n_ejected = df_plot["cantidad_eyectados"].to_list()

    ax.plot(time_us, n_ejected, marker="o", linestyle="-", color="#2ecc71", linewidth=1.5, markersize=4)

    ax.set_xlabel(r"tiempo [$\mu$s]", fontsize=_AXIS_FONTSIZE)
    ax.set_ylabel("Y", fontsize=_AXIS_FONTSIZE)
    ax.tick_params(labelsize=_AXIS_FONTSIZE)
    ax.text(
        0.02, 0.98, "(a)", transform=ax.transAxes,
        fontsize=14, fontweight="bold", va="top", ha="left",
    )
    ax.grid(alpha=0.3)

    param_text = (
        f"ff = {meta['ff']}\n"
        f"N = {meta['np']}\n"
        f"v₀ = {meta['vel']} m/s\n"
        f"θ = {meta['angle']}°"
    )
    ax.text(
        0.98, 0.02, param_text, transform=ax.transAxes,
        fontsize=9, verticalalignment="bottom", horizontalalignment="right",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )


def _draw_clusters(
    ax: plt.Axes,
    df_clusters: pl.DataFrame,
    meta: dict[str, str | float],
    step_to_us: float,
) -> None:
    """Dibuja el subplot (b): número de clusters vs tiempo.

    Args:
        ax: Eje matplotlib donde dibujar.
        df_clusters: DataFrame con columnas ``"i"`` y ``"n_clusters"``.
        meta: Metadatos de la simulación.
        step_to_us: Factor de conversión timestep a microsegundos.
    """
    time_us = [i * step_to_us for i in df_clusters["i"].to_list()]
    n_clusters = df_clusters["n_clusters"].to_list()

    ax.plot(time_us, n_clusters, marker="s", linestyle="-", color="#3498db", linewidth=1.5, markersize=4)

    ax.set_xlabel(r"tiempo [$\mu$s]", fontsize=_AXIS_FONTSIZE)
    ax.set_ylabel("N clusters", fontsize=_AXIS_FONTSIZE)
    ax.tick_params(labelsize=_AXIS_FONTSIZE)
    ax.text(
        0.02, 0.98, "(b)", transform=ax.transAxes,
        fontsize=14, fontweight="bold", va="top", ha="left",
    )
    ax.grid(alpha=0.3)


def _draw_gaps_cumulative(
    ax: plt.Axes,
    df_gaps: pl.DataFrame,
    step_to_us: float,
) -> None:
    """Dibuja el subplot (c): gaps acumulados vs tiempo.

    Args:
        ax: Eje matplotlib donde dibujar.
        df_gaps: DataFrame con columnas ``"i"`` y ``"gaps_acumulados"``.
        step_to_us: Factor de conversión timestep a microsegundos.
    """
    time_us = [i * step_to_us for i in df_gaps["i"].to_list()]
    gaps_cum = df_gaps["gaps_acumulados"].to_list()

    ax.plot(time_us, gaps_cum, marker="^", linestyle="-", color="#e74c3c", linewidth=1.5, markersize=4)

    ax.set_xlabel(r"tiempo [$\mu$s]", fontsize=_AXIS_FONTSIZE)
    ax.set_ylabel("Gaps acumulados", fontsize=_AXIS_FONTSIZE)
    ax.tick_params(labelsize=_AXIS_FONTSIZE)
    ax.text(
        0.02, 0.98, "(c)", transform=ax.transAxes,
        fontsize=14, fontweight="bold", va="top", ha="left",
    )
    ax.grid(alpha=0.3)


def _draw_gaps_per_step(
    ax: plt.Axes,
    df_gaps: pl.DataFrame,
    step_to_us: float,
) -> None:
    """Dibuja el subplot (d): gaps por step vs tiempo.

    Args:
        ax: Eje matplotlib donde dibujar.
        df_gaps: DataFrame con columnas ``"i"`` y ``"gaps"``.
        step_to_us: Factor de conversión timestep a microsegundos.
    """
    time_us = [i * step_to_us for i in df_gaps["i"].to_list()]
    gaps = df_gaps["gaps"].to_list()

    ax.plot(time_us, gaps, marker="v", linestyle="-", color="#9b59b6", linewidth=1.5, markersize=4)

    ax.set_xlabel(r"tiempo [$\mu$s]", fontsize=_AXIS_FONTSIZE)
    ax.set_ylabel("Gaps por step", fontsize=_AXIS_FONTSIZE)
    ax.tick_params(labelsize=_AXIS_FONTSIZE)
    ax.text(
        0.02, 0.98, "(d)", transform=ax.transAxes,
        fontsize=14, fontweight="bold", va="top", ha="left",
    )
    ax.grid(alpha=0.3)
