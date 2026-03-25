"""
Módulo de visualización para el análisis de simulaciones LAMMPS.

Genera gráficos SVG de la evolución temporal de partículas eyectadas a
partir de los DataFrames producidos por el pipeline de análisis. El nombre
del archivo de salida incluye los parámetros de la simulación y un timestamp
para garantizar trazabilidad y unicidad.

Física: Cada punto del gráfico representa la cantidad acumulada de
partículas que superaron el umbral de erosión en un timestep dado.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl


def plot_erosion(
    df: pl.DataFrame,
    sim_id: str,
    timestamp: str,
    dir_imagenes: Path,
    suffix: str = "",
) -> None:
    """Genera y guarda el gráfico de evolución de partículas eyectadas.

    Agrupa los datos por iteración, cuenta las partículas erosionadas
    (incluyendo el proyectil si fue eyectado), y produce un gráfico de
    línea SVG con metadatos de la simulación en el título y en el nombre
    del archivo.

    Args:
        df: DataFrame con las columnas ``"i"`` (índice de iteración),
            ``"proyectil_erosion"`` (bool o None), ``"ff"``, ``"np"``,
            ``"vel0"``, ``"angulo"``, ``"posx"``, ``"posy"``, ``"posz"``.
        sim_id: Identificador de la simulación (nombre de carpeta sin prefijo
            ``erosion_bed_SI_``). Se usa como parte del nombre del archivo.
        timestamp: Marca temporal de la simulación en formato
            ``"YYYY-MM-DD_HH-MM-SS"``. Se incluye en el nombre del archivo SVG.
        dir_imagenes: Directorio de destino para el archivo SVG.
        suffix: Sufijo opcional para el nombre del archivo.

    Returns:
        None

    Raises:
        KeyError: Si el DataFrame no contiene las columnas requeridas.

    Example:
        >>> plot_erosion(df, "ff_0.15_np_1_vel_-50.0_...", "2025-07-01_12-00-00",
        ...              Path("./resultados/Imagenes"))
    """
    dir_imagenes.mkdir(parents=True, exist_ok=True)

    df_plot = _aggregate_ejected(df)
    meta = _extract_plot_metadata(df)

    _draw_figure(df_plot, meta)

    filename = f"plot_{sim_id}{suffix}_{timestamp}.svg"
    output_path = dir_imagenes / filename
    plt.savefig(output_path, format="svg")
    plt.close()

    print(f"Gráfico guardado en: {output_path}")


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------


def _aggregate_ejected(df: pl.DataFrame) -> pl.DataFrame:
    """Agrupa por iteración y cuenta partículas eyectadas en cada timestep.

    Considera eyectada a toda partícula con ``proyectil_erosion`` nulo
    (partícula de la cama) o True (proyectil eyectado).

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


def _draw_figure(df_plot: pl.DataFrame, meta: dict[str, str | float]) -> None:
    """Dibuja la figura matplotlib con los datos y metadatos de la simulación.

    Args:
        df_plot: DataFrame con columnas ``"i"`` y ``"cantidad_eyectados"``.
        meta: Diccionario con parámetros físicos de la simulación para el
            título y la leyenda.
    """
    plt.figure(figsize=(10, 6))

    plt.plot(
        df_plot["i"].to_list(),
        df_plot["cantidad_eyectados"].to_list(),
        marker="o",
        linestyle="-",
        color="steelblue",
        linewidth=1.5,
        markersize=4,
    )

    titulo = (
        "Evolución de partículas eyectadas\n"
        f"v₀={meta['vel']} m/s  |  ff={meta['ff']}  |  "
        f"N={meta['np']}  |  θ={meta['angle']}°"
    )
    plt.title(titulo, fontsize=12)
    plt.xlabel("Iteración (timestep)", fontsize=11)
    plt.ylabel("Partículas eyectadas acumuladas", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)

    param_text = (
        f"ff = {meta['ff']}\n"
        f"N = {meta['np']}\n"
        f"v₀ = {meta['vel']} m/s\n"
        f"θ = {meta['angle']}°\n"
        f"pos = ({meta['move1']}, {meta['move2']}, {meta['move3']})"
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
