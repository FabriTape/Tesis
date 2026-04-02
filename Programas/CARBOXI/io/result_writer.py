"""
Escritor de resultados para el análisis de simulaciones CARBOXI.

Gestiona la persistencia de los resultados del pipeline de análisis en disco,
cubriendo tres tipos de salida: resultados de erosión completos (Parquet),
marcadores de simulaciones sin erosión (texto plano) y datos de onda de choque
detectada (Parquet fragmentado via sink).

Física: Los resultados almacenados representan el estado cinemático y
topológico de las partículas eyectadas de la cama de sílice en cada paso
temporal de la simulación LAMMPS.
"""

from pathlib import Path

import polars as pl


def save_erosion_results(
    results: list[pl.DataFrame],
    sim_id: str,
    timestamp: str,
    output_dir: Path,
    filter_thresholds: dict[str, float] | None = None,
    suffix: str = "",
    filter_name: str = "F3C",
) -> Path:
    """Consolida y guarda los resultados de erosión en formato Parquet.

    Args:
        results: Lista de DataFrames con los datos procesados por cada
            archivo de volcado (un DataFrame por timestep).
        sim_id: Identificador de la simulación (nombre de carpeta sin el
            prefijo ``erosion_bed_SI_``).
        timestamp: Marca temporal de inicio del análisis en formato
            ``"YYYY-MM-DD_HH-MM-SS"``.
        output_dir: Directorio raíz donde se escribirá el archivo.
        filter_thresholds: Diccionario opcional con los umbrales de filtrado
            utilizados. Se inyectan como columnas en el Parquet para
            trazabilidad del análisis.
        suffix: Sufijo opcional para el nombre del archivo.
        filter_name: Nombre del filtro de erosión aplicado (F1I, F1C, F2I,
            F2C, F3I, F3C). Se incluye en el nombre del archivo.

    Returns:
        Path al archivo Parquet creado.

    Example:
        >>> path = save_erosion_results(
        ...     results, "ff_0.15_np_1_vel_-50.0", "2025-07-01_12-00-00",
        ...     Path("./resultados"), filter_name="F3C"
        ... )
    """
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
    """Crea un archivo de texto indicando que la simulación no produjo erosión.

    Args:
        sim_id: Identificador de la simulación (nombre de carpeta sin prefijo).
        timestamp: Marca temporal en formato ``"YYYY-MM-DD_HH-MM-SS"``.
        output_dir: Directorio donde se creará el archivo de marcador.
        filter_name: Nombre del filtro de erosión aplicado.

    Returns:
        Path al archivo de texto creado.
    """
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
    """Guarda las partículas detectadas como parte de la onda de choque.

    Args:
        wave_frame: LazyFrame con las partículas clasificadas como onda de choque
            en el paso temporal de detección.
        sim_id: Identificador de la simulación.
        step_index: Índice numérico del archivo de volcado donde se detectó
            la onda (base 0).
        timestamp: Marca temporal del análisis en formato ``"YYYY-MM-DD_HH-MM-SS"``.
        output_dir: Directorio de salida.
        filter_name: Nombre del filtro de erosión aplicado.

    Returns:
        Path al archivo Parquet creado.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{sim_id}_ONDA_{filter_name}_{step_index}_{timestamp}.parquet"
    wave_frame.sink_parquet(path)
    return path
