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
) -> Path:
    """Consolida y guarda los resultados de erosión en formato Parquet.

    Concatena todos los DataFrames por paso temporal, inyecta los umbrales
    de filtrado como columnas de trazabilidad y serializa a disco. El nombre
    del archivo incluye el identificador de simulación y la marca temporal
    para garantizar unicidad y reproducibilidad.

    Args:
        results: Lista de DataFrames con los datos procesados por cada
            archivo de volcado (un DataFrame por timestep).
        sim_id: Identificador de la simulación (nombre de carpeta sin el
            prefijo ``erosion_bed_SI_``).
        timestamp: Marca temporal de inicio del análisis en formato
            ``"YYYY-MM-DD_HH-MM-SS"``.
        output_dir: Directorio raíz donde se escribirá el archivo.
        filter_thresholds: Diccionario opcional con los umbrales de filtrado
            utilizados (ej. ``{"umbral_vz_min": 0.0, "umbral_energia_min": 2e-4}``).
            Si se proporciona, se inyectan como columnas en el Parquet para
            trazabilidad del análisis.
        suffix: Sufijo opcional para el nombre del archivo (ej. ``"_filtrado_taylor"``).

    Returns:
        Path al archivo Parquet creado.

    Example:
        >>> path = save_erosion_results(
        ...     results, "ff_0.15_np_1_vel_-50.0", "2025-07-01_12-00-00",
        ...     Path("./resultados")
        ... )
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pl.concat(results, how="vertical_relaxed")

    if filter_thresholds:
        df = df.with_columns(**{k: pl.lit(v) for k, v in filter_thresholds.items()})

    path = output_dir / f"{sim_id}_TOTAL{suffix}_{timestamp}.parquet"
    df.write_parquet(path)
    return path


def save_no_erosion_flag(sim_id: str, timestamp: str, output_dir: Path) -> Path:
    """Crea un archivo de texto indicando que la simulación no produjo erosión.

    Se utiliza como marcador para que análisis futuros identifiquen que esta
    simulación ya fue procesada y no generó eyección significativa de partículas.

    Args:
        sim_id: Identificador de la simulación (nombre de carpeta sin prefijo).
        timestamp: Marca temporal en formato ``"YYYY-MM-DD_HH-MM-SS"``.
        output_dir: Directorio donde se creará el archivo de marcador.

    Returns:
        Path al archivo de texto creado.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{sim_id}_NADA_{timestamp}.txt"
    path.write_text(
        "No se obtuvo erosión significativa en esta simulación.\n"
        f"Simulación: {sim_id}\n"
        f"Timestamp: {timestamp}\n"
    )
    return path


def save_wave_data(
    wave_frame: pl.LazyFrame,
    sim_id: str,
    step_index: int,
    timestamp: str,
    output_dir: Path,
) -> Path:
    """Guarda las partículas detectadas como parte de la onda de choque.

    Serializa el LazyFrame con las partículas de la onda usando sink_parquet,
    evitando materializar todo el conjunto en memoria. El nombre del archivo
    incluye el índice del paso temporal donde se detectó la onda.

    Args:
        wave_frame: LazyFrame con las partículas clasificadas como onda de choque
            en el paso temporal de detección.
        sim_id: Identificador de la simulación.
        step_index: Índice numérico del archivo de volcado donde se detectó
            la onda (base 0).
        timestamp: Marca temporal del análisis en formato ``"YYYY-MM-DD_HH-MM-SS"``.
        output_dir: Directorio de salida.

    Returns:
        Path al archivo Parquet creado.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{sim_id}_ONDA_{step_index}_{timestamp}.parquet"
    wave_frame.sink_parquet(path)
    return path
