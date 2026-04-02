"""
Escritor de resultados para el análisis de simulaciones CARBOXI_MINI.

Gestiona la persistencia de resultados en disco: Parquet con propiedades
de clusters balísticos y marcadores de simulaciones sin eyección.

Física: Los resultados representan la fuerza Z neta y propiedades de
clusters de partículas eyectadas detectadas en cada paso temporal.
"""

from pathlib import Path

import polars as pl


def save_results(
    results: list[pl.DataFrame],
    sim_id: str,
    timestamp: str,
    output_dir: Path,
) -> Path:
    """Consolida y guarda los resultados de clusters en formato Parquet.

    Concatena los DataFrames por paso temporal, fusiona las columnas
    ``i`` y ``paso`` en ``step = i * paso``, y serializa a disco.

    Args:
        results: Lista de DataFrames con datos de clusters por timestep.
        sim_id: Identificador de la simulación (nombre de carpeta).
        timestamp: Marca temporal en formato ``"YYYY-MM-DD_HH-MM-SS"``.
        output_dir: Directorio raíz de salida.

    Returns:
        Path al archivo Parquet creado.

    Example:
        >>> path = save_results(results, "ff_0.15_...",
        ...     "2025-07-01_12-00-00", Path("./resultados"))
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pl.concat(results, how="vertical_relaxed")

    df = df.with_columns(
        step=(pl.col("i") * pl.col("paso")).cast(pl.Int64)
    ).drop("i", "paso")

    path = output_dir / f"{sim_id}_TOTAL_{timestamp}.parquet"
    df.write_parquet(path)
    return path


def save_no_result_flag(
    sim_id: str,
    timestamp: str,
    output_dir: Path,
) -> Path:
    """Crea un archivo de texto indicando que no hubo clusters balísticos.

    Args:
        sim_id: Identificador de la simulación.
        timestamp: Marca temporal en formato ``"YYYY-MM-DD_HH-MM-SS"``.
        output_dir: Directorio donde se creará el marcador.

    Returns:
        Path al archivo de texto creado.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{sim_id}_NADA_{timestamp}.txt"
    path.write_text(
        "No se detectaron clusters balísticos en esta simulación.\n"
        f"Simulación: {sim_id}\n"
        f"Timestamp: {timestamp}\n"
    )
    return path


def save_wave_flag(
    sim_id: str,
    timestamp: str,
    wave_halt_step: int,
    metadata: dict[str, str],
    output_dir: Path,
) -> Path:
    """Crea un archivo de texto indicando freno de emergencia por onda.

    Se usa cuando LAMMPS detecta una onda de choque y aborta la simulación
    antes del paso final. No se realiza ningún análisis de clusters.

    Args:
        sim_id: Identificador de la simulación.
        timestamp: Marca temporal en formato ``"YYYY-MM-DD_HH-MM-SS"``.
        wave_halt_step: Paso temporal donde LAMMPS activó el freno.
        metadata: Metadatos extraídos del nombre de carpeta (ff, np, etc.).
        output_dir: Directorio donde se creará el marcador.

    Returns:
        Path al archivo de texto creado.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{sim_id}_ONDA_{timestamp}.txt"

    lines = [
        "Freno de emergencia por onda de choque.",
        f"Simulación: {sim_id}",
        f"Step de freno: {wave_halt_step}",
        f"Timestamp: {timestamp}",
        (
            f"Metadata: ff={metadata.get('ff', 'N/A')}, "
            f"np={metadata.get('np', 'N/A')}, "
            f"vel={metadata.get('vel0', 'N/A')}, "
            f"angle={metadata.get('angulo', 'N/A')}"
        ),
    ]
    path.write_text("\n".join(lines) + "\n")
    return path
