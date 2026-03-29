"""
Lector de archivos dump de LAMMPS para CARBOXI_MINI.

Lee archivos ``output_aire.<step>.<core>.txt`` que contienen las partículas
eyectadas y el proyectil. Cada archivo tiene header estándar LAMMPS de 9
líneas seguido de datos espacio-con 12 columnas.

Física: Las columnas incluyen fuerza explícita en Z (``fz``), lo que
permite computar fuerza neta por cluster sin necesidad de derivarla
de posiciones o velocidades.
"""

import re
from pathlib import Path

import polars as pl

# Columnas estándar del dump de aire (sin encabezado en la lectura).
AIR_COLUMNS: list[str] = [
    "id", "type", "x", "y", "z",
    "vx", "vy", "vz",
    "omegax", "omegay", "omegaz", "fz",
]

# Líneas de header estándar LAMMPS a saltar.
_LAMMPS_HEADER_ROWS = 9

# Tamaño mínimo para considerar que un archivo tiene datos (header ~200 bytes).
# Archivos más pequeños solo contienen header sin filas de partículas.
_MIN_FILE_SIZE_BYTES = 500


def discover_air_files(sim_dir: Path) -> dict[int, list[Path]]:
    """Descubre archivos de aire y los agrupa por paso temporal.

    Busca ``output_aire.*.txt`` en el directorio y extrae el número de
    paso del nombre (formato ``output_aire.<step>.<core>.txt``).

    Args:
        sim_dir: Directorio de la simulación.

    Returns:
        Diccionario ``{step: [path_core0, path_core1, ...]}`` ordenado
        por step. Vacío si no se encuentran archivos.

    Example:
        >>> files = discover_air_files(Path("sim_dir"))
        >>> sorted(files.keys())
        [0, 25000, 50000]
    """
    all_files = sorted(Path(p) for p in sim_dir.glob("output_aire.*.txt"))

    grouped: dict[int, list[Path]] = {}
    for fpath in all_files:
        step = _extract_step_from_name(fpath.name)
        if step is not None:
            grouped.setdefault(step, []).append(fpath)

    return dict(sorted(grouped.items()))


def read_air_files(file_paths: list[Path]) -> pl.LazyFrame:
    """Lee y concatena archivos de aire de un mismo paso temporal.

    Cada archivo tiene header LAMMPS de 9 líneas que se descarta.
    Los datos de todos los cores se concatenan verticalmente.
    Los archivos vacíos o con solo header (cores sin eyección) se
    descartan automáticamente.

    Args:
        file_paths: Lista de archivos (uno por core) del mismo timestep.

    Returns:
        LazyFrame con las 12 columnas del dump de aire. Las columnas
        numéricas se leen como Float64, ``type`` como UInt8 e ``id``
        como Int64. Si todos los archivos están vacíos, retorna un
        LazyFrame con el schema correcto pero sin filas.

    Example:
        >>> lf = read_air_files([Path("output_aire.0.0.txt")])
        >>> lf.columns
        ['id', 'type', 'x', 'y', 'z', 'vx', 'vy', 'vz', ...]
    """
    frames: list[pl.LazyFrame] = []

    for fpath in file_paths:
        if fpath.stat().st_size < _MIN_FILE_SIZE_BYTES:
            continue

        lf = pl.scan_csv(
            fpath,
            separator=" ",
            has_header=False,
            skip_rows=_LAMMPS_HEADER_ROWS,
            truncate_ragged_lines=True,
            new_columns=AIR_COLUMNS,
            schema_overrides={
                "id": pl.Int64,
                "type": pl.UInt8,
                "fz": pl.String,
            },
            comment_prefix=None,
        )

        lf = lf.with_columns(
            pl.col("fz")
            .str.replace(r"#.*", "")
            .cast(pl.Float64, strict=False)
            .alias("fz")
        )

        frames.append(lf)

    if not frames:
        return _empty_air_lf()

    if len(frames) == 1:
        return frames[0]

    return pl.concat(frames, how="vertical_relaxed")


def get_timestep_interval(sorted_steps: list[int]) -> int:
    """Calcula el intervalo entre dumps a partir de los primeros dos pasos.

    Args:
        sorted_steps: Lista ordenada de números de paso temporal.

    Returns:
        Diferencia entre el segundo y primer paso (frecuencia de dump).

    Raises:
        ValueError: Si hay menos de 2 pasos disponibles.
    """
    if len(sorted_steps) < 2:
        raise ValueError(
            f"Se necesitan al menos 2 pasos para calcular intervalo, "
            f"se encontraron {len(sorted_steps)}"
        )
    return sorted_steps[1] - sorted_steps[0]


def _extract_step_from_name(filename: str) -> int | None:
    """Extrae el número de paso del nombre de archivo.

    Formato esperado: ``output_aire.<step>.<core>.txt``

    Args:
        filename: Nombre del archivo (sin ruta).

    Returns:
        Número de paso como entero, o None si no coincide.
    """
    match = re.search(r"output_aire\.(\d+)\.\d+\.txt", filename)
    if match:
        return int(match.group(1))
    return None


def _empty_air_lf() -> pl.LazyFrame:
    """Crea un LazyFrame vacío con el schema del dump de aire.

    Se usa cuando todos los archivos de un paso están vacíos (cores
    sin eyección). El LazyFrame tiene las 12 columnas correctas
    pero cero filas.

    Returns:
        LazyFrame vacío con schema: id→Int64, type→UInt8, resto→Float64.
    """
    return pl.LazyFrame({
        "id": pl.Series("id", [], dtype=pl.Int64),
        "type": pl.Series("type", [], dtype=pl.UInt8),
        "x": pl.Series("x", [], dtype=pl.Float64),
        "y": pl.Series("y", [], dtype=pl.Float64),
        "z": pl.Series("z", [], dtype=pl.Float64),
        "vx": pl.Series("vx", [], dtype=pl.Float64),
        "vy": pl.Series("vy", [], dtype=pl.Float64),
        "vz": pl.Series("vz", [], dtype=pl.Float64),
        "omegax": pl.Series("omegax", [], dtype=pl.Float64),
        "omegay": pl.Series("omegay", [], dtype=pl.Float64),
        "omegaz": pl.Series("omegaz", [], dtype=pl.Float64),
        "fz": pl.Series("fz", [], dtype=pl.Float64),
    })
