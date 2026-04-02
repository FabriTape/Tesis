"""
Lector de archivos de volcado (dump) de simulaciones LAMMPS.

Proporciona funciones para leer archivos comprimidos output.*.gz generados
por LAMMPS, extraer los límites espaciales de la caja de simulación y
calcular la altura máxima basal de la cama de partículas.

Física: Los archivos dump contienen posiciones y velocidades de todas las
partículas en cada timestep. Las primeras 9 líneas son encabezado; la línea
5 contiene los límites en X y la línea 6 en Y. El tipo 1 corresponde a
las partículas de la cama; el tipo 2 es el proyectil.
"""

import re
from pathlib import Path

import polars as pl

# Nombres de columna estándar en los archivos dump de esta simulación.
_ALL_COLUMNS: list[str] = ["id", "type", "x", "y", "z", "vx", "vy", "vz", "fz"]


def read_dump(path: Path, columns: list[str] | None = None) -> pl.LazyFrame:
    """Lee un archivo dump comprimido de LAMMPS como LazyFrame de Polars.

    Omite las 9 líneas de encabezado del formato dump y asigna los nombres
    de columna estándar. La evaluación perezosa permite procesar archivos
    grandes sin cargar todo el contenido en memoria.

    Args:
        path: Ruta al archivo ``.gz`` a leer.
        columns: Lista de columnas a seleccionar. Si es ``None``, devuelve
            todas las columnas estándar: ``["id", "type", "x", "y", "z",
            "vx", "vy", "vz", "fz"]``.

    Returns:
        LazyFrame con las columnas solicitadas y tipos inferidos. La columna
        ``"type"`` se almacena como ``UInt8`` para eficiencia de memoria.
        ``"fz"`` es la fuerza Z neta por partícula [N].

    Example:
        >>> lf = read_dump(Path("sim/output.100.gz"), ["x", "y", "z", "vz", "fz"])
        >>> df = lf.collect()
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
    """Lista y ordena cronológicamente los archivos dump de una simulación.

    Busca todos los archivos con patrón ``output.*.gz`` en el directorio
    indicado y los ordena por el número de timestep extraído del nombre.

    Args:
        directory: Directorio raíz de la simulación que contiene los archivos
            ``output.*.gz``.

    Returns:
        Lista de rutas absolutas a los archivos dump, ordenados de menor a
        mayor timestep. Devuelve lista vacía si no se encuentran archivos.

    Example:
        >>> files = list_dump_files(Path("sim/erosion_bed_SI_ff_0.15_np_1_..."))
        >>> files[0].name
        'output.0.gz'
    """
    files = list(directory.glob("output.*.gz"))
    files.sort(key=lambda p: _extract_timestep(p.name))
    return files


def get_timestep_interval(files: list[Path]) -> int:
    """Extrae el intervalo de volcado (dump frequency) desde la lista de archivos.

    El intervalo corresponde al número de timestep del segundo archivo, que
    en las simulaciones LAMMPS de erosión equivale a la frecuencia con la
    que se escriben los frames.

    Args:
        files: Lista de archivos dump ordenados cronológicamente.

    Returns:
        Intervalo de timestep entre archivos consecutivos. Devuelve ``0``
        si la lista tiene menos de 2 elementos.

    Example:
        >>> interval = get_timestep_interval(files)
        >>> print(interval)
        10000
    """
    if len(files) < 2:
        return 0
    return _extract_timestep(files[1].name)


def _extract_timestep(filename: str) -> int:
    """Extrae el número de timestep del nombre de un archivo dump.

    Args:
        filename: Nombre del archivo, ej. ``"output.80000.gz"``.

    Returns:
        Timestep como entero. Devuelve ``0`` si no se puede parsear.
    """
    match = re.search(r"output\.(\d+)\.gz", filename)
    return int(match.group(1)) if match else 0
