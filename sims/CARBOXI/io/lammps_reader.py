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

import gzip
import re
from pathlib import Path

import polars as pl

# Nombres de columna estándar en los archivos dump de esta simulación.
_ALL_COLUMNS: list[str] = ["id", "type", "x", "y", "z", "vx", "vy", "vz"]

# Líneas del encabezado donde se encuentran los límites de la caja.
# Índice base-0: línea 5 = eje X, línea 6 = eje Y.
_BOX_LINES: dict[str, int] = {"x": 5, "y": 6}


def read_dump(path: Path, columns: list[str] | None = None) -> pl.LazyFrame:
    """Lee un archivo dump comprimido de LAMMPS como LazyFrame de Polars.

    Omite las 9 líneas de encabezado del formato dump y asigna los nombres
    de columna estándar. La evaluación perezosa permite procesar archivos
    grandes sin cargar todo el contenido en memoria.

    Args:
        path: Ruta al archivo ``.gz`` a leer.
        columns: Lista de columnas a seleccionar. Si es ``None``, devuelve
            todas las columnas estándar: ``["id", "type", "x", "y", "z",
            "vx", "vy", "vz"]``.

    Returns:
        LazyFrame con las columnas solicitadas y tipos inferidos. La columna
        ``"type"`` se almacena como ``UInt8`` para eficiencia de memoria.

    Example:
        >>> lf = read_dump(Path("sim/output.100.gz"), ["x", "y", "z", "vz"])
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


def read_box_limits(path: Path) -> dict[str, list[float]]:
    """Extrae los límites espaciales de la caja de simulación desde un dump.

    Lee las líneas de encabezado del archivo comprimido sin cargarlo completo
    en memoria. Devuelve los límites mínimo y máximo para los ejes X e Y.

    Args:
        path: Ruta al archivo dump comprimido (``output.*.gz``).

    Returns:
        Diccionario con claves ``"x"`` e ``"y"``, cada una con una lista
        ``[min, max]`` expresada en metros [m].

    Example:
        >>> limits = read_box_limits(Path("sim/output.0.gz"))
        >>> limits["x"]
        [0.0, 0.002]
    """
    limits: dict[str, list[float]] = {"x": [], "y": []}
    line_to_axis = {v: k for k, v in _BOX_LINES.items()}
    max_line = max(_BOX_LINES.values())

    with gzip.open(path, "rt") as f:
        for i, line in enumerate(f):
            if i in line_to_axis:
                axis = line_to_axis[i]
                limits[axis] = [float(v) for v in line.strip().split()]
            if i >= max_line:
                break

    return limits


def get_max_bed_height(path: Path, radius: float, margin_radii: int = 5) -> float:
    """Calcula la altura máxima de la cama de partículas en el estado inicial.

    Filtra las partículas de tipo 1 (material de la cama) en el frame inicial
    y devuelve su coordenada Z máxima más un margen de seguridad expresado
    en número de radios.

    Args:
        path: Ruta al archivo dump inicial, típicamente ``output.0.gz``.
        radius: Radio físico de las partículas [m].
        margin_radii: Número de radios añadidos como margen de tolerancia
            sobre la altura máxima. Por defecto es ``5``.

    Returns:
        Altura de corte [m] usada como umbral para clasificar partículas
        como erosionadas (``z > max_height``).

    Raises:
        ValueError: Si no se encuentran partículas de tipo 1 en el archivo,
            lo que indica un formato de simulación no compatible.

    Example:
        >>> h = get_max_bed_height(Path("sim/output.0.gz"), radius=7.6e-7)
        >>> print(f"Altura de cama: {h:.4e} m")
    """
    lf = read_dump(path, ["type", "z"])
    max_z = lf.filter(pl.col("type") == 1).select(pl.col("z").max()).collect().item()

    if max_z is None:
        raise ValueError(
            "No se encontraron partículas de tipo 1. "
            "Verifique los IDs de partículas en el archivo dump."
        )

    return float(max_z) + margin_radii * radius


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
