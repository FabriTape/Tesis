"""Lector de archivos dump LAMMPS (copia de CARBOXI/io/lammps_reader.py)."""

import re
from pathlib import Path

import polars as pl

_ALL_COLUMNS: list[str] = ["id", "type", "x", "y", "z", "vx", "vy", "vz", "fz"]


def read_dump(path: Path, columns: list[str] | None = None) -> pl.LazyFrame:
    """Lee un archivo dump comprimido de LAMMPS como LazyFrame de Polars.

    Args:
        path: Ruta al archivo .gz a leer.
        columns: Columnas a seleccionar. None = todas.

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
    """Lista y ordena archivos dump de una simulación.

    Args:
        directory: Directorio con archivos output.*.gz.

    Returns:
        Lista de rutas ordenadas por timestep.
    """
    files = list(directory.glob("output.*.gz"))
    files.sort(key=lambda p: _extract_timestep(p.name), reverse=True)
    return files


def get_timestep_interval(files: list[Path]) -> int:
    """Extrae el intervalo de dump desde la lista de archivos.

    Args:
        files: Lista de archivos dump ordenados.

    Returns:
        Intervalo de timestep entre dumps consecutivos.
    """
    if len(files) < 2:
        return 0
    return _extract_timestep(files[1].name)


def _extract_timestep(filename: str) -> int:
    """Extrae el timestep del nombre del archivo.

    Args:
        filename: Nombre del archivo, ej. 'output.80000.gz'.

    Returns:
        Timestep como entero.
    """
    match = re.search(r"output\.(\d+)\.gz", filename)
    return int(match.group(1)) if match else 0
