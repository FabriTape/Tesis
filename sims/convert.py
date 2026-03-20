"""
Conversor de archivos dump mixtos LAMMPS a formato legacy gzip.

Transforma archivos ``output.<step>.<core>.txt`` (13 columnas) en
``output.<step>.gz`` (8 columnas, header LAMMPS) compatibles con
el pipeline CARBOXI. Reemplaza los archivos originales en el lugar.

Uso:
    python convert.py /ruta/a/tanda_de_simulaciones
"""

from __future__ import annotations

import argparse
import gzip
import re
import sys
from pathlib import Path

import polars as pl

# Columnas del formato mixto (13 cols, sin encabezado en lectura).
MIXED_COLUMNS: list[str] = [
    "id", "type", "x", "y", "z",
    "vx", "vy", "vz",
    "omegax", "omegay", "omegaz", "fz", "c_1",
]

# Columnas del formato legacy CARBOXI (8 cols).
LEGACY_COLUMNS: list[str] = ["id", "type", "x", "y", "z", "vx", "vy", "vz"]

# Líneas de header estándar LAMMPS a saltar.
_LAMMPS_HEADER_ROWS = 9

# Tamaño mínimo para considerar que un archivo tiene datos (header ~200 bytes).
_MIN_FILE_SIZE_BYTES = 500

# Regex para archivos mixtos: output.<step>.<core>.txt
_MIXED_PATTERN = re.compile(r"^output\.(\d+)\.\d+\.txt$")


def discover_mixed_files(sim_dir: Path) -> dict[int, list[Path]]:
    """Descubre archivos mixtos y los agrupa por paso temporal.

    Busca ``output.<step>.<core>.txt`` que no sean ``output_aire``
    ni ``output_cama``.

    Args:
        sim_dir: Directorio de la simulación.

    Returns:
        Diccionario ``{step: [archivos]}`` ordenado por step.
    """
    grouped: dict[int, list[Path]] = {}

    for fpath in sorted(sim_dir.glob("output.*.txt")):
        match = _MIXED_PATTERN.match(fpath.name)
        if match:
            step = int(match.group(1))
            grouped.setdefault(step, []).append(fpath)

    return dict(sorted(grouped.items()))


def convert_step(
    file_paths: list[Path],
    step: int,
    output_path: Path,
) -> int:
    """Convierte archivos mixtos de un paso a un .gz legacy.

    Lee cada archivo individualmente (filtrando vacíos por tamaño),
    extrae las 8 columnas originales y escribe un archivo gzip con
    header LAMMPS estándar.

    Args:
        file_paths: Archivos mixtos del mismo timestep.
        step: Número de paso temporal.
        output_path: Ruta de salida ``output.<step>.gz``.

    Returns:
        Número de átomos escritos.
    """
    dfs: list[pl.DataFrame] = []

    for fpath in file_paths:
        if fpath.stat().st_size < _MIN_FILE_SIZE_BYTES:
            continue

        dfs.append(
            pl.read_csv(
                fpath,
                separator=" ",
                has_header=False,
                skip_rows=_LAMMPS_HEADER_ROWS,
                new_columns=MIXED_COLUMNS,
                truncate_ragged_lines=True,
                comment_prefix=None,
            ).select(LEGACY_COLUMNS)
        )

    if not dfs:
        return 0

    df = pl.concat(dfs, how="vertical_relaxed")
    n_atoms = df.height

    with gzip.open(output_path, "wt") as f:
        f.write("ITEM: TIMESTEP\n")
        f.write(f"{step}\n")
        f.write("ITEM: NUMBER OF ATOMS\n")
        f.write(f"{n_atoms}\n")
        f.write("ITEM: BOX BOUNDS pp pp pp\n")
        f.write("0 1\n0 1\n0 1\n")
        f.write("ITEM: ATOMS id type x y z vx vy vz\n")
        df.write_csv(f, separator=" ", include_header=False)

    return n_atoms


def convert_simulation(sim_dir: Path) -> None:
    """Convierte todos los archivos mixtos de una simulación.

    Descubre archivos ``output.<step>.<core>.txt``, los agrupa por
    paso temporal, genera ``output.<step>.gz`` por cada paso y
    elimina los originales.

    Args:
        sim_dir: Directorio de la simulación.
    """
    mixed = discover_mixed_files(sim_dir)

    if not mixed:
        print("  0 archivos mixtos. Saltando.")
        return

    total_files = sum(len(v) for v in mixed.values())
    print(f"  {len(mixed)} pasos encontrados ({total_files} archivos)")

    converted: list[Path] = []

    for step, file_paths in mixed.items():
        output_path = sim_dir / f"output.{step}.gz"
        n_atoms = convert_step(file_paths, step, output_path)

        if output_path.stat().st_size > 0:
            converted.append(output_path)
        else:
            print(f"  Advertencia: output.{step}.gz vacío ({n_atoms} átomos)")

    if not converted:
        print("  No se generaron archivos. Abortando eliminación.")
        return

    deleted = 0
    for file_list in mixed.values():
        for fpath in file_list:
            fpath.unlink()
            deleted += 1

    names = ", ".join(p.name for p in converted)
    print(f"  Convertidos: {names}")
    print(f"  Eliminados: {deleted} archivos originales")


def main() -> None:
    """Parsea argumentos y ejecuta la conversión sobre todas las sims."""
    parser = argparse.ArgumentParser(
        description="Conversor de dumps mixtos LAMMPS a formato legacy gzip.",
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Ruta al directorio con simulación(es).",
    )

    args = parser.parse_args()
    base_dir: Path = args.directory.resolve()

    if not base_dir.exists() or not base_dir.is_dir():
        print(f"Error: '{base_dir}' no es un directorio válido.", file=sys.stderr)
        sys.exit(1)

    sim_dirs = _find_simulations(base_dir)

    if not sim_dirs:
        print("No se encontraron carpetas de simulación.")
        sys.exit(0)

    for sim_dir in sim_dirs:
        print(f"\nProcesando: {sim_dir.name}")
        convert_simulation(sim_dir)

    print(f"\n{'=' * 50}")
    print("CONVERSIÓN COMPLETADA")
    print(f"{'=' * 50}")


def _find_simulations(base_dir: Path) -> list[Path]:
    """Encuentra directorios de simulación con archivos mixtos.

    Si ``base_dir`` contiene archivos mixtos, lo trata como simulación
    única. Si no, busca subdirectorios con archivos mixtos.

    Args:
        base_dir: Directorio raíz.

    Returns:
        Lista de directorios de simulación.
    """
    mixed = list(base_dir.glob("output.*.txt"))
    real_mixed = [f for f in mixed if _MIXED_PATTERN.match(f.name)]

    if real_mixed:
        return [base_dir]

    sims: list[Path] = []
    for subdir in sorted(base_dir.iterdir()):
        if subdir.is_dir() and not subdir.name.startswith("."):
            sub_mixed = [
                f for f in subdir.glob("output.*.txt")
                if _MIXED_PATTERN.match(f.name)
            ]
            if sub_mixed:
                sims.append(subdir)

    return sims


if __name__ == "__main__":
    main()
