"""Descubrimiento de simulaciones LAMMPS en un directorio."""

import sys
from pathlib import Path

_SIM_PREFIX: str = "erosion_bed_SI_"


def discover_simulations(base_dir: Path) -> list[Path]:
    """Encuentra las carpetas de simulación a procesar desde un directorio base.

    Si ``base_dir`` ya es una carpeta de simulación (nombre que empieza con
    el prefijo estándar), devuelve una lista con ese único directorio.
    En caso contrario, busca subdirectorios con el prefijo y los devuelve
    ordenados alfabéticamente.

    Args:
        base_dir: Directorio raíz donde buscar. Puede ser directamente una
            carpeta de simulación o un contenedor de varias.

    Returns:
        Lista ordenada de rutas a las carpetas de simulación encontradas.

    Raises:
        SystemExit: Si no se encuentra ninguna simulación válida.
    """
    if base_dir.name.startswith(_SIM_PREFIX):
        return [base_dir]

    sims = sorted(
        p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith(_SIM_PREFIX)
    )

    if not sims:
        print(
            f"Error: No se encontraron carpetas '{_SIM_PREFIX}*' en '{base_dir}'.\n"
            "Verifique que la ruta seleccionada sea correcta."
        )
        sys.exit(1)

    print(f"Encontradas {len(sims)} simulaciones en '{base_dir.name}'.")
    return sims
