"""
Punto de entrada CLI para CARBOXI_MINI.

Uso:
    python -m CARBOXI_MINI /ruta/a/simulacion
"""

import argparse
import sys
from pathlib import Path

from .config import load_config
from .pipeline import run_all


def main() -> None:
    """Parsea argumentos CLI y ejecuta el pipeline de análisis."""
    parser = argparse.ArgumentParser(
        prog="CARBOXI_MINI",
        description=(
            "Análisis de clusters de partículas eyectadas en "
            "simulaciones LAMMPS de física granular."
        ),
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Ruta a la carpeta de simulación o directorio con múltiples simulaciones.",
    )

    args = parser.parse_args()
    base_dir: Path = args.directory.resolve()

    if not base_dir.exists():
        print(f"Error: La ruta '{base_dir}' no existe.", file=sys.stderr)
        sys.exit(1)

    if not base_dir.is_dir():
        print(f"Error: '{base_dir}' no es un directorio.", file=sys.stderr)
        sys.exit(1)

    config = load_config(Path(__file__).parent / "config.toml")

    print(f"CARBOXI_MINI — Radio: {config.radio_m * 1e6:.2f} μm")
    print(f"Directorio: {base_dir}")
    print(f"Salida: {config.dir_resultados.resolve()}")

    run_all(base_dir, config)


if __name__ == "__main__":
    main()
