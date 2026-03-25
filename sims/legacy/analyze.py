"""
Analizador automático de simulaciones LAMMPS.

Detecta el formato de los archivos de datos y ejecuta el pipeline
correspondiente (CARBOXI para output.*.gz o CARBOXIMINI para
output_aire.*.txt).

Uso:
    python analyze.py /ruta/a/simulaciones
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def detectar_formato(data_dir: Path) -> str:
    """Detecta el formato de los archivos de simulación.

    Busca recursivamente los patrones de nombre característicos de
    cada versión del pipeline.

    Args:
        data_dir: Directorio raíz a analizar.

    Returns:
        ``"mini"`` si se encuentran archivos ``output_aire.*.txt``,
        ``"legacy"`` si se encuentran archivos ``output.*.gz``.

    Raises:
        SystemExit: Si no se encuentra ningún formato conocido.
    """
    if list(data_dir.rglob("output_aire.*.txt")):
        return "mini"

    if list(data_dir.rglob("output.*.gz")):
        return "legacy"

    print(
        f"Error: No se encontraron archivos de simulación en '{data_dir}'.\n"
        "Se esperan archivos 'output_aire.*.txt' (CARBOXIMINI) "
        "o 'output.*.gz' (CARBOXI)."
    )
    sys.exit(1)


def find_config_path(data_dir: Path, script_dir: Path) -> Path:
    """Busca config.toml en el directorio de datos y luego en el script.

    Prioridad:
    1. ``<data_dir>/config.toml`` (config de la simulación)
    2. ``<script_dir>/config.toml`` (config del paquete)

    Args:
        data_dir: Directorio de datos de simulación.
        script_dir: Directorio donde vive analyze.py.

    Returns:
        Ruta al config.toml encontrado.

    Raises:
        SystemExit: Si no se encuentra config.toml.
    """
    candidates = [
        data_dir / "config.toml",
        script_dir / "config.toml",
    ]

    for path in candidates:
        if path.exists():
            return path

    paths = "\n  ".join(str(p) for p in candidates)
    print(f"Error: config.toml no encontrado en:\n  {paths}")
    sys.exit(1)


def main() -> None:
    """Detecta el formato y ejecuta el pipeline correspondiente."""
    parser = argparse.ArgumentParser(
        description="Analizador automático de simulaciones LAMMPS.",
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Ruta al directorio de simulación(es).",
    )

    args = parser.parse_args()
    base_dir: Path = args.directory.resolve()

    if not base_dir.exists() or not base_dir.is_dir():
        print(f"Error: '{base_dir}' no es un directorio válido.", file=sys.stderr)
        sys.exit(1)

    # Permite imports de CARBOXI y CARBOXIMINI desde el directorio padre
    script_dir = Path(__file__).resolve().parent
    parent_dir = str(script_dir.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    config_path = find_config_path(base_dir, script_dir)
    formato = detectar_formato(base_dir)

    if formato == "mini":
        from CARBOXIMINI.config import load_config
        from CARBOXIMINI.pipeline import run_all

        config = load_config(config_path)
        print("Formato detectado: CARBOXIMINI (output_aire.*.txt)")
        print(f"CARBOXIMINI — Radio: {config.radio_m * 1e6:.2f} μm")
    else:
        from CARBOXI.config import load_config
        from CARBOXI.pipeline import run_all

        config = load_config(config_path)
        print("Formato detectado: CARBOXI (output.*.gz)")
        print("\n  CARBOXI — Análisis de simulaciones LAMMPS")
        print(f"  Radio de partícula : {config.radio_m * 1e6:.4f} μm")

    print(f"Directorio: {base_dir}")
    print(f"Config: {config_path}")
    print(f"Salida: {config.dir_resultados.resolve()}")

    run_all(base_dir, config)


if __name__ == "__main__":
    main()
