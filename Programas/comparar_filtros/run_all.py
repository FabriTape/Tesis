"""Script para correr las 6 variaciones de filtro.

Ejecuta main.py una vez por cada configuración de margen definida en
CONFIGS, pasando como argumentos: radios (margen sobre la cama),
tolerancia Δv fija (1e-6) y el nombre del subdirectorio de salida.

Uso:
    python run_all.py
"""

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
PYTHON = sys.executable

# (nombre_subdir, radios)
# radios < 0 → bed_height = base - N radios (umbral más bajo)
# radios > 0 → max_height = base + N radios (margen sobre la cama)
CONFIGS = [
    ("-5r", -5),
    ("0r", 0),
    ("5r", 5),
]


def main() -> None:
    """Ejecuta las 6 variaciones."""
    for name, radios in CONFIGS:
        print(f"\n{'=' * 60}")
        print(f"  {name}: radios={radios}")
        print(f"{'=' * 60}")

        result = subprocess.run(
            [PYTHON, str(BASE_DIR / "main.py"), str(radios), "1e-6", name],
            cwd=str(BASE_DIR),
        )

        if result.returncode != 0:
            print(f"  ERROR en {name}")
            sys.exit(1)

    print(f"\n{'=' * 60}")
    print("  Todas las configuraciones completadas.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
