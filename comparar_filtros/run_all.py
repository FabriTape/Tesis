"""Script para correr las 6 variaciones de filtro.

Uso:
    python run_all.py
"""

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
PYTHON = sys.executable

# (nombre_subdir, radios, tracking_on_off)
CONFIGS = [
    ("0r_no_tracking", 0, "off"),
    ("0r_tracking", 0, "on"),
    ("2r_no_tracking", 2, "off"),
    ("2r_tracking", 2, "on"),
    ("5r_no_tracking", 5, "off"),
    ("5r_tracking", 5, "on"),
]


def main() -> None:
    """Ejecuta las 6 variaciones."""
    for name, radios, tracking in CONFIGS:
        print(f"\n{'=' * 60}")
        print(f"  {name}: radios={radios}, tracking={tracking}")
        print(f"{'=' * 60}")

        result = subprocess.run(
            [PYTHON, str(BASE_DIR / "main.py"), str(radios), tracking, "1e-6", name],
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
