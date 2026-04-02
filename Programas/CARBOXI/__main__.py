"""Punto de entrada para ejecutar CARBOXI como módulo.

Uso:
    python3.11 -m CARBOXI [ruta_a_carpeta_de_simulaciones]

Si no se proporciona ruta, se abre el navegador interactivo de carpetas.
"""

import sys
from pathlib import Path

from .browser import browse_for_simulation
from .config import load_config
from .pipeline import run_all


def main() -> None:
    """Función principal que orquesta la carga de configuración y ejecución."""
    if len(sys.argv) > 1:
        base_dir = Path(sys.argv[1]).resolve()
    else:
        selected = browse_for_simulation()
        if not selected:
            print("No se seleccionó ninguna carpeta. Saliendo.")
            return
        base_dir = selected[0].parent if len(selected) == 1 else selected[0].parent

    config = load_config()
    run_all(base_dir, config)


if __name__ == "__main__":
    main()
