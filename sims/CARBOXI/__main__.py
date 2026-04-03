"""
Punto de entrada del programa CARBOXI.

Ejecutar con:
    python -m CARBOXI
    python -m CARBOXI /ruta/a/carpeta/simulaciones

Si se proporciona una ruta como argumento, la usa directamente.
Si no, despliega un menú interactivo de navegación de carpetas.

Física: CARBOXI analiza simulaciones de erosión por impacto de partículas
generadas con LAMMPS, extrayendo datos cinéticos y topológicos de la cama
de sílice y el proyectil en cada timestep.
"""

from __future__ import annotations

import sys
from pathlib import Path

from .config import load_config
from .pipeline import browse_for_simulation, run_all


def main() -> None:
    """Inicializa la configuración y lanza el pipeline de análisis.

    Si se pasa una ruta por argumento CLI, la usa directamente para
    descubrir simulaciones. En caso contrario, abre el menú de navegación
    interactivo de carpetas.

    Carga ``config.toml`` desde el directorio del paquete y delega el
    procesamiento a ``run_all``.

    Returns:
        None

    Raises:
        SystemExit: Si el directorio de datos no existe o no se encuentran
            simulaciones válidas.
    """
    # Cargar configuración desde el directorio del paquete
    config_path = Path(__file__).parent / "config.toml"
    config = load_config(config_path)

    print("\n  CARBOXI — Análisis de simulaciones LAMMPS")
    print(f"  Radio de partícula : {config.radio_m * 1e6:.4f} µm")
    print(f"  Directorio salida  : {config.dir_resultados.resolve()}\n")

    # Resolver directorio de datos: argumento CLI o navegador interactivo
    if len(sys.argv) > 1:
        base_dir = Path(sys.argv[1]).resolve()
        if not base_dir.exists():
            print(f"Error: El directorio '{base_dir}' no existe.")
            sys.exit(1)
        run_all(base_dir, config)
    else:
        sim_dirs = browse_for_simulation()
        if not sim_dirs:
            print("No se seleccionó ninguna carpeta. Saliendo.")
            sys.exit(0)
        for base_dir in sim_dirs:
            run_all(base_dir, config)


if __name__ == "__main__":
    main()
