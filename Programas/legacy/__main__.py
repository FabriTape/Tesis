"""
Punto de entrada del programa CARBOXI.

Ejecutar con:
    python -m CARBOXI /ruta/a/carpeta/simulaciones
    python -m CARBOXI /ruta/a/carpeta/simulaciones --config /ruta/config.toml

Si se proporciona --config, usa ese archivo de configuración.
Si no, busca config.toml en la carpeta de simulaciones (copiado por el script).
Si no existe, usa el config.toml por defecto del paquete.

Física: CARBOXI analiza simulaciones de erosión por impacto de partículas
generadas con LAMMPS, extrayendo datos cinéticos y topológicos de la cama
de sílice y el proyectil en cada timestep.
"""

from __future__ import annotations

import sys
from pathlib import Path

from .config import load_config
from .pipeline import browse_for_simulation, run_all


def _parse_args() -> tuple[Path | None, Path | None]:
    """Parsea argumentos CLI para extraer directorio de datos y ruta de config.

    Returns:
        Tupla (data_dir, config_path). Cada una puede ser None si no se proporcionó.

    Example:
        >>> data, cfg = _parse_args()
        >>> # python -m CARBOXI /sims --config /sims/config.toml
    """
    data_dir: Path | None = None
    config_path: Path | None = None

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--config" and i + 1 < len(args):
            config_path = Path(args[i + 1]).resolve()
            i += 2
        elif data_dir is None:
            data_dir = Path(args[i]).resolve()
            i += 1
        else:
            i += 1

    return data_dir, config_path


def _resolve_config(data_dir: Path | None, cli_config: Path | None) -> Path:
    """Resuelve la ruta del config.toml con fallback automático.

    Prioridad:
    1. --config explícito
    2. config.toml en el directorio de datos (copiado por el script)
    3. config.toml del paquete (por defecto)

    Args:
        data_dir: Directorio de simulaciones.
        cli_config: Ruta proporcionada por --config.

    Returns:
        Path al config.toml a usar.
    """
    if cli_config and cli_config.exists():
        return cli_config

    if data_dir:
        sim_config = data_dir / "config.toml"
        if sim_config.exists():
            return sim_config

    # Fallback: buscar en la carpeta de simulaciones del repo
    repo_config = Path(__file__).parents[1] / "cluster" / "config.toml"
    if repo_config.exists():
        return repo_config

    raise FileNotFoundError(
        "No se encontró config.toml. Use --config /ruta/config.toml "
        "o coloque config.toml en el directorio de simulaciones."
    )


def main() -> None:
    """Inicializa la configuración y lanza el pipeline de análisis.

    Si se pasa una ruta por argumento CLI, la usa directamente para
    descubrir simulaciones. En caso contrario, abre el menú de navegación
    interactivo de carpetas.

    Returns:
        None

    Raises:
        SystemExit: Si el directorio de datos no existe o no se encuentran
            simulaciones válidas.
    """
    data_dir, cli_config = _parse_args()
    config_path = _resolve_config(data_dir, cli_config)
    config = load_config(config_path)

    print("\n  CARBOXI — Análisis de simulaciones LAMMPS")
    print(f"  Configuración      : {config_path}")
    print(f"  Radio de partícula : {config.radio_m * 1e6:.4f} µm")
    print(f"  Directorio salida  : {config.dir_resultados.resolve()}\n")

    # Resolver directorio de datos: argumento CLI o navegador interactivo
    if data_dir:
        if not data_dir.exists():
            print(f"Error: El directorio '{data_dir}' no existe.")
            sys.exit(1)
        run_all(data_dir, config)
    else:
        sim_dirs = browse_for_simulation()
        if not sim_dirs:
            print("No se seleccionó ninguna carpeta. Saliendo.")
            sys.exit(0)
        for base_dir in sim_dirs:
            run_all(base_dir, config)


if __name__ == "__main__":
    main()
