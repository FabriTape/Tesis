"""
Cargador de configuración centralizado para CARBOXI_MINI.

Lee config.toml con tomllib (stdlib Python 3.11+) y expone los parámetros
de análisis como dataclass inmutable (frozen=True).

Física: El radio de partícula define el cutoff topológico de OVITO para
cluster analysis. La tolerancia_fz define el umbral de fuerza Z neta
para considerar un cluster en vuelo balístico libre (sin contacto con cama).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tomllib


@dataclass(frozen=True)
class Config:
    """Configuración completa del programa CARBOXI_MINI.

    Se construye desde config.toml y no debe modificarse en tiempo
    de ejecución.

    Attributes:
        radio_m: Radio de las partículas en metros [m].
        factor_cutoff: Factor multiplicador sobre el radio para
            definir el cutoff de OVITO en cluster analysis.
            cutoff = radio_m * factor_cutoff.
        tolerancia_fz: Umbral absoluto de fuerza Z neta por cluster.
            Un cluster se considera en vuelo balístico si
            |fz_sum| < tolerancia_fz [N].
        dir_resultados: Directorio raíz de salida de resultados.
    """

    radio_m: float
    factor_cutoff: float
    tolerancia_fz: float
    dir_resultados: Path


def load_config(path: Path = Path("config.toml")) -> Config:
    """Carga y parsea el archivo de configuración TOML.

    Usa tomllib (stdlib Python 3.11+), sin dependencias externas.
    Si el archivo no existe en la ruta dada, busca en el directorio
    del propio módulo.

    Args:
        path: Ruta al archivo config.toml.

    Returns:
        Instancia inmutable de Config con todos los parámetros.

    Raises:
        FileNotFoundError: Si config.toml no se encuentra.
        KeyError: Si faltan secciones obligatorias en el TOML.

    Example:
        >>> cfg = load_config(Path("config.toml"))
        >>> cfg.radio_m
        7.6e-07
    """
    resolved = _resolve_config_path(path)

    with resolved.open("rb") as f:
        raw: dict = tomllib.load(f)

    return _build_config(raw)


def _resolve_config_path(path: Path) -> Path:
    """Resuelve la ruta del config.toml con fallback al directorio del módulo.

    Args:
        path: Ruta preferida.

    Returns:
        Ruta válida y existente.

    Raises:
        FileNotFoundError: Si no se encuentra en ninguna ubicación.
    """
    if path.exists():
        return path

    module_dir = Path(__file__).parent / "config.toml"
    if module_dir.exists():
        return module_dir

    raise FileNotFoundError(
        f"config.toml no encontrado en '{path}' ni en '{module_dir}'."
    )


def _build_config(raw: dict) -> Config:
    """Construye la instancia Config desde el diccionario TOML parseado.

    Soporta dos formatos de config.toml:

    - Formato local (``[defaults]`` + ``[rutas]``): usado cuando
      CARBOXIMINI tiene su propio config.toml.
    - Formato compartido (``[analysis]``): usado cuando se lee el
      config.toml centralizado del proyecto.

    Args:
        raw: Diccionario crudo de tomllib.

    Returns:
        Instancia inmutable de Config.
    """
    if "analysis" in raw:
        section = raw["analysis"]
        return Config(
            radio_m=float(section["radio_um"]) * 1e-6,
            factor_cutoff=float(section["factor_cutoff"]),
            tolerancia_fz=float(section["tolerancia_fz"]),
            dir_resultados=Path(section["resultados"]),
        )

    defaults = raw["defaults"]
    rutas = raw["rutas"]

    return Config(
        radio_m=float(defaults["radio_um"]) * 1e-6,
        factor_cutoff=float(defaults["factor_cutoff"]),
        tolerancia_fz=float(defaults["tolerancia_fz"]),
        dir_resultados=Path(rutas["resultados"]),
    )
