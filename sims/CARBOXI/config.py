"""
Cargador de configuración centralizado para CARBOXI.

Lee el archivo config.toml usando tomllib (stdlib Python 3.11+) y expone
todos los parámetros físicos y de análisis como dataclasses inmutables
(frozen=True), garantizando acceso tipado sin dependencias externas.

Física: Los parámetros cubren umbrales cinéticos para detección de onda de
choque, criterios de erosión de la cama, propiedades del material sílice y
configuración del modelo JKR.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tomllib


@dataclass(frozen=True)
class WaveConfig:
    """Parámetros para la detección automática de onda de choque.

    Attributes:
        v_objetivo: Velocidad de impacto objetivo [m/s].
        tolerancia_v: Margen de error para comparar velocidades [m/s].
        np_criticos: Cantidades de partículas proyectil que activan el análisis
            automáticamente (sin verificar velocidad).
        move_referencia: Vector de posición de reposo del proyectil [m, m, m].
        tolerancia_move: Tolerancia para considerar desplazamiento no nulo [m].
        umbral_v2_min: Energía cinética mínima para partícula activa [m²/s²].
        umbral_v2_max: Energía cinética máxima para impacto crítico [m²/s²].
        particulas_criticas: Cantidad de partículas para onda macroscópica.
        umbral_techo: Margen vertical en número de radios respecto a la altura.
        umbral_pared: Radios de proximidad requerida a las paredes laterales.
    """

    v_objetivo: float
    tolerancia_v: float
    np_criticos: tuple[int, ...]
    move_referencia: tuple[float, ...]
    tolerancia_move: float
    umbral_v2_min: float
    umbral_v2_max: float
    particulas_criticas: int
    umbral_techo: int
    umbral_pared: int


@dataclass(frozen=True)
class FilterConfig:
    """Umbrales de filtrado para el criterio de erosión de partículas.

    Attributes:
        umbral_vz_min: Velocidad vertical mínima para considerar eyección [m/s].
        umbral_energia_min: Energía cinética mínima de erosión [m²/s²].
        umbral_techo_radios: Margen de seguridad vertical en número de radios.
        tolerancia_fz: Tolerancia de fuerza Z neta por cluster para vuelo
            balístico libre. Un cluster se considera eyectado si
            |Σfz_cluster| < tolerancia_fz [N].
        lonja_radios: Espesor de la lonja de análisis por debajo de
            max_height en número de radios (método taylor).
        method: Método de detección de erosión. "v1" usa filtro z+vz,
            "taylor" usa lonja con análisis de clusters por CM.
    """

    umbral_vz_min: float
    umbral_energia_min: float
    umbral_techo_radios: float
    tolerancia_fz: float
    lonja_radios: int
    method: str


@dataclass(frozen=True)
class Config:
    """Configuración completa del programa CARBOXI.

    Punto de entrada único para todos los parámetros de análisis. Se construye
    a partir de config.toml y no debe modificarse en tiempo de ejecución.

    Attributes:
        radio_m: Radio de las partículas en metros [m].
        usar_caja: Si True, usa los límites fijos de la caja como volumen de
            control; si False, usa bordes dinámicos según posición de partículas.
        wave: Parámetros de detección de onda de choque.
        filters: Umbrales de filtrado para criterio de erosión.
        factor_cutoff: Factor multiplicador sobre el radio para cutoff de OVITO.
        dir_resultados: Directorio raíz de salida de resultados.
        dir_imagenes: Subdirectorio de salida para gráficos SVG.
    """

    radio_m: float
    usar_caja: bool
    bed_height_ff_015: float
    bed_height_ff_035: float
    caja_x: float
    caja_y: float
    wave: WaveConfig
    filters: FilterConfig
    factor_cutoff: float
    dir_resultados: Path
    dir_imagenes: Path


def load_config(path: Path = Path("config.toml")) -> Config:
    """Carga y parsea el archivo de configuración TOML en una instancia de Config.

    Lee config.toml con tomllib (stdlib Python 3.11+), sin dependencias externas.
    Si el archivo no se encuentra en la ruta indicada, intenta buscarlo en el
    directorio del propio módulo.

    Args:
        path: Ruta al archivo config.toml. Por defecto busca en el directorio
            de trabajo actual.

    Returns:
        Config: Instancia inmutable con todos los parámetros de análisis.

    Raises:
        FileNotFoundError: Si config.toml no se encuentra en ninguna ruta válida.
        KeyError: Si faltan secciones obligatorias en el TOML.

    Example:
        >>> cfg = load_config(Path("config.toml"))
        >>> print(cfg.radio_m)
        7.6e-07
    """
    resolved = _resolve_config_path(path)

    with resolved.open("rb") as f:
        raw: dict = tomllib.load(f)

    return _build_config(raw)


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------


def _resolve_config_path(path: Path) -> Path:
    """Resuelve la ruta del archivo config.toml con fallback al directorio del módulo.

    Args:
        path: Ruta preferida para config.toml.

    Returns:
        Path: Ruta válida y existente al archivo de configuración.

    Raises:
        FileNotFoundError: Si no se encuentra config.toml en ninguna ubicación.
    """
    if path.exists():
        return path

    module_dir = Path(__file__).parent / "config.toml"
    if module_dir.exists():
        return module_dir

    raise FileNotFoundError(
        f"config.toml no encontrado en '{path}' ni en '{module_dir}'. "
        "Asegúrese de que el archivo de configuración esté presente."
    )


def _build_config(raw: dict) -> Config:
    """Construye la instancia Config a partir del diccionario TOML parseado.

    Args:
        raw: Diccionario crudo cargado desde TOML.

    Returns:
        Config: Instancia inmutable de configuración.
    """
    radio_um: float = raw["defaults"]["radio_um"]

    wave_raw = raw["analisis_onda"]
    onda_raw = raw["onda"]
    wave = WaveConfig(
        v_objetivo=float(wave_raw["v_objetivo"]),
        tolerancia_v=float(wave_raw["tolerancia_v"]),
        np_criticos=tuple(int(n) for n in wave_raw["np_criticos"]),
        move_referencia=tuple(float(v) for v in wave_raw["move_referencia"]),
        tolerancia_move=float(wave_raw["tolerancia_move"]),
        umbral_v2_min=float(onda_raw["umbral_v2_min"]),
        umbral_v2_max=float(onda_raw["umbral_v2_max"]),
        particulas_criticas=int(onda_raw["particulas_criticas"]),
        umbral_techo=int(onda_raw["umbral_techo"]),
        umbral_pared=int(onda_raw["umbral_pared"]),
    )

    filtros_raw = raw["filtros"]
    filters = FilterConfig(
        umbral_vz_min=float(filtros_raw["umbral_vz_min"]),
        umbral_energia_min=float(filtros_raw["umbral_energia_min"]),
        umbral_techo_radios=float(filtros_raw["umbral_techo_radios"]),
        tolerancia_fz=float(filtros_raw["tolerancia_fz"]),
        lonja_radios=int(filtros_raw.get("lonja_radios", 10)),
        method=str(filtros_raw.get("method", "v1")),
    )

    rutas = raw["rutas"]
    return Config(
        radio_m=radio_um * 1e-6,
        usar_caja=bool(raw["defaults"].get("usar_caja", True)),
        bed_height_ff_015=float(raw["defaults"]["bed_height_ff_015"]),
        bed_height_ff_035=float(raw["defaults"]["bed_height_ff_035"]),
        caja_x=float(raw["defaults"]["caja_x"]),
        caja_y=float(raw["defaults"]["caja_y"]),
        wave=wave,
        filters=filters,
        factor_cutoff=float(raw["jkr"]["factor_cutoff"]),
        dir_resultados=Path(rutas["resultados"]),
        dir_imagenes=Path(rutas["imagenes"]),
    )
