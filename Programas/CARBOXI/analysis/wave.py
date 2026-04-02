"""
Detección de onda de choque en simulaciones de erosión LAMMPS.

Provee funciones para evaluar si una simulación requiere análisis de onda
y para identificar el instante en que se propaga una onda macroscópica
en la cama de partículas usando criterios cinéticos y espaciales.

Física: La onda de choque se manifiesta cuando un gran número de partículas
de la cama adquieren velocidad simultáneamente cerca de las paredes laterales,
indicando que el impulso del proyectil se propagó a través del lecho.
"""

from __future__ import annotations

import math
from pathlib import Path

import polars as pl

from ..config import Config, WaveConfig
from ..io.lammps_reader import read_dump
from ..io.result_writer import save_wave_data
from ..utils import print_progress

# Columnas mínimas necesarias para el análisis de onda.
_WAVE_COLUMNS: list[str] = ["id", "type", "x", "y", "z", "vx", "vy", "vz"]


def is_critical_simulation(metadata: dict[str, str], config: Config) -> bool:
    """Determina si una simulación requiere análisis de onda de choque.

    Evalúa tres condiciones físicas en orden de prioridad:

    1. **Número de partículas del proyectil**: Si ``np`` está en
       ``np_criticos`` (típicamente 1 o 3), la simulación es automáticamente
       crítica sin verificar otros parámetros, ya que son los únicos valores
       posibles en ese tipo de experimento.
    2. **Velocidad de impacto**: El módulo de la velocidad debe coincidir
       con ``v_objetivo`` dentro de ``tolerancia_v``.
    3. **Desplazamiento del proyectil**: La norma del vector de posición
       inicial debe ser mayor que ``tolerancia_move`` respecto al vector
       de referencia (indica que hay un desplazamiento lateral).

    Args:
        metadata: Diccionario con parámetros extraídos del nombre de carpeta.
            Claves esperadas: ``"np"``, ``"vel0"``, ``"posx"``, ``"posy"``,
            ``"posz"``.
        config: Configuración completa del programa con umbrales de onda.

    Returns:
        ``True`` si la simulación cumple las condiciones para análisis de onda,
        ``False`` en caso contrario.

    Example:
        >>> meta = {"np": "1", "vel0": "-50.0", "posx": "0.1",
        ...         "posy": "0", "posz": "0"}
        >>> is_critical_simulation(meta, config)
        True
    """
    wave_cfg = config.wave

    np_val = _safe_int(metadata.get("np", "0"))
    if np_val is None:
        return False

    # Regla 1: np en np_criticos → automáticamente crítico (única opción posible).
    if np_val in wave_cfg.np_criticos:
        return _has_nonzero_displacement(metadata, wave_cfg)

    # Regla 2: verificación de velocidad de impacto.
    vel0 = _safe_float(metadata.get("vel0", "0"))
    if vel0 is None:
        return False
    if abs(vel0) >= abs(wave_cfg.v_objetivo) - wave_cfg.tolerancia_v:
        return False

    # Regla 3: desplazamiento no nulo del proyectil.
    return _has_nonzero_displacement(metadata, wave_cfg)


def detect_wave_particles(
    lf: pl.LazyFrame,
    radius: float,
    max_height: float,
    box_limits: dict[str, list[float]],
    use_static_box: bool,
    wave_cfg: WaveConfig,
) -> tuple[pl.LazyFrame, int, float | None]:
    """Filtra las partículas que forman parte del frente de onda de choque.

    Aplica tres criterios secuenciales:

    - **Vertical**: descarta partículas eyectadas muy por encima de la cama
      (``z < max_height - umbral_techo * R``), para no confundirlas con
      proyectiles rebotados.
    - **Cinético**: descarta partículas estáticas (``v² > umbral_v2_min``).
    - **Perimetral**: exige que el evento alcance las fronteras laterales
      (proximidad a paredes en X o Y), indicando propagación macroscópica.

    Args:
        lf: LazyFrame con columnas ``["id", "type","x", "y", "z", "vx", "vy", "vz"]``.
        radius: Radio físico de las partículas [m].
        max_height: Altura máxima basal de la cama en reposo [m].
        box_limits: Diccionario ``{"x": [min, max], "y": [min, max]}`` con
            los límites de la caja de simulación [m].
        use_static_box: Si ``True``, usa los límites fijos de ``box_limits``
            como referencia lateral; si ``False``, usa la posición máxima
            dinámica de las partículas.
        wave_cfg: Configuración de parámetros de detección de onda.

    Returns:
        Tupla de tres elementos:

        - ``pl.LazyFrame``: Subconjunto de partículas clasificadas como onda.
        - ``int``: Cantidad de partículas detectadas en la onda.
        - ``float | None``: Velocidad cuadrática máxima ``v²`` entre las
          partículas de la onda. ``None`` si no hay partículas.
    """
    x_max = box_limits["x"][1] if use_static_box else pl.col("x").max()
    y_max = box_limits["y"][1] if use_static_box else pl.col("y").max()

    lf_wave = lf.with_columns(
        (pl.col("vx") ** 2 + pl.col("vy") ** 2 + pl.col("vz") ** 2).alias("v2")
    ).filter(
        (pl.col("z") < max_height - wave_cfg.umbral_techo * radius)
        & (pl.col("v2") > wave_cfg.umbral_v2_min)
        & (
            (pl.col("x") < wave_cfg.umbral_pared * radius)
            | (x_max - wave_cfg.umbral_pared * radius < pl.col("x"))
            | (pl.col("y") < wave_cfg.umbral_pared * radius)
            | (y_max - wave_cfg.umbral_pared * radius < pl.col("y"))
        )
    )

    stats = lf_wave.select(
        pl.len().alias("count"),
        pl.col("v2").max().alias("max_v2"),
    ).collect()

    count = int(stats["count"].item())
    max_v2 = stats["max_v2"].item()

    return lf_wave, count, max_v2


def run_wave_scan(
    dump_files: list[Path],
    radius: float,
    max_height: float,
    box_limits: dict[str, list[float]],
    use_static_box: bool,
    sim_id: str,
    timestamp: str,
    output_dir: Path,
    config: Config,
    filter_name: str = "F3C",
) -> tuple[bool, int]:
    """Escanea archivos dump secuencialmente buscando la onda de choque.

    Itera sobre los archivos de volcado en orden cronológico. En cada paso
    aplica ``detect_wave_particles`` y evalúa si se cumple el criterio de
    onda macroscópica (Fail-Fast). Al detectarla, guarda los datos del frame
    y devuelve el índice de corte para que la Fase 2 no procese más allá.

    Args:
        dump_files: Lista de rutas a archivos ``output.*.gz`` en orden
            cronológico.
        radius: Radio físico de las partículas [m].
        max_height: Altura máxima basal de la cama [m].
        box_limits: Límites espaciales de la caja de simulación [m].
        use_static_box: Si ``True``, usa bordes fijos para el cálculo perimetral.
        sim_id: Identificador de la simulación (sin prefijo ``erosion_bed_SI_``).
        timestamp: Marca temporal del análisis en formato ``"YYYY-MM-DD_HH-MM-SS"``.
        output_dir: Directorio donde guardar los datos de onda detectada.
        config: Configuración completa con umbrales de detección.
        filter_name: Nombre del filtro de erosión aplicado.

    Returns:
        Tupla de dos elementos:

        - ``bool``: ``True`` si se detectó una onda macroscópica.
        - ``int``: Índice del archivo donde se detuvo el escaneo. Si no se
          detectó onda, equivale a ``len(dump_files)``.
    """
    wave_cfg = config.wave
    total = len(dump_files)

    for i, file_path in enumerate(dump_files):
        print_progress(i, total, label="onda")

        lf = read_dump(file_path, _WAVE_COLUMNS)
        lf_wave, count, max_v2 = detect_wave_particles(
            lf, radius, max_height, box_limits, use_static_box, wave_cfg
        )

        if _wave_threshold_reached(count, max_v2, wave_cfg):
            print(
                f"\n[ONDA] Detectada en '{file_path.name}' (paso {i}): "
                f"{count} partículas, v²_max={max_v2:.4e}"
            )
            save_wave_data(lf_wave, sim_id, i, timestamp, output_dir, filter_name)
            print_progress(total, total, label="onda")
            print()
            return True, i

    print_progress(total, total, label="onda")
    print()
    return False, total


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------


def _wave_threshold_reached(
    count: int, max_v2: float | None, wave_cfg: WaveConfig
) -> bool:
    """Evalúa si el frame actual supera el umbral crítico de onda macroscópica.

    Args:
        count: Número de partículas detectadas en la onda.
        max_v2: Velocidad cuadrática máxima entre las partículas [m²/s²].
        wave_cfg: Configuración de umbrales de detección.

    Returns:
        ``True`` si se supera el umbral de cantidad de partículas o de
        velocidad máxima.
    """
    v2_exceeded = isinstance(max_v2, float) and max_v2 >= wave_cfg.umbral_v2_max
    return count > wave_cfg.particulas_criticas or v2_exceeded


def _has_nonzero_displacement(metadata: dict[str, str], wave_cfg: WaveConfig) -> bool:
    """Verifica que el proyectil tenga un desplazamiento lateral no nulo.

    Args:
        metadata: Diccionario con claves ``"posx"``, ``"posy"``, ``"posz"``.
        wave_cfg: Configuración con vector de referencia y tolerancia.

    Returns:
        ``True`` si la norma euclidiana del desplazamiento supera
        ``tolerancia_move``.
    """
    posx = _safe_float(metadata.get("posx", "0")) or 0.0
    posy = _safe_float(metadata.get("posy", "0")) or 0.0
    posz = _safe_float(metadata.get("posz", "0")) or 0.0

    ref = wave_cfg.move_referencia
    norm = math.sqrt((posx - ref[0]) ** 2 + (posy - ref[1]) ** 2 + (posz - ref[2]) ** 2)
    return norm > wave_cfg.tolerancia_move


def _safe_float(value: str | None) -> float | None:
    """Convierte una cadena a float, devolviendo None ante error de parseo."""
    try:
        return float(value) if value is not None else None
    except (ValueError, TypeError):
        return None


def _safe_int(value: str | None) -> int | None:
    """Convierte una cadena a int, devolviendo None ante error de parseo."""
    try:
        return int(value) if value is not None else None
    except (ValueError, TypeError):
        return None
