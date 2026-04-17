"""
Análisis de onda de choque en simulaciones LAMMPS.

Detecta el instante en que se propaga una onda macroscópica en la cama
de partículas usando criterios cinéticos y espaciales.

Física: La onda de choque se manifiesta cuando un gran número de partículas
de la cama adquieren velocidad simultáneamente cerca de las paredes laterales,
indicando que el impulso del proyectil se propagó a través del lecho.
"""

from __future__ import annotations

import math
from pathlib import Path

import polars as pl

from .config_and_discovery import Config, WaveConfig, print_progress
from .storage_and_viz import LammpsTrajectoryReader, save_wave_data

_WAVE_COLUMNS: list[str] = ["id", "type", "x", "y", "z", "vx", "vy", "vz"]


def is_critical_simulation(metadata: dict[str, str], config: Config) -> bool:
    """Determina si una simulación requiere análisis de onda de choque.

    Evalúa tres condiciones físicas en orden de prioridad:
    1. Número de partículas del proyectil en np_criticos.
    2. Velocidad de impacto dentro de tolerancia.
    3. Desplazamiento lateral no nulo del proyectil.

    Args:
        metadata: Diccionario con np, vel0, posx, posy, posz.
        config: Configuración con umbrales de onda.

    Returns:
        True si la simulación requiere análisis de onda.
    """
    wave_cfg = config.wave

    np_val = _safe_int(metadata.get("np", "0"))
    if np_val is None:
        return False

    if np_val in wave_cfg.np_criticos:
        return _has_nonzero_displacement(metadata, wave_cfg)

    vel0 = _safe_float(metadata.get("vel0", "0"))
    if vel0 is None:
        return False
    if abs(vel0) >= abs(wave_cfg.v_objetivo) - wave_cfg.tolerancia_v:
        return False

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

    Aplica tres criterios:
    - Vertical: z < max_height - umbral_techo * R
    - Cinético: v² > umbral_v2_min
    - Perimetral: proximidad a paredes laterales

    Returns:
        Tuple (LazyFrame filtrado, count, max_v2)
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
    reader: LammpsTrajectoryReader,
    radius: float,
    max_height: float,
    box_limits: dict[str, list[float]],
    use_static_box: bool,
    sim_id: str,
    timestamp: str,
    output_dir: Path,
    config: Config,
) -> tuple[bool, int]:
    """Escanea frames secuencialmente buscando la onda de choque.

    Args:
        reader: LammpsTrajectoryReader con los frames de la simulación.
        radius: Radio de las partículas [m].
        max_height: Altura máxima para filtrado [m].
        box_limits: Límites de la caja en X e Y.
        use_static_box: Si True, usa los límites fijos de la caja.
        sim_id: Identificador de la simulación.
        timestamp: Marca temporal para archivos de salida.
        output_dir: Directorio de resultados.
        config: Configuración completa.

    Returns:
        Tuple (wave_detected, cut_index).
    """
    wave_cfg = config.wave
    total = reader.num_frames

    for i in range(total):
        print_progress(i, total, label="onda")

        lf = reader.get_frame_lazy(i)
        lf_wave, count, max_v2 = detect_wave_particles(
            lf, radius, max_height, box_limits, use_static_box, wave_cfg
        )

        if _wave_threshold_reached(count, max_v2, wave_cfg):
            print(f"\n[ONDA] Detectada en frame {i}: {count} partículas, v²_max={max_v2:.4e}")
            save_wave_data(lf_wave, sim_id, i, timestamp, output_dir, "F3C")
            print_progress(total, total, label="onda")
            print()
            return True, i

    print_progress(total, total, label="onda")
    print()
    return False, total


def _wave_threshold_reached(
    count: int, max_v2: float | None, wave_cfg: WaveConfig
) -> bool:
    """Evalúa si el frame actual supera el umbral crítico de onda macroscópica."""
    v2_exceeded = isinstance(max_v2, float) and max_v2 >= wave_cfg.umbral_v2_max
    return count > wave_cfg.particulas_criticas or v2_exceeded


def _has_nonzero_displacement(metadata: dict[str, str], wave_cfg: WaveConfig) -> bool:
    """Verifica que el proyectil tenga un desplazamiento lateral no nulo."""
    posx = _safe_float(metadata.get("posx", "0")) or 0.0
    posy = _safe_float(metadata.get("posy", "0")) or 0.0
    posz = _safe_float(metadata.get("posz", "0")) or 0.0

    ref = wave_cfg.move_referencia
    norm = math.sqrt((posx - ref[0]) ** 2 + (posy - ref[1]) ** 2 + (posz - ref[2]) ** 2)
    return norm > wave_cfg.tolerancia_move


def _safe_float(value: str | None) -> float | None:
    """Convierte una cadena a float, devolviendo None ante error."""
    try:
        return float(value) if value is not None else None
    except (ValueError, TypeError):
        return None


def _safe_int(value: str | None) -> int | None:
    """Convierte una cadena a int, devolviendo None ante error."""
    try:
        return int(value) if value is not None else None
    except (ValueError, TypeError):
        return None
