"""
Pipeline principal de análisis para CARBOXI.

Orquesta el flujo completo de procesamiento de simulaciones LAMMPS:
análisis de onda de choque (Fase 1), análisis topológico con OVITO (Fase 2),
exportación de resultados y generación de gráficos SVG.

Física: Cada simulación representa un experimento de impacto de un
proyectil sobre una cama de partículas de sílice. El pipeline extrae
las partículas eyectadas, sus propiedades topológicas y el instante
en que se propaga una onda macroscópica en la cama.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl

from .analysis.topology import build_ovito_pipeline, run_topology_analysis
from .analysis.wave import is_critical_simulation, run_wave_scan
from .config import Config
from .discovery import discover_simulations
from .io.lammps_reader import (
    get_timestep_interval,
    list_dump_files,
)
from .io.result_writer import (
    save_erosion_results,
    save_no_erosion_flag,
)
from .plot import _plot_empty, plot_erosion
from .utils import extraer_metadatos

# Prefijo estándar de las carpetas de simulación LAMMPS de erosión.
_SIM_PREFIX: str = "erosion_bed_SI_"

# Zona horaria de Argentina (UTC-3) para timestamps de los resultados.
_TZ_ARG = timezone(timedelta(hours=-3))


def run_simulation(sim_dir: Path, config: Config) -> None:
    """Ejecuta el pipeline completo de análisis para una simulación individual.

    Flujo:
    1. Preparación: listar dumps, extraer metadatos, calcular alturas.
    2. Decisión de onda: determinar si aplicar Fase 1.
    3. Fase 1 (opcional): escaneo rápido de onda de choque.
    4. Fase 2: análisis topológico con OVITO sobre frames relevantes.
    5. Exportación: guarda resultados Parquet y gráfico SVG.

    Args:
        sim_dir: Directorio de la simulación (nombre comienza con
            ``erosion_bed_SI_``).
        config: Configuración completa cargada desde ``config.toml``.
    """
    sim_id = sim_dir.name.removeprefix(_SIM_PREFIX)
    timestamp = datetime.now(_TZ_ARG).strftime("%Y-%m-%d_%H-%M-%S")
    filter_name = config.filters.filtro_activo

    print(f"\n{'=' * 55}")
    print(f"  Procesando: {sim_dir.name}")
    print(f"  Filtro activo: {filter_name}")
    print(f"{'=' * 55}")

    dump_files = list_dump_files(sim_dir)
    if not dump_files:
        print(f"  Sin archivos output.*.gz en '{sim_dir.name}'. Saltando...")
        return

    timestep_interval = get_timestep_interval(dump_files)
    metadata = extraer_metadatos(sim_id)

    if not _metadata_valid(metadata):
        print("  No se pudieron extraer metadatos del nombre de carpeta. Saltando...")
        return

    bed_height = _select_bed_height(metadata, config)
    max_height = bed_height + config.filters.umbral_techo_radios * config.radio_m

    box_limits = {"x": [0.0, config.caja_x], "y": [0.0, config.caja_y]}

    run_wave = is_critical_simulation(metadata, config)
    if run_wave:
        print(
            f"  [!] Simulación crítica — análisis de onda ACTIVADO\n"
            f"      np={metadata.get('np')}, "
            f"vel={metadata.get('vel0')} m/s, "
            f"pos=({metadata.get('posx')}, {metadata.get('posy')}, {metadata.get('posz')})"
        )

    cutoff = config.radio_m * config.factor_cutoff
    pipeline = build_ovito_pipeline(cutoff=cutoff)

    if run_wave:
        _, cut_index = run_wave_scan(
            dump_files=dump_files,
            radius=config.radio_m,
            max_height=max_height,
            box_limits=box_limits,
            use_static_box=True,
            sim_id=sim_id,
            timestamp=timestamp,
            output_dir=config.dir_resultados,
            config=config,
            filter_name=filter_name,
        )
    else:
        cut_index = len(dump_files)

    files_fase2 = dump_files[:cut_index]
    results = run_topology_analysis(
        dump_files=files_fase2,
        pipeline=pipeline,
        bed_height=bed_height,
        max_height=max_height,
        sim_id=sim_id,
        timestep_interval=timestep_interval,
        tolerancia_v_cm=config.filters.tolerancia_v_cm,
        freeze_height=config.freeze_height,
        filter_name=filter_name,
    )

    filter_thresholds = {
        "umbral_vz_min": config.filters.umbral_vz_min,
        "umbral_techo_radios": config.filters.umbral_techo_radios,
        "tolerancia_v_cm": config.filters.tolerancia_v_cm,
    }

    if not results:
        path = save_no_erosion_flag(
            sim_id, timestamp, config.dir_resultados, filter_name
        )
        print(f"  Sin erosión significativa. Marcador guardado en: {path}")
        return

    result_path = save_erosion_results(
        results,
        sim_id,
        timestamp,
        config.dir_resultados,
        filter_thresholds,
        filter_name=filter_name,
    )
    print(f"\n  Resultados guardados en: {result_path}")

    df_total = pl.concat(results, how="vertical_relaxed")

    if df_total.is_empty() or df_total.height == 0:
        print("  Sin datos para graficar. Generando gráfico vacío...")
        _plot_empty(sim_id, timestamp, config.dir_imagenes, filter_name)
        return

    try:
        plot_erosion(df_total, sim_id, timestamp, config.dir_imagenes, filter_name)
    except Exception as exc:  # noqa: BLE001
        print(f"  Advertencia: no se pudo generar el gráfico: {exc}")


def run_all(base_dir: Path, config: Config) -> None:
    """Procesa todas las simulaciones encontradas en el directorio base.

    Args:
        base_dir: Directorio que contiene una o varias carpetas de simulación.
        config: Configuración completa del programa.
    """
    config.dir_resultados.mkdir(parents=True, exist_ok=True)
    config.dir_imagenes.mkdir(parents=True, exist_ok=True)

    simulations = discover_simulations(base_dir)

    for sim_dir in simulations:
        run_simulation(sim_dir, config)

    print(f"\n{'=' * 55}")
    print("  ANÁLISIS COMPLETADO")
    print(f"  Resultados en: {config.dir_resultados.resolve()}")
    print(f"{'=' * 55}\n")


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------


def _metadata_valid(metadata: dict[str, str]) -> bool:
    """Verifica que los metadatos esenciales estén presentes.

    Args:
        metadata: Diccionario con parámetros extraídos del nombre de carpeta.

    Returns:
        True si ff, np y vel0 están presentes y no son "N/A".
    """
    return (
        metadata.get("ff") != "N/A"
        and metadata.get("np") != "N/A"
        and metadata.get("vel0") != "N/A"
    )


def _select_bed_height(metadata: dict[str, str], config: Config) -> float:
    """Selecciona la altura de la cama según el valor de ff.

    Args:
        metadata: Diccionario con clave ``"ff"``.
        config: Configuración con alturas para ff_015 y ff_035.

    Returns:
        Altura de la cama en metros [m].
    """
    ff_value = metadata.get("ff", "N/A")
    if ff_value == "0.15":
        return config.bed_height_ff_015
    elif ff_value == "0.35":
        return config.bed_height_ff_035
    else:
        print(
            f"  Advertencia: ff={ff_value} no reconocido, usando ff_0.15 por defecto."
        )
        return config.bed_height_ff_015
