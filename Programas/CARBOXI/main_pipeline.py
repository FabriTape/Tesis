# -*- coding: utf-8 -*-
"""
Pipeline principal y punto de entrada de CARBOXI.

Orquesta el flujo completo de procesamiento de simulaciones LAMMPS:
- Análisis de onda de choque (Fase 1)
- Análisis topológico con OVITO (Fase 2)
- Exportación de resultados y gráficos SVG

Física: Cada simulación representa un experimento de impacto de un
proyectil sobre una cama de partículas de sílice.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl

from .config_and_discovery import (
    Config,
    browse_for_simulation,
    discover_simulations,
    extraer_metadatos,
    extraer_rotacion,
    load_config,
)
from .erosion_analysis import build_ovito_pipeline, run_topology_analysis
from .io import (
    detect_and_create_reader,
    plot_empty,
    plot_erosion,
    save_erosion_results,
    save_no_erosion_flag,
)
from .wave_analysis import is_critical_simulation, run_wave_scan

_SIM_PREFIX: str = "erosion_bed_SI_"
_TZ_ARG = timezone(timedelta(hours=-3))


def _parse_args(argv: list[str]) -> tuple[Path | None, Path | None]:
    """Extrae la carpeta de simulación y la raíz del proyecto desde argv.

    Returns:
        (base_dir, project_root); cualquiera puede ser None.
    """
    base_dir: Path | None = None
    project_root: Path | None = None
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg == "--project-root":
            i += 1
            if i >= len(argv):
                raise SystemExit("--project-root requiere un valor")
            project_root = Path(argv[i]).resolve()
        elif arg.startswith("--project-root="):
            project_root = Path(arg.split("=", 1)[1]).resolve()
        elif base_dir is None:
            base_dir = Path(arg).resolve()
        i += 1
    return base_dir, project_root


def main() -> None:
    """Punto de entrada principal de CARBOXI."""
    base_dir, project_root = _parse_args(sys.argv)

    if base_dir is None:
        selected = browse_for_simulation()
        if not selected:
            print("No se seleccionó ninguna carpeta. Saliendo.")
            return
        base_dir = selected[0].parent if len(selected) > 1 else selected[0]

    config = load_config(project_root=project_root)
    run_all(base_dir, config)


def run_simulation(sim_dir: Path, config: Config) -> None:
    """Ejecuta el pipeline completo para una simulación individual.

    Flujo:
    1. Preparación: listar dumps, extraer metadatos, calcular alturas.
    2. Decisión de onda: determinar si aplicar Fase 1.
    3. Fase 1 (opcional): escaneo de onda de choque.
    4. Fase 2: análisis topológico con OVITO.
    5. Exportación: guarda resultados Parquet y gráfico SVG.
    """
    sim_id = sim_dir.name.removeprefix(_SIM_PREFIX)
    timestamp = datetime.now(_TZ_ARG).strftime("%Y-%m-%d_%H-%M-%S")
    filter_name = config.filters.filtro_activo

    print(f"""\n{"=" * 55}
  Procesando: {sim_dir.name}
  Filtro activo: {filter_name}
{"=" * 55}""")

    reader = detect_and_create_reader(sim_dir)
    if not reader:
        print(f"  Sin datos (.dump o .gz) en '{sim_dir.name}'. Saltando...")
        return

    timestep_interval = reader.timestep_interval
    num_frames = reader.num_frames
    metadata = extraer_metadatos(sim_id)

    if not _metadata_valid(metadata):
        print("  No se pudieron extraer metadatos del nombre de carpeta. Saltando...")
        return

    # Extraer rotación desde el nombre de la carpeta PADRE
    rotacion = extraer_rotacion(sim_dir.parent.name)
    metadata.update(rotacion)
    rot = (
        f"_rot_{rotacion['rot_angulo']}_{rotacion['rot_eje_x']}"
        f"_{rotacion['rot_eje_y']}_{rotacion['rot_eje_z']}"
    )
    sim_id = f"{sim_id}{rot}"

    bed_height = config.bed_height_ff_015
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
            reader,
            config.radio_m,
            max_height,
            box_limits,
            sim_id,
            timestamp,
            config.dir_resultados,
            config,
        )
    else:
        cut_index = num_frames

    results = run_topology_analysis(
        reader=reader,
        cut_index=cut_index,
        pipeline=pipeline,
        bed_height=bed_height,
        max_height=max_height,
        metadata=metadata,
        timestep_interval=timestep_interval,
        freeze_height=config.freeze_height,
    )

    thresholds = {
        "umbral_vz_min": config.filters.umbral_vz_min,
        "umbral_techo_radios": config.filters.umbral_techo_radios,
    }

    if not results:
        path = save_no_erosion_flag(
            sim_id, timestamp, config.dir_resultados, filter_name
        )
        print(f"  Sin erosión significativa. Marcador guardado en: {path}")
        return

    path = save_erosion_results(
        results,
        sim_id,
        timestamp,
        config.dir_resultados,
        thresholds,
        filter_name=filter_name,
    )
    if not path:
        return
    print(f"\n  Resultados guardados en: {path}")

    df_total = pl.concat(results, how="vertical")
    if df_total.is_empty():
        print("  Sin datos para graficar. Generando gráfico vacío...")
        plot_empty(sim_id, timestamp, config.dir_imagenes, filter_name)
        return

    total_timesteps = df_total["i"].max()
    try:
        plot_erosion(
            df_total,
            sim_id,
            timestamp,
            config.dir_imagenes,
            filter_name,
            total_timesteps=total_timesteps,
        )
    except Exception as exc:
        print(f"  Advertencia: no se pudo generar el gráfico: {exc}")


def run_all(base_dir: Path, config: Config) -> None:
    """Procesa todas las simulaciones encontradas en el directorio base."""
    config.dir_resultados.mkdir(parents=True, exist_ok=True)
    config.dir_imagenes.mkdir(parents=True, exist_ok=True)

    simulations = discover_simulations(base_dir)

    for sim_dir in simulations:
        run_simulation(sim_dir, config)

    print(f"\n{'=' * 55}")
    print("  ANÁLISIS COMPLETADO")
    print(f"  Resultados en: {config.dir_resultados.resolve()}")
    print(f"{'=' * 55}\n")


def _metadata_valid(metadata: dict[str, str]) -> bool:
    """Verifica que los metadatos esenciales estén presentes."""
    return (
        metadata.get("ff") != "N/A"
        and metadata.get("np") != "N/A"
        and metadata.get("vel0") != "N/A"
    )


if __name__ == "__main__":
    main()
