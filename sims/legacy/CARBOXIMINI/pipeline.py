"""
Pipeline principal de análisis para CARBOXI_MINI.

Orquesta el flujo de procesamiento: descubrimiento de simulaciones,
detección de freno de emergencia (wave halt) desde salida.log,
análisis de clusters con OVITO, exportación Parquet y gráficos.

Física: Cada simulación representa un impacto de proyectil sobre cama
de sílice. El pipeline procesa los archivos de aire (partículas eyectadas
+ proyectil), agrupa por cluster y filtra los de vuelo balístico libre.
"""

from __future__ import annotations

import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl

from .analysis.topology import build_ovito_pipeline, run_topology_analysis
from .config import Config
from .io.lammps_reader import discover_air_files, get_timestep_interval
from .io.result_writer import save_no_result_flag, save_results, save_wave_flag
from .plot import plot_erosion
from .utils import extraer_metadatos

# Prefijo estándar de carpetas de simulación.
_SIM_PREFIX: str = "erosion_bed_SI_"

# Zona horaria Argentina (UTC-3).
_TZ_ARG = timezone(timedelta(hours=-3))


# ---------------------------------------------------------------------------
# Detección de freno de emergencia
# ---------------------------------------------------------------------------


def detect_wave_halt(sim_dir: Path) -> int | None:
    """Parsea salida.log buscando la condición de freno por onda de choque.

    Busca una línea con formato:
    ``Fix halt condition for fix-id wave_trigger met on step N ...``

    Args:
        sim_dir: Directorio de la simulación.

    Returns:
        Número de paso donde se activó el freno, o None si no se encontró
        (la simulación completó todos los pasos planificados).
    """
    log_path = sim_dir / "salida.log"
    if not log_path.exists():
        return None

    pattern = re.compile(r"met on step (\d+)")

    try:
        with log_path.open("r", errors="replace") as f:
            for line in f:
                if "Fix halt condition" in line and "wave_trigger" in line:
                    match = pattern.search(line)
                    if match:
                        return int(match.group(1))
    except OSError:
        return None

    return None


# ---------------------------------------------------------------------------
# Descubrimiento de simulaciones
# ---------------------------------------------------------------------------


def discover_simulations(base_dir: Path) -> list[Path]:
    """Encuentra carpetas de simulación a procesar.

    Si ``base_dir`` contiene archivos ``output_aire.*.txt``, se trata como
    una simulación única. Si no, busca subdirectorios que contengan dichos
    archivos.

    Args:
        base_dir: Directorio raíz proporcionado por CLI.

    Returns:
        Lista ordenada de rutas a directorios de simulación.

    Raises:
        SystemExit: Si no se encuentra ninguna simulación válida.
    """
    air_files = list(base_dir.glob("output_aire.*.txt"))
    if air_files:
        return [base_dir]

    sims: list[Path] = []
    for subdir in sorted(base_dir.iterdir()):
        if subdir.is_dir() and not subdir.name.startswith("."):
            if list(subdir.glob("output_aire.*.txt")):
                sims.append(subdir)

    if not sims:
        print(
            f"Error: No se encontraron archivos 'output_aire.*.txt' en "
            f"'{base_dir}' ni en sus subdirectorios.\n"
            "Verifique que la ruta sea correcta."
        )
        sys.exit(1)

    print(f"Encontradas {len(sims)} simulaciones en '{base_dir.name}'.")
    return sims


# ---------------------------------------------------------------------------
# Pipeline por simulación
# ---------------------------------------------------------------------------


def run_simulation(sim_dir: Path, config: Config) -> None:
    """Ejecuta el pipeline completo para una simulación individual.

    Flujo:

    1. Extrae metadatos del nombre de carpeta.
    2. Detecta freno de emergencia desde ``salida.log``.
       Si se detecta, guarda ``*_ONDA_*.txt`` y retorna sin análisis.
    3. Descubre archivos de aire agrupados por paso temporal.
    4. Ejecuta análisis de clusters + física con OVITO/Polars.
    5. Exporta resultados Parquet o marcador NADA.
    6. Genera gráfico SVG.

    Args:
        sim_dir: Directorio de la simulación.
        config: Configuración del programa.
    """
    sim_id = sim_dir.name.removeprefix(_SIM_PREFIX)
    if sim_id == sim_dir.name:
        sim_id = sim_dir.name

    timestamp = datetime.now(_TZ_ARG).strftime("%Y-%m-%d_%H-%M-%S")

    print(f"\n{'=' * 55}")
    print(f"  Procesando: {sim_dir.name}")
    print(f"{'=' * 55}")

    # --- 1. Metadatos ---
    metadata = extraer_metadatos(sim_id)

    # --- 2. Frenos de emergencia (sin análisis si se detecta) ---
    wave_halt_step = detect_wave_halt(sim_dir)
    if wave_halt_step is not None:
        print(f"  [!] Freno de emergencia en step {wave_halt_step}. Sin análisis.")
        path = save_wave_flag(
            sim_id, timestamp, wave_halt_step, metadata, config.dir_resultados
        )
        print(f"  Marcador guardado en: {path}")
        return

    # --- 3. Descubrir archivos de aire ---
    air_files_by_step = discover_air_files(sim_dir)
    if not air_files_by_step:
        print(f"  Sin archivos output_aire en '{sim_dir.name}'. Saltando...")
        return

    sorted_steps = sorted(air_files_by_step.keys())
    timestep_interval = get_timestep_interval(sorted_steps)
    print(f"  {len(sorted_steps)} pasos encontrados (intervalo={timestep_interval})")

    # --- 4. Análisis de clusters ---
    cutoff = config.radio_m * config.factor_cutoff
    pipeline = build_ovito_pipeline(cutoff=cutoff)

    results = run_topology_analysis(
        air_files_by_step=air_files_by_step,
        pipeline=pipeline,
        tolerancia_fz=config.tolerancia_fz,
        sim_id=sim_id,
        metadata=metadata,
        cutoff_step=sorted_steps[-1],
        timestep_interval=timestep_interval,
    )

    # --- 5. Exportación ---
    if not results:
        path = save_no_result_flag(sim_id, timestamp, config.dir_resultados)
        print(f"  Sin clusters balísticos. Marcador: {path}")
        return

    result_path = save_results(results, sim_id, timestamp, config.dir_resultados)
    print(f"\n  Resultados guardados en: {result_path}")

    # --- 6. Gráfico ---
    dir_imagenes = config.dir_resultados / "Imagenes"
    try:
        df_total = pl.concat(results, how="vertical_relaxed")
        plot_erosion(df_total, sim_id, timestamp, dir_imagenes)
    except Exception as exc:  # noqa: BLE001
        print(f"  Advertencia: no se pudo generar el gráfico: {exc}")


# ---------------------------------------------------------------------------
# Bucle principal
# ---------------------------------------------------------------------------


def run_all(base_dir: Path, config: Config) -> None:
    """Procesa todas las simulaciones encontradas en el directorio base.

    Args:
        base_dir: Directorio con una o varias simulaciones.
        config: Configuración del programa.
    """
    config.dir_resultados.mkdir(parents=True, exist_ok=True)

    simulations = discover_simulations(base_dir)

    for sim_dir in simulations:
        run_simulation(sim_dir, config)

    print(f"\n{'=' * 55}")
    print("  ANÁLISIS COMPLETADO")
    print(f"  Resultados en: {config.dir_resultados.resolve()}")
    print(f"{'=' * 55}\n")
