"""
Pipeline principal de análisis para CARBOXI.

Orquesta el flujo completo de procesamiento de simulaciones LAMMPS:
navegación interactiva de carpetas, análisis de onda de choque (Fase 1),
análisis topológico con OVITO (Fase 2), exportación de resultados
y generación de gráficos SVG.

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
from .io.lammps_reader import (
    get_timestep_interval,
    list_dump_files,
)
from .io.result_writer import (
    save_erosion_results,
    save_no_erosion_flag,
)
from .plot import plot_erosion
from .utils import extraer_metadatos

# Prefijo estándar de las carpetas de simulación LAMMPS de erosión.
_SIM_PREFIX: str = "erosion_bed_SI_"

# Zona horaria de Argentina (UTC-3) para timestamps de los resultados.
_TZ_ARG = timezone(timedelta(hours=-3))


# ---------------------------------------------------------------------------
# Navegación interactiva de carpetas
# ---------------------------------------------------------------------------


def browse_for_simulation() -> list[Path]:
    """Muestra un menú interactivo para navegar el sistema de archivos y
    seleccionar la carpeta (o serie de carpetas) de simulación a procesar.

    La navegación es estrictamente hacia adelante: el usuario elige un
    subdirectorio por número o selecciona la carpeta actual. Para subir
    un nivel usa ``[-1]``. No hay comandos de salida — Ctrl+C cierra el
    programa si fuera necesario.

    Al seleccionar una carpeta (opción ``[0]``):

    - Si contiene subcarpetas ``erosion_bed_SI_*``, las devuelve todas
      ordenadas para procesamiento en lote.
    - Si es ella misma una carpeta ``erosion_bed_SI_*``, la devuelve sola.
    - En cualquier otro caso, devuelve la carpeta para que
      ``discover_simulations`` la evalúe.

    Returns:
        Lista de rutas absolutas a las carpetas de simulación seleccionadas.

    Example:
        >>> rutas = browse_for_simulation()
        --- Ubicación actual: /data/sims ---
        [0] Seleccionar esta carpeta
        [-1] Subir un nivel
        [1] batch_01
        [2] batch_02
        Elección: 1
    """
    current = Path(".").resolve()

    while True:
        subdirs = _list_subdirs(current)
        erosion_dirs = [p for p in subdirs if p.name.startswith(_SIM_PREFIX)]
        dump_files = list(current.glob("output.*.gz"))

        _print_browser_header(current, subdirs, erosion_dirs, dump_files)

        try:
            raw = input("Elección: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return []

        if raw == "0":
            return _select_current(current, subdirs)

        if raw == "-1":
            current = current.parent
            _clear()
            continue

        try:
            idx = int(raw)
        except ValueError:
            _clear()
            print("Entrada inválida — ingrese un número de la lista.\n")
            continue

        if 1 <= idx <= len(subdirs):
            current = subdirs[idx - 1]
            _clear()
        else:
            _clear()
            print("Número fuera de rango.\n")


# ---------------------------------------------------------------------------
# Descubrimiento de simulaciones
# ---------------------------------------------------------------------------


def discover_simulations(base_dir: Path) -> list[Path]:
    """Encuentra las carpetas de simulación a procesar desde un directorio base.

    Si ``base_dir`` ya es una carpeta de simulación (nombre que empieza con
    el prefijo estándar), devuelve una lista con ese único directorio.
    En caso contrario, busca subdirectorios con el prefijo y los devuelve
    ordenados alfabéticamente.

    Args:
        base_dir: Directorio raíz donde buscar. Puede ser directamente una
            carpeta de simulación o un contenedor de varias.

    Returns:
        Lista ordenada de rutas a las carpetas de simulación encontradas.

    Raises:
        SystemExit: Si no se encuentra ninguna simulación válida.
    """
    import sys

    if base_dir.name.startswith(_SIM_PREFIX):
        return [base_dir]

    sims = sorted(
        p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith(_SIM_PREFIX)
    )

    if not sims:
        print(
            f"Error: No se encontraron carpetas '{_SIM_PREFIX}*' en '{base_dir}'.\n"
            "Verifique que la ruta seleccionada sea correcta."
        )
        sys.exit(1)

    print(f"Encontradas {len(sims)} simulaciones en '{base_dir.name}'.")
    return sims


# ---------------------------------------------------------------------------
# Pipeline por simulación
# ---------------------------------------------------------------------------


def run_simulation(sim_dir: Path, config: Config) -> None:
    """Ejecuta el pipeline completo de análisis para una simulación individual.

    Flujo interno:

    1. Preparación: listar archivos dump, leer límites de caja, calcular
       altura de la cama.
    2. Decisión de onda: determinar automáticamente si aplicar Fase 1.
    3. Fase 1 (opcional): escaneo rápido de onda de choque.
    4. Fase 2: análisis topológico con OVITO sobre los frames relevantes.
    5. Exportación directa: guarda resultados Parquet y gráfico SVG sin
       verificar ni preguntar por resultados previos.

    Args:
        sim_dir: Directorio de la simulación, cuyo nombre debe comenzar con
            ``erosion_bed_SI_``.
        config: Configuración completa cargada desde ``config.toml``.
    """
    sim_id = sim_dir.name.removeprefix(_SIM_PREFIX)
    timestamp = datetime.now(_TZ_ARG).strftime("%Y-%m-%d_%H-%M-%S")

    print(f"\n{'=' * 55}")
    print(f"  Procesando: {sim_dir.name}")
    print(f"{'=' * 55}")

    # --- 1. Preparación de archivos ---
    dump_files = list_dump_files(sim_dir)
    if not dump_files:
        print(f"  Sin archivos output.*.gz en '{sim_dir.name}'. Saltando...")
        return

    timestep_interval = get_timestep_interval(dump_files)

    # --- 2. Límites de la caja (fijos desde config) ---
    box_limits = {"x": [0.0, config.caja_x], "y": [0.0, config.caja_y]}
    use_static_box = True  # Siempre usamos caja fija

    # --- 3. Extracción de metadatos y altura de la cama ---
    metadata = extraer_metadatos(sim_id)

    if (
        metadata.get("ff") == "N/A"
        or metadata.get("np") == "N/A"
        or metadata.get("vel0") == "N/A"
    ):
        print("  No se pudieron extraer metadatos del nombre de carpeta. Saltando...")
        return

    # Seleccionar altura de cama según valor de ff
    ff_value = metadata.get("ff", "N/A")
    if ff_value == "0.15":
        max_height = config.bed_height_ff_015
    elif ff_value == "0.35":
        max_height = config.bed_height_ff_035
    else:
        print(
            f"  Advertencia: ff={ff_value} no reconocido, usando ff_0.15 por defecto."
        )
        max_height = config.bed_height_ff_015

    # Agregar radios al umbral de altura para el análisis de erosión
    max_height = max_height + config.filters.umbral_techo_radios * config.radio_m

    # --- 4. Decisión automática de análisis de onda ---
    run_wave = is_critical_simulation(metadata, config)
    if run_wave:
        print(
            f"  [!] Simulación crítica — análisis de onda ACTIVADO\n"
            f"      np={metadata.get('np')}, "
            f"vel={metadata.get('vel0')} m/s, "
            f"pos=({metadata.get('posx')}, {metadata.get('posy')}, {metadata.get('posz')})"
        )

    # --- 5. Fase 1: Detección de onda (opcional) ---
    cutoff = config.radio_m * config.factor_cutoff
    pipeline = build_ovito_pipeline(cutoff=cutoff)

    if run_wave:
        _, cut_index = run_wave_scan(
            dump_files=dump_files,
            radius=config.radio_m,
            max_height=max_height,
            box_limits=box_limits,
            use_static_box=use_static_box,
            sim_id=sim_id,
            timestamp=timestamp,
            output_dir=config.dir_resultados,
            config=config,
        )
    else:
        cut_index = len(dump_files)

    # --- 6. Fase 2: Análisis topológico con OVITO ---
    files_fase2 = dump_files[:cut_index]

    results = run_topology_analysis(
        dump_files=files_fase2,
        pipeline=pipeline,
        max_height=max_height,
        sim_id=sim_id,
        timestep_interval=timestep_interval,
        tolerancia_v_cm=config.filters.tolerancia_v_cm,
    )

    # --- 7. Exportación directa de resultados (sin verificar previos) ---
    filter_thresholds = {
        "umbral_vz_min": config.filters.umbral_vz_min,
        "umbral_energia_min": config.filters.umbral_energia_min,
        "umbral_techo_radios": config.filters.umbral_techo_radios,
        "tolerancia_v_cm": config.filters.tolerancia_v_cm,
    }

    if not results:
        path = save_no_erosion_flag(sim_id, timestamp, config.dir_resultados)
        print(f"  Sin erosión significativa. Marcador guardado en: {path}")
        return

    result_path = save_erosion_results(
        results,
        sim_id,
        timestamp,
        config.dir_resultados,
        filter_thresholds,
    )
    print(f"\n  Resultados guardados en: {result_path}")

    # --- 7. Generación de gráfico ---
    try:
        df_total = pl.concat(results, how="vertical_relaxed")
        plot_erosion(df_total, sim_id, timestamp, config.dir_imagenes)
    except Exception as exc:  # noqa: BLE001
        print(f"  Advertencia: no se pudo generar el gráfico: {exc}")


# ---------------------------------------------------------------------------
# Bucle principal sobre múltiples simulaciones
# ---------------------------------------------------------------------------


def run_all(base_dir: Path, config: Config) -> None:
    """Procesa todas las simulaciones encontradas en el directorio base.

    Descubre las carpetas de simulación y ejecuta ``run_simulation`` en cada
    una directamente, sin verificar ni preguntar por resultados previos.

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
# Helpers privados del browser
# ---------------------------------------------------------------------------


def _list_subdirs(directory: Path) -> list[Path]:
    """Lista subdirectorios visibles (no ocultos) de un directorio, ordenados.

    Args:
        directory: Directorio a listar.

    Returns:
        Lista ordenada de rutas a los subdirectorios.
    """
    try:
        return sorted(
            p for p in directory.iterdir() if p.is_dir() and not p.name.startswith(".")
        )
    except PermissionError:
        print(f"  Sin permiso para leer '{directory}'.")
        return []


def _print_browser_header(
    current: Path,
    subdirs: list[Path],
    erosion_dirs: list[Path],
    dump_files: list[Path],
) -> None:
    """Imprime la cabecera del menú de navegación con el estado actual.

    Args:
        current: Directorio actualmente visible.
        subdirs: Lista de subdirectorios disponibles.
        erosion_dirs: Subcarpetas con prefijo ``erosion_bed_SI_``.
        dump_files: Archivos ``output.*.gz`` encontrados en ``current``.
    """
    print(f"\n--- Ubicación actual: {current} ---")

    if dump_files:
        print(f"  ✓ {len(dump_files)} archivos output.*.gz (datos de simulación)")
    if erosion_dirs:
        print(f"  ✓ {len(erosion_dirs)} carpeta(s) 'erosion_bed_SI_*'")

    print()
    print("[0]  Seleccionar esta carpeta")
    print("[-1] Subir un nivel")
    print()

    for i, subdir in enumerate(subdirs, start=1):
        tag = " ★" if subdir.name.startswith(_SIM_PREFIX) else ""
        print(f"[{i}] {subdir.name}{tag}")

    print()


def _select_current(current: Path, subdirs: list[Path]) -> list[Path]:
    """Procesa la selección de la carpeta actual por el usuario.

    Si la carpeta actual contiene subdirectorios ``erosion_bed_SI_*``,
    los devuelve todos para procesamiento en lote. Si es ella misma una
    simulación, la devuelve sola. En caso contrario devuelve la carpeta
    actual para que ``discover_simulations`` la evalúe.

    Args:
        current: Carpeta seleccionada por el usuario.
        subdirs: Subdirectorios de ``current`` ya listados.

    Returns:
        Lista de rutas a procesar.
    """
    erosion_subs = [p for p in subdirs if p.name.startswith(_SIM_PREFIX)]

    if erosion_subs:
        print(
            f"\n→ Se procesarán {len(erosion_subs)} subcarpetas "
            f"'erosion_bed_SI_*' en '{current.name}'."
        )
        return sorted(erosion_subs)

    return [current]


def _clear() -> None:
    """Limpia la terminal de forma multiplataforma."""
    import os

    os.system("cls" if os.name == "nt" else "clear")
