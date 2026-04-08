"""
Carga de configuración y descubrimiento de simulaciones LAMMPS.

Este módulo combina:
- Carga de parámetros desde config.toml
- Descubrimiento de carpetas de simulación
- Extracción de metadatos desde nombres de carpeta
- Navegación interactiva de carpetas (opcional)

Física: Los parámetros definen umbrales para detección de onda de choque,
criterios de erosión de la cama, propiedades del material sílice y
configuración del modelo JKR.
"""

from __future__ import annotations

import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

_SIM_PREFIX: str = "sims_SI_"

_PATRONES: dict[str, str] = {
    "ff": r"ff_([^_]+)",
    "np": r"np_([^_]+)",
    "vel0": r"vel_([^_]+)",
    "angulo": r"angle_([^_]+)",
    "move": r"move_([^_]+)_([^_]+)_([^_]+)",
}


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
        umbral_techo_radios: Margen de seguridad vertical en número de radios.
        tolerancia_v_cm: Tolerancia de |Δv_cm| entre dumps consecutivos [m/s].
        filtro_activo: Nombre del filtro de erosión (F3C).
    """

    umbral_vz_min: float
    umbral_techo_radios: float
    tolerancia_v_cm: float
    filtro_activo: str = "F3C"


@dataclass(frozen=True)
class Config:
    """Configuración completa del programa CARBOXI.

    Attributes:
        radio_m: Radio de las partículas en metros [m].
        usar_caja: Si True, usa los límites fijos de la caja.
        bed_height_ff_015: Altura de la cama para ff=0.15 [m].
        bed_height_ff_035: Altura de la cama para ff=0.35 [m].
        caja_x: Dimensión de la caja en X [m].
        caja_y: Dimensión de la caja en Y [m].
        freeze_height: Altura de congelamiento para filtro F3C [m].
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
    freeze_height: float
    wave: WaveConfig
    filters: FilterConfig
    factor_cutoff: float
    dir_resultados: Path
    dir_imagenes: Path


def load_config(path: Path = Path("config.toml")) -> Config:
    """Carga y parsea el archivo de configuración TOML embebido.

    Args:
        path: Ruta al archivo config.toml (ignorada, se usa el embebido).

    Returns:
        Config: Instancia inmutable con todos los parámetros de análisis.

    Raises:
        FileNotFoundError: Si el config.toml embebido no se encuentra.
        KeyError: Si faltan secciones obligatorias en el TOML.
    """
    resolved = _resolve_config_path(path)

    with resolved.open("rb") as f:
        raw: dict = tomllib.load(f)

    return _build_config(raw)


def _resolve_config_path(path: Path) -> Path:
    """Resuelve la ruta del archivo config.toml usando siempre el embebido."""
    module_config = Path(__file__).parent / "config.toml"
    if module_config.exists():
        return module_config

    raise FileNotFoundError(f"config.toml embebido no encontrado en '{module_config}'.")


def _build_config(raw: dict) -> Config:
    """Construye la instancia Config a partir del diccionario TOML."""
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
        umbral_techo_radios=float(filtros_raw["umbral_techo_radios"]),
        tolerancia_v_cm=float(filtros_raw.get("tolerancia_v_cm", 0.0)),
        filtro_activo=str(filtros_raw.get("filtro_activo", "F3C")),
    )

    rutas = raw["rutas"]
    return Config(
        radio_m=radio_um * 1e-6,
        usar_caja=bool(raw["defaults"].get("usar_caja", True)),
        bed_height_ff_015=float(raw["defaults"]["bed_height_ff_015"]),
        bed_height_ff_035=float(raw["defaults"]["bed_height_ff_035"]),
        caja_x=float(raw["defaults"]["caja_x"]),
        caja_y=float(raw["defaults"]["caja_y"]),
        freeze_height=float(raw["defaults"]["freeze_height"]),
        wave=wave,
        filters=filters,
        factor_cutoff=float(raw["jkr"]["factor_cutoff"]),
        dir_resultados=Path(rutas["resultados"]),
        dir_imagenes=Path(rutas["imagenes"]),
    )


def discover_simulations(
    base_dir: Path, prefix: str = _SIM_PREFIX, type: str = "dir"
) -> list[Path]:
    """Encuentra las carpetas de simulación a procesar desde un directorio base.

    Args:
        base_dir: Directorio raíz donde buscar.
        prefix: Prefijo de las carpetas de simulación a buscar.
        type: Tipo de simulación a buscar ("dir" para carpetas, "file" para archivos).

    Returns:
        Lista ordenada de rutas a las carpetas de simulación encontradas.

    Raises:
        SystemExit: Si no se encuentra ninguna simulación válida.
    """
    if base_dir.name.startswith(prefix):
        return [base_dir]
    if type == "dir":
        sims = sorted(
            p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith(prefix)
        )
    else:
        sims = sorted(
            p for p in base_dir.iterdir() if p.is_file() and p.name.startswith(prefix)
        )

    if not sims:
        print(
            f"Error: No se encontraron {'carpetas' if type == 'dir' else 'archivos'} '{prefix}*' en '{base_dir}'.\n"
            "Verifique que la ruta seleccionada sea correcta."
        )
        sys.exit(1)

    print(f"Encontradas {len(sims)} simulaciones en '{base_dir.name}'.")
    return sims


def extraer_metadatos(sim_id: str) -> dict[str, str]:
    """Extrae parámetros físicos de la simulación desde el identificador de carpeta.

    Args:
        sim_id: Identificador de la simulación (sin prefijo erosion_bed_SI_).

    Returns:
        Diccionario con claves: ff, np, vel0, angulo, posx, posy, posz.
    """
    metadatos: dict[str, str] = {}

    for clave, patron in _PATRONES.items():
        match = re.search(patron, sim_id)
        if match:
            if clave == "move":
                metadatos["posx"] = match.group(1)
                metadatos["posy"] = match.group(2)
                metadatos["posz"] = match.group(3)
            else:
                metadatos[clave] = match.group(1)
        else:
            print(f"Advertencia: parámetro '{clave}' no encontrado en '{sim_id}'.")
            if clave == "move":
                metadatos.update({"posx": "0", "posy": "0", "posz": "0"})
            else:
                metadatos[clave] = "N/A"

    return metadatos


def print_progress(current: int, total: int, label: str = "análisis") -> None:
    """Actualiza en línea una barra de progreso ASCII en la terminal.

    Args:
        current: Índice de la iteración actual (base 0).
        total: Número total de iteraciones esperadas.
        label: Etiqueta descriptiva del proceso en curso.
    """
    if total == 0:
        return

    pct = int(current * 100 / total)
    filled = pct // 5
    bar = "_" * filled + "@ï" + "·" * (18 - filled)

    if current == 0:
        print(f"Iniciando análisis de {label}...")

    print(f"\r{pct:03d}/100% [{bar}] ({label})", end="", flush=True)


def browse_for_simulation() -> list[Path]:
    """Muestra un menú interactivo para navegar el sistema de archivos y
    seleccionar la carpeta de simulación a procesar.

    Returns:
        Lista de rutas absolutas a las carpetas de simulación seleccionadas.
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


def _list_subdirs(directory: Path) -> list[Path]:
    """Lista subdirectorios visibles (no ocultos) de un directorio."""
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
    """Imprime la cabecera del menú de navegación."""
    print(f"\n--- Ubicación actual: {current} ---")

    if dump_files:
        print(f"  ✓ {len(dump_files)} archivos output.*.gz")
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
    """Procesa la selección de la carpeta actual por el usuario."""
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
