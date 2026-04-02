"""Navegación interactiva de carpetas para seleccionar simulaciones."""

from pathlib import Path

_SIM_PREFIX: str = "erosion_bed_SI_"


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
