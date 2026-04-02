"""
Utilidades compartidas para el análisis de simulaciones LAMMPS.

Funciones auxiliares puras (sin efectos de I/O) usadas transversalmente
en el pipeline: extracción de metadatos desde nombres de carpeta y
visualización de progreso en terminal.
"""

import re

_PATRONES: dict[str, str] = {
    "ff": r"ff_([^_]+)",
    "np": r"np_([^_]+)",
    "vel0": r"vel_([^_]+)",
    "angulo": r"angle_([^_]+)",
    "move": r"move_([^_]+)_([^_]+)_([^_]+)",
}


def extraer_metadatos(sim_id: str) -> dict[str, str]:
    """Extrae parámetros físicos desde el identificador de carpeta.

    Parsea nombres con formato
    ``erosion_bed_SI_ff_<val>_np_<val>_vel_<val>_angle_<val>_move_<x>_<y>_<z>``.
    Si un parámetro no se encuentra, asigna ``"N/A"`` (o ``"0"`` para
    coordenadas).

    Args:
        sim_id: Nombre de carpeta (con o sin prefijo ``erosion_bed_SI_``).

    Returns:
        Dict con claves ``"ff"``, ``"np"``, ``"vel0"``, ``"angulo"``,
        ``"posx"``, ``"posy"``, ``"posz"``. Valores como strings.

    Example:
        >>> meta = extraer_metadatos("ff_0.15_np_1_vel_-50.0_angle_0_move_0.1_0_0")
        >>> meta["ff"]
        '0.15'
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
            if clave == "move":
                metadatos.update({"posx": "0", "posy": "0", "posz": "0"})
            else:
                metadatos[clave] = "N/A"

    return metadatos


def print_progress(current: int, total: int, label: str = "análisis") -> None:
    """Barra de progreso ASCII que se actualiza en la misma línea.

    Args:
        current: Iteración actual (base 0).
        total: Total de iteraciones.
        label: Etiqueta del proceso.
    """
    if total == 0:
        return

    pct = int(current * 100 / total)
    filled = pct // 5
    bar = "_" * filled + "·" * (20 - filled)

    if current == 0:
        print(f"Iniciando análisis de {label}...")

    print(f"\r{pct:03d}/100% [{bar}] ({label})", end="", flush=True)
