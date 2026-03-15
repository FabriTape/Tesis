"""
Utilidades compartidas para el análisis de simulaciones LAMMPS.

Módulo de funciones auxiliares puras (sin efectos de I/O ni interacción
con el usuario) utilizadas transversalmente en el pipeline de análisis.
Incluye extracción de metadatos desde nombres de carpeta y visualización
de progreso en terminal.
"""

import re

# Expresiones regulares para extraer parámetros del nombre de carpeta.
# Formato esperado: erosion_bed_SI_ff_<val>_np_<val>_vel_<val>_angle_<val>_move_<x>_<y>_<z>
_PATRONES: dict[str, str] = {
    "ff": r"ff_([^_]+)",
    "np": r"np_([^_]+)",
    "vel0": r"vel_([^_]+)",
    "angulo": r"angle_([^_]+)",
    "move": r"move_([^_]+)_([^_]+)_([^_]+)",
}


def extraer_metadatos(sim_id: str) -> dict[str, str]:
    """Extrae parámetros físicos de la simulación desde el identificador de carpeta.

    Utiliza expresiones regulares para parsear el nombre de la carpeta de
    simulación y devuelve un diccionario con los parámetros físicos relevantes.
    Si un parámetro no se encuentra, se asigna "N/A" (o "0" para coordenadas).

    Args:
        sim_id: Identificador de la simulación, correspondiente al nombre de
            la carpeta sin el prefijo ``erosion_bed_SI_``. Ejemplo:
            ``"ff_0.15_np_1_vel_-50.0_angle_0_move_0.1_0_0"``.

    Returns:
        Diccionario con las claves ``"ff"``, ``"np"``, ``"vel0"``,
        ``"angulo"``, ``"posx"``, ``"posy"``, ``"posz"``.
        Todos los valores son cadenas de texto.

    Example:
        >>> meta = extraer_metadatos("ff_0.15_np_1_vel_-50.0_angle_0_move_0.1_0_0")
        >>> meta["ff"]
        '0.15'
        >>> meta["posx"]
        '0.1'
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

    Imprime una barra de progreso porcentual que se sobreescribe en la misma
    línea usando el caracter de retorno de carro ``\\r``. Diseñada para ser
    llamada en cada iteración de un bucle de procesamiento.

    Args:
        current: Índice de la iteración actual (base 0).
        total: Número total de iteraciones esperadas.
        label: Etiqueta descriptiva del proceso en curso. Por defecto
            ``"análisis"``.

    Example:
        >>> for i, archivo in enumerate(archivos):
        ...     print_progress(i, len(archivos), label="onda")
        ...     # procesar archivo
        >>> print_progress(100, 100, label="onda")  # finalizar
        >>> print()
    """
    if total == 0:
        return

    pct = int(current * 100 / total)
    filled = pct // 5
    bar = "_" * filled + "·" * (20 - filled)

    if current == 0:
        print(f"Iniciando análisis de {label}...")

    print(f"\r{pct:03d}/100% [{bar}] ({label})", end="", flush=True)
