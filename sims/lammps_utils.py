"""
Utilidades para la lectura y extracción de datos de simulaciones LAMMPS.

Proporciona funciones para leer archivos comprimidos de LAMMPS, extraer
límites de la caja de simulación, y calcular parámetros físicos como
la altura máxima de la cama de partículas.
"""

import gzip
from pathlib import Path

import polars as pl

from utils import limpiar_pantalla


def extractor_de_borde(
    ruta: Path,
    nombre_archivo: str,
    columnas: dict[str, int] = None,
) -> dict[str, list[float]]:
    """
    Extrae los límites espaciales de la caja de simulación a partir de un archivo
    de salida comprimido.

    Args:
        ruta (Path): Ruta absoluta o relativa al directorio que contiene los datos
                     de la simulación.
        nombre_archivo (str): Nombre del archivo comprimido (ej. 'output.0.gz').
        columnas (dict[str, int] | None): Diccionario que mapea los nombres de los
                                          ejes a la línea específica del encabezado
                                          donde se encuentran sus límites.
                                          Por defecto: {"x": 5, "y": 6}.

    Returns:
        dict[str, list[float]]: Diccionario donde las llaves son los ejes analizados
                                y los valores son listas conteniendo los límites
                                mínimos y máximos de la caja en dicho eje.
    """
    if columnas is None:
        columnas = {"x": 5, "y": 6}

    resultados: dict[str, list[float]] = {}
    # Inicializa las listas vacías adaptándose a los ejes solicitados en 'columnas'
    for llave in columnas.keys():
        resultados[llave] = []

    # Abre y lee el archivo comprimido en modo texto
    ruta_archivo = ruta / nombre_archivo
    with gzip.open(ruta_archivo, "rt") as f:
        # Inversión del diccionario para acceder rápidamente a la llave por el
        # número de línea
        indice_a_llave = {v: k for k, v in columnas.items()}

        # Iteración línea por línea para no cargar todo el archivo en memoria
        for i, line in enumerate(f):
            # Verifica si la línea actual corresponde a uno de los límites buscados
            if i in indice_a_llave:
                llave_actual = indice_a_llave[i]

                # Limpia la línea, separa los valores por espacio y los convierte
                # a flotantes
                datos = [float(number) for number in line.replace("\n", "").split(" ")]

                # Almacena los límites [mínimo, máximo] encontrados
                resultados[llave_actual].extend(datos)

    return resultados


def lector(archivo_gz: Path, datos: list[str]) -> pl.LazyFrame:
    """
    Lee archivos de volcado (dump) comprimidos de LAMMPS optimizando el uso de
    memoria.

    Utiliza evaluación perezosa (LazyFrame) de Polars para escanear el archivo,
    omitiendo encabezados innecesarios y filtrando únicamente las columnas
    requeridas.

    Args:
        archivo_gz (Path): Ruta completa al archivo comprimido a procesar.
        datos (list[str]): Lista de cadenas con los nombres de las columnas que
                           se desean extraer.

    Returns:
        pl.LazyFrame: Estructura de datos perezosa de Polars lista para ser
                      procesada.
    """
    nombres_columnas = ["id", "type", "x", "y", "z", "vx", "vy", "vz"]

    lf_archivo = pl.scan_csv(
        archivo_gz,
        skip_rows=9,
        has_header=False,
        separator=" ",
        truncate_ragged_lines=True,
        new_columns=nombres_columnas,
        infer_schema_length=10000,
        schema_overrides={"type": pl.UInt8},
    )
    return lf_archivo.select(datos)


def max_altura(ruta: Path, R: float, umbral_techo: int = 5) -> float:
    """
    Determina la altura máxima basal de la cama de partículas en el estado
    inicial de la simulación.

    Args:
        ruta (Path): Directorio donde se encuentra el archivo inicial
                     'output.0.gz'.
        R (float): Radio físico de la partícula en metros [m].
        umbral_techo (int, opcional): Margen de seguridad añadido, expresado en
                                      número de radios. Por defecto es 5.

    Returns:
        float: Altura máxima calculada (límite superior de la cama) para usar
               como umbral de filtrado.

    Raises:
        ValueError: Si no se encuentran partículas de tipo 1 en la simulación.
    """
    ruta_archivo = ruta / "output.0.gz"
    lf0 = lector(ruta_archivo, ["type", "z"])

    # Localiza la partícula tipo 1 (asumida como material del lecho) con mayor
    # cota Z
    max_z_type1 = (
        lf0.filter(pl.col("type") == 1).select(pl.col("z").max()).collect().item()
    )

    if max_z_type1 is None:
        raise ValueError(
            "Error en la extracción: La altura máxima de la cama no puede ser "
            "nula (None). Verifique las IDs de las partículas."
        )

    # Añade el margen de tolerancia a la altura absoluta
    altura = umbral_techo * R + max_z_type1
    return altura


def control_caja(
    limites_i: dict[str, list[float]],
    limites_f: dict[str, list[float]],
    archivos: list[str],
) -> bool:
    """
    Evalúa la consistencia volumétrica de la caja de simulación entre el estado
    inicial y final.

    Si detecta variaciones en los límites espaciales (por ejemplo, debido a
    condiciones de contorno dinámicas o ensambles NPT), interrumpe el flujo
    para consultar al usuario si desea mantener la caja estática original como
    volumen de control o utilizar bordes dinámicos.

    Args:
        limites_i (dict[str, list[float]]): Diccionario con los límites espaciales
                                            iniciales de la caja.
        limites_f (dict[str, list[float]]): Diccionario con los límites espaciales
                                            finales de la caja.
        archivos (list[str]): Lista de nombres de los archivos de volcado, utilizada
                              para indicarle al usuario entre qué iteraciones ocurrió
                              el cambio.

    Returns:
        bool: True si se decide utilizar los bordes fijos de la caja para el cálculo.
              False en caso contrario.
    """
    # Gestión de divergencia en los límites del volumen de control
    # (ej. condiciones NPT)
    if limites_f != limites_i:
        while True:
            seguir = input(
                f"""Los bordes de las caja cambiaron entre {archivos[0]} y {archivos[-1]}
En {archivos[0]} son:
{limites_i}
En {archivos[-1]} son:
{limites_f}
¿Deseás aun así usar los bordes de la caja para el cálculo? (Y/N): """
            )
            if seguir in "Nn":
                return False

            elif seguir in "Yy":
                return True
            else:
                limpiar_pantalla()
                print("(No se registró ninguna entrada válida. Por favor, usar Y o N)")
    else:
        return True
