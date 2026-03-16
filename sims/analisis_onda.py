"""
Módulo de detección de ondas de choque en simulaciones LAMMPS.

Proporciona funciones para identificar y filtrar las partículas que componen
la onda de choque generada en la cama de partículas, utilizando criterios
cinéticos y espaciales.
"""

from pathlib import Path

import polars as pl

# ==========================================
# PARÁMETROS FÍSICOS Y UMBRALES DE ANÁLISIS
# ==========================================
# Energía cinética mínima para considerar una partícula activa
UMBRAL_V2_MIN = 4e-4  # [m²/s²]
# Energía cinética máxima para considerar un impacto crítico
UMBRAL_V2_MAX = 4e-2  # [m²/s²]
# Cantidad de partículas en movimiento para detectar onda macroscópica
PARTICULAS_CRITICAS = 3000


def detector_de_ondas(
    lf: pl.LazyFrame,
    R: float,
    altura: float,
    limites: dict[str, list[float]],
    usar_caja: bool,
    umbral_techo: int = 3,
    umbral_pared: int = 20,
) -> tuple[pl.LazyFrame, pl.DataFrame]:
    """
    Identifica y filtra las partículas que componen la onda de choque generada en la cama.

    Evalúa parámetros cinéticos (velocidad cuadrática) y espaciales (altura y proximidad
    a fronteras) para discriminar la onda del resto del sistema.

    Args:
        lf (pl.LazyFrame): Conjunto de datos de las partículas.
        R (float): Radio de las partículas en el sistema.
        altura (float): Altura de referencia (máxima) de la cama de partículas en reposo.
        limites (dict[str, list[float]]): Diccionario con los límites espaciales estáticos
                                          de la caja.
        usar_caja (bool): Determina si se evalúan los bordes respecto a 'limites' (True)
                          o respecto a los extremos móviles de las partículas (False).
        umbral_techo (int, opcional): Número de radios de tolerancia respecto a la altura.
                                      Por defecto es 3.
        umbral_pared (int, opcional): Número de radios de proximidad requerida a las
                                      paredes. Por defecto es 20.

    Returns:
        tuple[pl.LazyFrame, pl.DataFrame]:
            - pl.LazyFrame: Subconjunto de partículas clasificadas como parte de la onda.
            - pl.DataFrame: Resumen métrico (cantidad de partículas involucradas y su
                            velocidad máxima).
    """
    # Determinamos los límites de las paredes (estáticos de la caja o dinámicos)
    x_max = limites["x"][1] if usar_caja else pl.col("x").max()
    y_max = limites["y"][1] if usar_caja else pl.col("y").max()

    lf_onda = lf.with_columns(
        # Cálculo de la magnitud de la velocidad al cuadrado para criterio cinético
        (pl.col("vx") ** 2 + pl.col("vy") ** 2 + pl.col("vz") ** 2).alias("v^2")
    ).filter(
        # 1. Filtro vertical: Descartar partículas eyectadas muy por encima de la cama
        (pl.col("z") < altura - umbral_techo * R)
        # 2. Filtro cinético: Excluir partículas estáticas o con energía despreciable
        & (pl.col("v^2") > UMBRAL_V2_MIN)
        & (
            # 3. Filtro perimetral: Asegurar que el evento alcance las fronteras
            (pl.col("x") < umbral_pared * R)
            | (x_max - umbral_pared * R < pl.col("x"))
            | (pl.col("y") < umbral_pared * R)
            | (y_max - umbral_pared * R < pl.col("y"))
        )
    )

    # Recopilación materializada de las métricas principales de la onda detectada
    df_res_onda = lf_onda.select(
        [pl.len().alias("cantidad"), pl.col("v^2").max().alias("v^2_max")]
    ).collect()

    return lf_onda, df_res_onda


def cortar_analisis_onda(
    cantidad: int,
    bool_v2_max: bool,
    max_v2: float | None,
    i: int,
    file: str,
    datos: str,
    fecha_hora: str,
    lf_onda: pl.LazyFrame,
    dir_resultados: Path,
) -> bool:
    """
    Evalúa si se alcanzó el criterio físico crítico para considerar la propagación
    de una onda masiva.

    Si se cumple la condición de corte (por cantidad de partículas involucradas o por
    energía cinética máxima), imprime una alerta, serializa los datos de la onda a disco
    y devuelve una señal para interrumpir el escaneo secuencial (Fail Fast).

    Args:
        cantidad (int): Número de partículas detectadas como parte de la onda en el
                        timestep actual.
        bool_v2_max (bool): Bandera que indica si alguna partícula superó el umbral
                            crítico de velocidad.
        max_v2 (float | None): Valor absoluto de la velocidad cuadrática máxima registrada.
        i (int): Índice numérico de la iteración actual.
        file (str): Nombre del archivo de volcado actual (ej. 'output.80.gz').
        datos (str): Identificador base de los parámetros de la simulación.
        fecha_hora (str): Marca temporal para generar nombres de archivo únicos.
        lf_onda (pl.LazyFrame): Estructura de datos perezosa que contiene las partículas
                                de la onda.
        dir_resultados (Path): Directorio donde se guardarán los resultados.

    Returns:
        bool: True si se detectó la onda y se guardó el archivo (señal de corte).
              False si el evento aún no alcanza la magnitud crítica.
    """
    if cantidad > PARTICULAS_CRITICAS or bool_v2_max:
        print(
            f"""\nSe analizaron {i} archivos y en el archivo {file}:
            Se detectó una onda masiva de {cantidad} partículas
            Con una v² máxima absoluta de {max_v2}"""
        )

        # Generación dinámica de la ruta de salida usando Path
        ruta_guardado = dir_resultados / f"{datos}_ONDA_{i}_{fecha_hora}.parquet"
        lf_onda.sink_parquet(ruta_guardado)

        return True
    else:
        return False


def ejecutar_fase_onda(
    archivos: list[str],
    ruta: Path,
    columnas: list[str],
    R: float,
    altura: float,
    limites_f: dict[str, list[float]],
    usar_caja: bool,
    datos: str,
    fecha_hora: str,
) -> tuple[bool, int]:
    """
    Ejecuta la primera fase del análisis: búsqueda rápida de la onda de choque.

    Itera secuencialmente sobre los archivos de volcado (dumps) para detectar el
    momento crítico en el que se propaga una onda macroscópica en la cama de partículas.
    Aplica criterios cinéticos (velocidad umbral) y espaciales. Si el evento alcanza
    una magnitud crítica, guarda los datos de la onda e interrumpe el escaneo secuencial.

    Args:
        archivos (list[str]): Nombres de los archivos '.gz' a procesar, ordenados
                              cronológicamente.
        ruta (Path): Directorio raíz donde se alojan los volcados de la simulación.
        columnas (list[str]): Nombres de las columnas que se extraerán al leer los
                              archivos.
        R (float): Radio físico de los granos expresado en metros [m].
        altura (float): Altura de la cama basal en reposo, usada como nivel de referencia.
        limites_f (dict[str, list[float]]): Límites espaciales de la caja en el último
                                            frame de la simulación.
        usar_caja (bool): Flag que determina si las fronteras de análisis son estáticas
                          (True) o dinámicas respecto a la posición de las partículas
                          (False).
        datos (str): Identificador de parámetros (ej. prefijo del nombre de la carpeta).
        fecha_hora (str): Marca temporal (timestamp) utilizada para versionar los archivos
                          de salida.

    Returns:
        tuple[bool, int]:
            - onda_detectada (bool): True si se registró el impacto crítico, False en
                                     caso contrario.
            - paso_corte (int): Índice numérico de la iteración donde se detuvo el
                                análisis. Si no hay onda, equivale a la longitud total
                                de 'archivos'.
    """
    # Importamos acá para evitar dependencias circulares
    from lammps_utils import lector
    from utils import DIR_RESULTADOS, limpiar_pantalla, progress

    onda_detectada = False
    paso_corte = len(archivos)
    # Solo necesitamos estas columnas para detectar la onda (x, y, z, vx, vy, vz)
    cols_necesarias = ["x", "y", "z", "vx", "vy", "vz"]

    for i, file in enumerate(archivos):
        progress(i, len(archivos), parte=1)

        ruta_archivo = ruta / file
        archivo_lf = lector(ruta_archivo, cols_necesarias)

        lf_onda, df_res_onda = detector_de_ondas(
            archivo_lf, R, altura, limites_f, usar_caja
        )

        cantidad = df_res_onda.item(0, "cantidad")
        max_v2 = df_res_onda.item(0, "v^2_max")
        bool_v2_max = (max_v2 >= UMBRAL_V2_MAX) if isinstance(max_v2, float) else False

        # Criterio de impacto crítico (Trigger de la onda)
        if cortar_analisis_onda(
            cantidad,
            bool_v2_max,
            max_v2,
            i,
            file,
            datos,
            fecha_hora,
            lf_onda,
            DIR_RESULTADOS,
        ):
            paso_corte = i
            onda_detectada = True
            break

    progress(100, 100, parte=1)
    print("")
    limpiar_pantalla()

    return onda_detectada, paso_corte
