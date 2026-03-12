"""
Utilidades para el análisis topológico utilizando OVITO.

Este módulo proporciona funciones para inicializar y ejecutar el pipeline
de OVITO, calculando coordinación y clusters de partículas erosionadas.
"""

from pathlib import Path

import polars as pl
from numpy import empty as npempty
from ovito.data import DataCollection, SimulationCell
from ovito.modifiers import ClusterAnalysisModifier, CoordinationAnalysisModifier
from ovito.pipeline import Pipeline, StaticSource

from lammps_utils import lector
from utils import extraer_metadatos, progress


def inicializar_pipeline_ovito(cutoff: float) -> Pipeline:
    """
    Inicializa la estructura estática de OVITO una sola vez para ahorrar memoria.

    Configura un pipeline con modificadores de análisis de coordinación y clusters,
    utilizando el radio de las partículas como distancia de corte.

    Args:
        cutoff (float): Distancia de corte para el análisis de coordinación y clusters,
                        generalmente el radio de las partículas.

    Returns:
        Pipeline: Pipeline de OVITO configurado y listo para usar.
    """
    data = DataCollection()
    particles = data.create_particles()

    # Creamos la propiedad con un array vacío como punto de partida
    particles.create_property("Position", data=npempty((0, 3)))

    cell = SimulationCell()
    cell[:, :] = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
    data.objects.append(cell)

    pipeline = Pipeline(source=StaticSource(data=data))
    pipeline.modifiers.append(CoordinationAnalysisModifier(cutoff=cutoff))
    pipeline.modifiers.append(ClusterAnalysisModifier(cutoff=cutoff))

    return pipeline


def analisis_ovito(lf: pl.LazyFrame, pipeline: Pipeline, altura: float) -> pl.DataFrame:
    """
    Analiza las partículas erosionadas utilizando OVITO para calcular coordinación y clusters.

    Filtra las partículas que fueron erosionadas (z > altura AND vz > 0),
    calcula coordinación y clusters para ellas, y preserva la información
    del proyectil (tipo 2) indicando si fue erosionado o no.

    Args:
        lf (pl.LazyFrame): Datos posicionales y de velocidad de las partículas.
        pipeline (Pipeline): Flujo de procesamiento de OVITO ya inicializado.
        altura (float): Umbral vertical de la cama para filtrar partículas.

    Returns:
        pl.DataFrame: DataFrame con las propiedades calculadas 'coordinacion' y 'cluster',
                      además de la columna 'proyectil_erosion' que indica si el proyectil
                      fue erosionado.
    """
    # Identificar si una partícula fue erosionada
    lf = lf.with_columns(
        ((pl.col("z") > altura) & (pl.col("vz") > 0)).alias("es_erosionada")
    )

    # El DataFrame final debe contener todas las partículas erosionadas + el proyectil (tipo 2)
    df_interes = lf.filter((pl.col("es_erosionada")) | (pl.col("type") == 2)).collect()

    if df_interes.shape[0] == 0:
        return df_interes.drop("es_erosionada").with_columns(
            [
                pl.Series("coordinacion", [], dtype=pl.Float64),
                pl.Series("cluster", [], dtype=pl.UInt32),
                pl.Series("proyectil_erosion", [], dtype=pl.Boolean),
            ]
        )

    # 1. Separar las erosionadas para pasar a OVITO (no pasamos el proyectil si NO se erosionó, o lo pasamos si se erosionó)
    # Si queremos que OVITO analice el proyectil que se erosionó, filtramos por 'es_erosionada'
    df_erosionadas = df_interes.filter(pl.col("es_erosionada"))

    # Si hay partículas erosionadas, calculamos cluster/coordinacion con OVITO
    if df_erosionadas.shape[0] > 0:
        positions = df_erosionadas.select(["x", "y", "z"]).to_numpy()

        nuevo_data = DataCollection()
        particles = nuevo_data.create_particles(count=positions.shape[0])
        particles.create_property("Position", data=positions)

        cell = SimulationCell()
        cell[:, :] = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
        nuevo_data.objects.append(cell)

        pipeline.source = StaticSource(data=nuevo_data)
        computed_data = pipeline.compute()

        coordinacion = computed_data.particles["Coordination"][:]
        cluster = computed_data.particles["Cluster"][:]

        # Agregar al dataframe de erosionadas
        df_erosionadas = df_erosionadas.with_columns(
            [
                pl.Series("coordinacion", coordinacion, dtype=pl.Float64),
                pl.Series("cluster", cluster, dtype=pl.UInt32),
            ]
        )
    else:
        df_erosionadas = df_erosionadas.with_columns(
            [
                pl.Series("coordinacion", [], dtype=pl.Float64),
                pl.Series("cluster", [], dtype=pl.UInt32),
            ]
        )

    # 2. Manejar las que NO se erosionaron (que solo pueden ser el proyectil, por el filtro previo)
    df_no_erosionadas = df_interes.filter(~pl.col("es_erosionada"))
    df_no_erosionadas = df_no_erosionadas.with_columns(
        [
            pl.lit(None).alias("coordinacion").cast(pl.Float64),
            pl.lit(None).alias("cluster").cast(pl.UInt32),
        ]
    )

    # 3. Unir todo nuevamente
    df_final = pl.concat([df_erosionadas, df_no_erosionadas], how="vertical_relaxed")

    # 4. Asignar la columna 'proyectil_erosion'
    # Será True si type == 2 y es_erosionada == True
    # Será False si type == 2 y es_erosionada == False
    # Será Null/NaN si type != 2
    df_final = df_final.with_columns(
        pl.when(pl.col("type") == 2)
        .then(pl.col("es_erosionada"))
        .otherwise(pl.lit(None))
        .alias("proyectil_erosion")
    ).drop("es_erosionada")

    return df_final


def ejecutar_fase_ovito(
    archivos_fase2: list[str],
    ruta: Path,
    columnas: list[str],
    pipeline_global: Pipeline,
    altura: float,
    datos: str,
    paso_int: int,
    onda_detect_bool: bool,
) -> list[pl.DataFrame]:
    """
    Ejecuta la Fase 2 del análisis topológico pesado utilizando OVITO.

    Procesa secuencialmente cada archivo de volcado, aplicando el criterio de erosión
    y calculando las propiedades topológicas (coordinación y clusters) para las
    partículas erosionadas.

    Args:
        archivos_fase2 (list[str]): Lista de nombres de archivos a procesar en esta fase.
        ruta (Path): Directorio raíz donde se alojan los volcados de la simulación.
        columnas (list[str]): Nombres de las columnas que se extraerán al leer los archivos.
        pipeline_global (Pipeline): Pipeline de OVITO inicializado para reutilizar.
        altura (float): Umbral vertical de la cama para filtrar partículas.
        datos (str): Identificador de parámetros extraído del nombre de la carpeta.
        paso_int (int): Frecuencia de volcado de datos (timestep) entre archivos.
        onda_detect_bool (bool): Flag que indica si se detectó una onda en la Fase 1.

    Returns:
        list[pl.DataFrame]: Lista de DataFrames procesados para cada archivo.
    """
    lista_resultados: list[pl.DataFrame] = []
    LenFase2 = len(archivos_fase2)
    # Extracción robusta de metadatos mediante Regex
    metadatos = extraer_metadatos(datos)

    for i, file in enumerate(archivos_fase2):
        progress(i, LenFase2, parte=0)
        ruta_archivo = ruta / file
        archivo_lf = lector(ruta_archivo, columnas)

        df_procesados = analisis_ovito(archivo_lf, pipeline_global, altura)

        # Inyección dinámica de metadatos y parámetros de ejecución
        df_procesados_final = df_procesados.with_columns(
            **{k: pl.lit(v) for k, v in metadatos.items()},  # type: ignore[arg-type]
            paso=pl.lit(paso_int),
            i=pl.lit(i),
            onda=pl.lit(onda_detect_bool),
        )

        lista_resultados.append(df_procesados_final)

    if LenFase2 > 0:
        progress(100, 100, parte=0)

    return lista_resultados
