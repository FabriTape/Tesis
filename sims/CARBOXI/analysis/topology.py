"""
Análisis topológico de partículas erosionadas usando OVITO.

Calcula el número de coordinación y el índice de cluster para las partículas
que fueron eyectadas de la cama de sílice durante el impacto. El proyectil
(tipo 2) siempre se incluye en el resultado, con un indicador booleano de
si fue eyectado o no.

Física: La coordinación mide cuántos vecinos tiene cada partícula dentro del
radio de corte (cutoff = radio * factor_cutoff). El análisis de clusters agrupa
partículas eyectadas que forman agregados, lo que permite estudiar la
fragmentación del material durante la erosión por impacto.
"""

from pathlib import Path

import polars as pl
from numpy import empty as np_empty
from ovito.data import DataCollection, SimulationCell  # type: ignore[import-untyped]
from ovito.modifiers import (  # type: ignore[import-untyped]
    ClusterAnalysisModifier,
    CoordinationAnalysisModifier,
)
from ovito.pipeline import Pipeline, StaticSource  # type: ignore[import-untyped]

from ..io.lammps_reader import read_dump
from ..utils import extraer_metadatos, print_progress

# Columnas necesarias para el análisis topológico completo.
_ANALYSIS_COLUMNS: list[str] = ["id", "type", "x", "y", "z", "vx", "vy", "vz", "fz"]


def build_ovito_pipeline(cutoff: float) -> Pipeline:
    """Inicializa el pipeline de OVITO con los modificadores de coordinación y clusters.

    Crea una estructura de datos vacía como fuente estática y adjunta los
    modificadores de análisis. El pipeline se reutiliza en todos los timesteps
    para evitar overhead de inicialización repetida.

    Args:
        cutoff: Distancia de corte para los modificadores de coordinación y
            clusters [m]. Generalmente se calcula como ``radio * factor_cutoff``.

    Returns:
        Pipeline de OVITO configurado con ``CoordinationAnalysisModifier`` y
        ``ClusterAnalysisModifier``, listo para recibir nuevos datos en cada
        timestep.

    Example:
        >>> pipeline = build_ovito_pipeline(cutoff=7.6e-7 * 1.5)
        >>> # Reutilizar pipeline en el bucle de análisis
    """
    data = DataCollection()
    particles = data.create_particles()
    particles.create_property("Position", data=np_empty((0, 3)))

    cell = SimulationCell()
    cell[:, :] = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
    data.objects.append(cell)

    pipeline = Pipeline(source=StaticSource(data=data))
    pipeline.modifiers.append(CoordinationAnalysisModifier(cutoff=cutoff))
    pipeline.modifiers.append(ClusterAnalysisModifier(cutoff=cutoff))

    return pipeline


def analyze_erosion(
    lf: pl.LazyFrame,
    pipeline: Pipeline,
    max_height: float,
    tolerancia_fz: float,
) -> pl.DataFrame:
    """Calcula coordinación y clusters para las partículas erosionadas de un timestep.

    Aplica el criterio de erosión (``z > max_height`` AND ``vz > 0``), ejecuta
    cluster analysis con OVITO y luego filtra por vuelo balístico: solo se
    conservan los clusters cuya fuerza Z neta es cercana a cero
    (``|Σfz| < tolerancia_fz``), indicando que no están en contacto activo
    con la cama.

    El proyectil (tipo 2) siempre se incluye en el resultado.

    Args:
        lf: LazyFrame con las columnas del dump LAMMPS incluyendo ``fz``:
            ``["id", "type", "x", "y", "z", "vx", "vy", "vz", "fz"]``.
        pipeline: Pipeline de OVITO ya inicializado con ``build_ovito_pipeline``.
        max_height: Umbral vertical de la cama [m]. Las partículas con
            ``z > max_height`` y ``vz > 0`` se consideran eyectadas.
        tolerancia_fz: Tolerancia de fuerza Z neta por cluster [N]. Un cluster
            se considera en vuelo balístico si ``|Σfz| < tolerancia_fz``.

    Returns:
        DataFrame con las partículas erosionadas balísticas + proyectil,
        incluyendo las columnas adicionales:

        - ``"coordinacion"`` (Float64): número de vecinos dentro del cutoff.
        - ``"cluster"`` (UInt32): índice de cluster asignado por OVITO.
        - ``"proyectil_erosion"`` (Boolean | None): ``True`` si el proyectil
          fue eyectado, ``False`` si no lo fue, ``None`` para partículas de
          la cama.

    Example:
        >>> pipeline = build_ovito_pipeline(cutoff=1.14e-6)
        >>> df = analyze_erosion(lf_timestep, pipeline, max_height=5.2e-5,
        ...                      tolerancia_fz=1e-6)
    """
    lf = lf.with_columns(
        ((pl.col("z") > max_height) & (pl.col("vz") > 0)).alias("eroded")
    )
    df = lf.filter((pl.col("eroded")) | (pl.col("type") == 2)).collect()

    if df.is_empty():
        return _empty_result(df)

    df_eroded = df.filter(pl.col("eroded"))
    df_not_eroded = df.filter(~pl.col("eroded"))

    df_eroded = (
        _run_ovito(df_eroded, pipeline)
        if not df_eroded.is_empty()
        else (
            df_eroded.with_columns(
                pl.lit(None).cast(pl.Float64).alias("coordinacion"),
                pl.lit(None).cast(pl.UInt32).alias("cluster"),
            )
        )
    )

    ballistic_ids = _filter_ballistic_clusters(df_eroded, tolerancia_fz)

    if ballistic_ids:
        df_eroded = df_eroded.filter(
            pl.col("cluster").is_in(ballistic_ids) | (pl.col("type") == 2)
        )
    else:
        df_eroded = df_eroded.filter(pl.col("type") == 2)

    df_not_eroded = df_not_eroded.with_columns(
        pl.lit(None).cast(pl.Float64).alias("coordinacion"),
        pl.lit(None).cast(pl.UInt32).alias("cluster"),
    )

    df_final = pl.concat([df_eroded, df_not_eroded], how="vertical_relaxed")

    return df_final.with_columns(
        pl.when(pl.col("type") == 2)
        .then(pl.col("eroded"))
        .otherwise(pl.lit(None))
        .alias("proyectil_erosion")
    ).drop("eroded")


def analyze_erosion_taylor(
    lf: pl.LazyFrame,
    pipeline: Pipeline,
    max_height: float,
    tolerancia_fz: float,
    umbral_vz_min: float,
    radio_m: float,
    lonja_radios: int,
) -> pl.DataFrame:
    """Detecta erosión por análisis de clusters en una lonja de la cama.

    Selecciona todas las partículas por debajo de ``max_height`` hasta
    ``max_height - lonja_radios * radio_m`` (sin filtro previo de velocidad),
    ejecuta cluster analysis con OVITO y filtra clusters por su centro de masa:
    un cluster se considera eyectado si su CM tiene ``vz > umbral_vz_min``,
    ``z > max_height`` y ``|Σfz| < tolerancia_fz``.

    Luego filtra las partículas individuales del cluster para conservar solo
    las que están por encima de ``max_height`` (realmente eyectadas).

    El proyectil (tipo 2) siempre se incluye en el resultado.

    Args:
        lf: LazyFrame con columnas del dump LAMMPS incluyendo ``fz``:
            ``["id", "type", "x", "y", "z", "vx", "vy", "vz", "fz"]``.
        pipeline: Pipeline de OVITO inicializado con ``build_ovito_pipeline``.
        max_height: Umbral de altura de la cama [m] (con ``umbral_techo_radios``
            ya aplicado).
        tolerancia_fz: Tolerancia de fuerza Z neta por cluster [N].
        umbral_vz_min: Velocidad vertical mínima del CM del cluster [m/s].
        radio_m: Radio de partícula en metros [m].
        lonja_radios: Espesor de la lonja en número de radios.

    Returns:
        DataFrame con partículas de clusters eyectados + proyectil,
        incluyendo columnas ``"coordinacion"``, ``"cluster"`` y
        ``"proyectil_erosion"``.
    """
    z_min_slice = max_height - lonja_radios * radio_m

    lf = lf.with_columns(pl.lit(False).alias("eroded"))
    df = lf.filter((pl.col("z") > z_min_slice) | (pl.col("type") == 2)).collect()

    if df.is_empty():
        return _empty_result(df)

    df_slice = df.filter(pl.col("z") > z_min_slice)
    df_proj = df.filter(pl.col("type") == 2)

    if df_slice.is_empty():
        return _empty_result(df)

    df_slice = _run_ovito(df_slice, pipeline)
    ballistic_ids = _filter_ballistic_clusters_taylor(
        df_slice, tolerancia_fz, umbral_vz_min, max_height
    )

    if ballistic_ids:
        df_eroded = df_slice.filter(
            pl.col("cluster").is_in(ballistic_ids)
            & (pl.col("z") > max_height)
            & (pl.col("vz") > 0)
        )
    else:
        df_eroded = df_slice.filter(pl.lit(False))

    if not df_proj.is_empty():
        df_proj = df_proj.with_columns(
            pl.lit(None).cast(pl.Float64).alias("coordinacion"),
            pl.lit(None).cast(pl.UInt32).alias("cluster"),
        )
        df_final = pl.concat([df_eroded, df_proj], how="vertical_relaxed")
    else:
        df_final = df_eroded

    return df_final.with_columns(
        pl.when(pl.col("type") == 2)
        .then(pl.lit(True))
        .otherwise(pl.lit(None))
        .alias("proyectil_erosion")
    ).drop("eroded")


def run_topology_analysis(
    dump_files: list[Path],
    pipeline: Pipeline,
    max_height: float,
    sim_id: str,
    timestep_interval: int,
    tolerancia_fz: float,
    method: str = "v1",
    radio_m: float = 0.0,
    lonja_radios: int = 10,
    umbral_vz_min: float = 0.0,
) -> list[pl.DataFrame]:
    """Ejecuta el análisis topológico completo sobre una lista de archivos dump.

    Procesa secuencialmente cada archivo de volcado LAMMPS, aplica el criterio
    de erosión con filtro balístico por fuerza Z neta y calcula coordinación
    y clusters con OVITO. Inyecta los metadatos de la simulación en cada
    DataFrame resultado para garantizar trazabilidad en el archivo Parquet final.

    Args:
        dump_files: Lista de rutas a los archivos ``output.*.gz`` a procesar,
            ordenados cronológicamente.
        pipeline: Pipeline de OVITO inicializado con ``build_ovito_pipeline``.
        max_height: Umbral de altura de la cama [m] para el criterio de erosión.
        sim_id: Identificador de la simulación (nombre de carpeta sin prefijo
            ``erosion_bed_SI_``). Se usa para extraer metadatos con regex.
        timestep_interval: Intervalo de volcado entre archivos consecutivos
            (frecuencia de dump en timesteps LAMMPS).
        tolerancia_fz: Tolerancia de fuerza Z neta por cluster [N] para el
            filtro balístico.
        method: Método de detección de erosión. ``"v1"`` usa filtro z+vz,
            ``"taylor"`` usa lonja con análisis de clusters por CM.
        radio_m: Radio de partícula en metros [m]. Solo se usa con method="taylor".
        lonja_radios: Espesor de la lonja en radios. Solo se usa con method="taylor".
        umbral_vz_min: Velocidad mínima del CM del cluster [m/s]. Solo con "taylor".

    Returns:
        Lista de DataFrames, uno por cada archivo procesado. Cada DataFrame
        contiene las partículas erosionadas del timestep con sus propiedades
        topológicas y metadatos de simulación inyectados como columnas literales.
        Retorna lista vacía si ``dump_files`` está vacío.
    """
    if not dump_files:
        return []

    metadata = extraer_metadatos(sim_id)
    results: list[pl.DataFrame] = []
    total = len(dump_files)

    for i, file_path in enumerate(dump_files):
        print_progress(i, total, label="erosión")

        lf = read_dump(file_path, _ANALYSIS_COLUMNS)

        if method == "taylor":
            df = analyze_erosion_taylor(
                lf,
                pipeline,
                max_height,
                tolerancia_fz,
                umbral_vz_min,
                radio_m,
                lonja_radios,
            )
        else:
            df = analyze_erosion(lf, pipeline, max_height, tolerancia_fz)

        df = df.with_columns(
            **{k: pl.lit(v) for k, v in metadata.items()},
            paso=pl.lit(timestep_interval),
            i=pl.lit(i),
        )
        results.append(df)

    print_progress(100, 100, label="erosión")
    print()

    return results


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------


def _filter_ballistic_clusters(df: pl.DataFrame, tolerancia_fz: float) -> list[int]:
    """Identifica clusters en vuelo balístico por fuerza Z neta cercana a cero.

    Agrupa por ``cluster`` y calcula la suma de ``fz`` para cada cluster.
    Un cluster se considera balístico (libre de contacto con la cama) si
    la fuerza Z neta es menor que la tolerancia.

    Args:
        df: DataFrame con columnas ``"cluster"`` y ``"fz"``.
        tolerancia_fz: Umbral absoluto de fuerza Z neta [N].

    Returns:
        Lista de IDs de clusters que pasan el filtro balístico.
    """
    if df.is_empty() or "cluster" not in df.columns:
        return []

    cluster_fz = (
        df.group_by("cluster")
        .agg(pl.col("fz").sum().alias("fz_sum"))
        .filter(pl.col("fz_sum").abs() < tolerancia_fz)
    )

    return cluster_fz["cluster"].cast(pl.Int64).to_list()


def _filter_ballistic_clusters_taylor(
    df: pl.DataFrame,
    tolerancia_fz: float,
    umbral_vz_min: float,
    max_height: float,
) -> list[int]:
    """Identifica clusters eyectados por propiedades del centro de masa.

    Agrupa por ``cluster`` y calcula el CM (z, vz) y la suma de fz.
    Un cluster pasa el filtro si TODAS las condiciones se cumplen:
    - ``vz_cm > umbral_vz_min`` (el CM se mueve hacia arriba)
    - ``z_cm > max_height`` (el CM está por encima de la superficie)
    - ``|Σfz| < tolerancia_fz`` (fuerza neta ≈ 0, vuelo balístico)

    Args:
        df: DataFrame con columnas ``"cluster"``, ``"z"``, ``"vz"``, ``"fz"``.
        tolerancia_fz: Umbral absoluto de fuerza Z neta [N].
        umbral_vz_min: Velocidad vertical mínima del CM [m/s].
        max_height: Altura mínima del CM para considerar eyección [m].

    Returns:
        Lista de IDs de clusters que pasan los tres filtros.
    """
    if df.is_empty() or "cluster" not in df.columns:
        return []

    cluster_stats = (
        df.group_by("cluster")
        .agg(
            pl.col("z").mean().alias("z_cm"),
            pl.col("vz").mean().alias("vz_cm"),
            pl.col("fz").sum().alias("fz_sum"),
        )
        .filter(
            (pl.col("vz_cm") > umbral_vz_min)
            & (pl.col("z_cm") > max_height)
            & (pl.col("fz_sum").abs() < tolerancia_fz)
        )
    )

    return cluster_stats["cluster"].cast(pl.Int64).to_list()


def _run_ovito(df: pl.DataFrame, pipeline: Pipeline) -> pl.DataFrame:
    """Aplica el pipeline de OVITO a un DataFrame de partículas erosionadas.

    Construye una nueva DataCollection con las posiciones del DataFrame,
    ejecuta el pipeline y extrae los arrays de coordinación y cluster
    calculados por OVITO.

    Args:
        df: DataFrame con columnas ``["x", "y", "z"]`` para las partículas
            a analizar. Debe contener al menos una fila.
        pipeline: Pipeline de OVITO con modificadores de coordinación y
            cluster ya adjuntos.

    Returns:
        DataFrame original con dos columnas añadidas:
        - ``"coordinacion"`` (Float64): número de vecinos por partícula.
        - ``"cluster"`` (UInt32): índice de cluster asignado.
    """
    positions = df.select(["x", "y", "z"]).to_numpy()

    new_data = DataCollection()
    particles = new_data.create_particles(count=positions.shape[0])
    particles.create_property("Position", data=positions)

    cell = SimulationCell()
    cell[:, :] = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
    new_data.objects.append(cell)

    pipeline.source = StaticSource(data=new_data)
    computed = pipeline.compute()

    return df.with_columns(
        pl.Series(
            "coordinacion", computed.particles["Coordination"][:], dtype=pl.Float64
        ),
        pl.Series("cluster", computed.particles["Cluster"][:], dtype=pl.UInt32),
    )


def _empty_result(df: pl.DataFrame) -> pl.DataFrame:
    """Devuelve un DataFrame vacío con el esquema esperado de salida.

    Se usa cuando no hay partículas de interés en un timestep dado.

    Args:
        df: DataFrame de referencia (puede estar vacío) del que se elimina
            la columna temporal ``"eroded"``.

    Returns:
        DataFrame con columnas ``"coordinacion"``, ``"cluster"`` y
        ``"proyectil_erosion"`` vacías y con los tipos correctos.
    """
    return df.drop("eroded", strict=False).with_columns(
        pl.Series("coordinacion", [], dtype=pl.Float64),
        pl.Series("cluster", [], dtype=pl.UInt32),
        pl.Series("proyectil_erosion", [], dtype=pl.Boolean),
    )
