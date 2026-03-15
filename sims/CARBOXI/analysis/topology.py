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
_ANALYSIS_COLUMNS: list[str] = ["id", "type", "x", "y", "z", "vx", "vy", "vz"]


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
) -> pl.DataFrame:
    """Calcula coordinación y clusters para las partículas erosionadas de un timestep.

    Aplica el criterio de erosión (``z > max_height`` AND ``vz > 0``) y pasa
    las partículas candidatas a OVITO para calcular sus propiedades topológicas.
    El proyectil (tipo 2) siempre se incluye en el resultado independientemente
    de si fue eyectado.

    Args:
        lf: LazyFrame con las columnas estándar del dump LAMMPS:
            ``["id", "type", "x", "y", "z", "vx", "vy", "vz"]``.
        pipeline: Pipeline de OVITO ya inicializado con ``build_ovito_pipeline``.
        max_height: Umbral vertical de la cama [m]. Las partículas con
            ``z > max_height`` y ``vz > 0`` se consideran eyectadas.

    Returns:
        DataFrame con todas las partículas de interés (erosionadas + proyectil),
        incluyendo las columnas adicionales:

        - ``"coordinacion"`` (Float64): número de vecinos dentro del cutoff.
        - ``"cluster"`` (UInt32): índice de cluster asignado por OVITO.
        - ``"proyectil_erosion"`` (Boolean | None): ``True`` si el proyectil
          fue eyectado, ``False`` si no lo fue, ``None`` para partículas de
          la cama.

    Example:
        >>> pipeline = build_ovito_pipeline(cutoff=1.14e-6)
        >>> df = analyze_erosion(lf_timestep, pipeline, max_height=5.2e-5)
        >>> df.filter(pl.col("coordinacion") == 0)  # partículas aisladas
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


def run_topology_analysis(
    dump_files: list[Path],
    pipeline: Pipeline,
    max_height: float,
    sim_id: str,
    timestep_interval: int,
    wave_detected: bool,
) -> list[pl.DataFrame]:
    """Ejecuta el análisis topológico completo sobre una lista de archivos dump.

    Procesa secuencialmente cada archivo de volcado LAMMPS, aplica el criterio
    de erosión y calcula coordinación y clusters con OVITO. Inyecta los
    metadatos de la simulación en cada DataFrame resultado para garantizar
    trazabilidad en el archivo Parquet final.

    Args:
        dump_files: Lista de rutas a los archivos ``output.*.gz`` a procesar,
            ordenados cronológicamente.
        pipeline: Pipeline de OVITO inicializado con ``build_ovito_pipeline``.
        max_height: Umbral de altura de la cama [m] para el criterio de erosión.
        sim_id: Identificador de la simulación (nombre de carpeta sin prefijo
            ``erosion_bed_SI_``). Se usa para extraer metadatos con regex.
        timestep_interval: Intervalo de volcado entre archivos consecutivos
            (frecuencia de dump en timesteps LAMMPS).
        wave_detected: ``True`` si la Fase 1 detectó una onda macroscópica.
            Se almacena como columna ``"onda"`` en los resultados.

    Returns:
        Lista de DataFrames, uno por cada archivo procesado. Cada DataFrame
        contiene las partículas erosionadas del timestep con sus propiedades
        topológicas y metadatos de simulación inyectados como columnas literales.
        Retorna lista vacía si ``dump_files`` está vacío.

    Example:
        >>> pipeline = build_ovito_pipeline(cutoff=1.14e-6)
        >>> results = run_topology_analysis(
        ...     dump_files[:50], pipeline, max_height, sim_id,
        ...     timestep_interval=10000, wave_detected=True
        ... )
        >>> df_total = pl.concat(results, how="vertical_relaxed")
    """
    if not dump_files:
        return []

    metadata = extraer_metadatos(sim_id)
    results: list[pl.DataFrame] = []
    total = len(dump_files)

    for i, file_path in enumerate(dump_files):
        print_progress(i, total, label="erosión")

        lf = read_dump(file_path, _ANALYSIS_COLUMNS)
        df = analyze_erosion(lf, pipeline, max_height)

        df = df.with_columns(
            **{k: pl.lit(v) for k, v in metadata.items()},
            paso=pl.lit(timestep_interval),
            i=pl.lit(i),
            onda=pl.lit(wave_detected),
        )
        results.append(df)

    print_progress(100, 100, label="erosión")
    print()

    return results


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------


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
