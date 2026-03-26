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
    bed_height: float,
    max_height: float,
    prev_dump: pl.DataFrame | None,
    tolerancia_v_cm: float,
) -> pl.DataFrame:
    """Calcula coordinación y clusters para las partículas erosionadas de un timestep.

    Filtrado en dos etapas:

    1. **Candidatas**: partículas con ``z > bed_height`` y ``vz > 0``. Captura
       todas las partículas sobre la superficie de la cama, incluso las que
       oscilan cerca del umbral.
    2. **Clusters por CM**: tras ejecutar cluster analysis con OVITO, se
       conservan solo clusters cuyo centro de masa cumple
       ``z_cm > max_height`` (donde ``max_height = bed_height + margen``).
       Esto elimina la sensibilidad a oscilaciones térmicas de partículas
       individuales.

    El proyectil (tipo 2) siempre se incluye en el resultado.

    Args:
        lf: LazyFrame con las columnas del dump LAMMPS incluyendo ``fz``:
            ``["id", "type", "x", "y", "z", "vx", "vy", "vz", "fz"]``.
        pipeline: Pipeline de OVITO ya inicializado con ``build_ovito_pipeline``.
        bed_height: Altura cruda de la cama [m]. Filtro inicial de candidatas.
        max_height: Umbral de CM del cluster [m]. ``bed_height + margen``.
        prev_dump: Dump anterior completo con columnas ``["id", "vx", "vy", "vz"]``.
            None si es el primer dump.
        tolerancia_v_cm: Tolerancia de |Δv_cm| entre dumps [m/s].

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
        >>> df = analyze_erosion(lf_timestep, pipeline, bed_height=5.0e-5,
        ...                      max_height=5.4e-5, prev_dump=None,
        ...                      tolerancia_v_cm=0.0)
    """
    lf = lf.with_columns(
        ((pl.col("z") > bed_height) & (pl.col("vz") > 0)).alias("eroded")
    )
    df = lf.filter((pl.col("eroded")) | (pl.col("type") == 2)).collect()

    if df.is_empty():
        return _empty_result(df)

    df_eroded = df.filter(pl.col("eroded"))
    df_not_eroded = df.filter(~pl.col("eroded"))

    if not df_eroded.is_empty():
        df_eroded = _run_ovito(df_eroded, pipeline)

        # Filtrar clusters por CM z > max_height
        cluster_cm = (
            df_eroded.group_by("cluster")
            .agg(pl.col("z").mean().alias("z_cm"))
        )
        valid_clusters = cluster_cm.filter(
            pl.col("z_cm") > max_height
        )["cluster"].to_list()

        df_eroded = df_eroded.filter(pl.col("cluster").is_in(valid_clusters))

        if not df_eroded.is_empty():
            ballistic_ids = _filter_ballistic_clusters(
                df_eroded, tolerancia_v_cm, prev_dump
            )
            if ballistic_ids:
                df_eroded = df_eroded.filter(
                    pl.col("cluster").is_in(ballistic_ids)
                    | (pl.col("type") == 2)
                )
            else:
                df_eroded = df_eroded.filter(pl.col("type") == 2)
        else:
            df_eroded = df_eroded.filter(pl.col("type") == 2)
    else:
        df_eroded = df_eroded.with_columns(
            pl.lit(None).cast(pl.Float64).alias("coordinacion"),
            pl.lit(None).cast(pl.UInt32).alias("cluster"),
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
    bed_height: float,
    max_height: float,
    sim_id: str,
    timestep_interval: int,
    tolerancia_v_cm: float,
) -> list[pl.DataFrame]:
    """Ejecuta el análisis topológico completo sobre una lista de archivos dump.

    Procesa secuencialmente cada archivo de volcado LAMMPS, aplica el criterio
    de erosión con filtro por CM del cluster y comparación inter-timestep.
    Inyecta los metadatos de la simulación en cada DataFrame resultado para
    garantizar trazabilidad.

    Args:
        dump_files: Lista de rutas a los archivos ``output.*.gz`` a procesar,
            ordenados cronológicamente.
        pipeline: Pipeline de OVITO inicializado con ``build_ovito_pipeline``.
        bed_height: Altura cruda de la cama [m] para filtro inicial de candidatas.
        max_height: Umbral de CM del cluster [m] (bed_height + margen).
        sim_id: Identificador de la simulación (nombre de carpeta sin prefijo
            ``erosion_bed_SI_``). Se usa para extraer metadatos con regex.
        timestep_interval: Intervalo de volcado entre archivos consecutivos
            (frecuencia de dump en timesteps LAMMPS).
        tolerancia_v_cm: Tolerancia de |Δv_cm| entre dumps [m/s].

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
    prev_dump: pl.DataFrame | None = None
    total = len(dump_files)

    for i, file_path in enumerate(dump_files):
        print_progress(i, total, label="erosión")

        lf = read_dump(file_path, _ANALYSIS_COLUMNS)
        df_raw = lf.collect()

        df = analyze_erosion(
            lf, pipeline, bed_height, max_height, prev_dump, tolerancia_v_cm
        )

        prev_dump = df_raw.select(["id", "vx", "vy", "vz"])

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


def _filter_ballistic_clusters(
    df: pl.DataFrame,
    tolerancia_v_cm: float,
    prev_dump: pl.DataFrame | None = None,
) -> list[int]:
    """Identifica clusters cuyo CM no cambió de velocidad entre dumps.

    Para cada cluster en el dump actual, encuentra partículas que existían
    en el dump anterior (por ID). Si el |Δv_cm| entre dumps es menor que
    la tolerancia, el cluster se considera estático y se filtra.

    Si ``prev_dump`` es None (primer dump), no se filtra y se devuelven
    todos los clusters.

    Args:
        df: DataFrame con columnas ``"id"``, ``"cluster"``, ``"vx"``,
            ``"vy"``, ``"vz"``.
        tolerancia_v_cm: Tolerancia de |Δv_cm| [m/s]. 0 = estricto.
        prev_dump: DataFrame del dump anterior con columnas ``"id"``,
            ``"vx"``, ``"vy"``, ``"vz"``. None si es el primer dump.

    Returns:
        Lista de IDs de clusters que pasan el filtro (CM en movimiento).
    """
    if df.is_empty() or "cluster" not in df.columns:
        return []

    if prev_dump is None:
        return df["cluster"].unique().cast(pl.Int64).to_list()

    id_prev = set(prev_dump["id"])

    if not id_prev:
        return df["cluster"].unique().cast(pl.Int64).to_list()

    cluster_stats = (
        df.with_columns(pl.col("id").is_in(id_prev).alias("in_prev"))
        .group_by("cluster")
        .agg(
            pl.col("vx").filter(pl.col("in_prev")).mean().alias("vx_prev"),
            pl.col("vy").filter(pl.col("in_prev")).mean().alias("vy_prev"),
            pl.col("vz").filter(pl.col("in_prev")).mean().alias("vz_prev"),
            pl.col("vx").mean().alias("vx_curr"),
            pl.col("vy").mean().alias("vy_curr"),
            pl.col("vz").mean().alias("vz_curr"),
            pl.col("in_prev").sum().alias("n_prev"),
        )
        .filter(pl.col("n_prev") > 0)
        .with_columns(
            (
                (pl.col("vx_curr") - pl.col("vx_prev")) ** 2
                + (pl.col("vy_curr") - pl.col("vy_prev")) ** 2
                + (pl.col("vz_curr") - pl.col("vz_prev")) ** 2
            )
            .sqrt()
            .alias("delta_v_cm")
        )
        .filter(pl.col("delta_v_cm") < tolerancia_v_cm)
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
