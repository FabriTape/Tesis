"""
Análisis topológico de partículas erosionadas usando OVITO.

Calcula el número de coordinación y el índice de cluster para las partículas
que fueron eyectadas de la cama de sílice durante el impacto. El proyectil
(tipo 2) siempre se incluye en el resultado, con un indicador booleano de
si fue eyectado o no.

Seis configuraciones de filtrado están disponibles:

Filtros individuales (F1I, F2I, F3I):
  z > bed_height → sin Ovito

Filtros de cluster (F1C, F2C, F3C):
  z > bed_height → Ovito → filtrar por CM del cluster

El filtro activo se selecciona desde config.toml (sección [filtros],
clave ``filtro_activo``).

Física: La coordinación mide cuántos vecinos tiene cada partícula dentro del
radio de corte (cutoff = radio * factor_cutoff). El análisis de clusters agrupa
partículas eyectadas que forman agregados, lo que permite estudiar la
fragmentación del material durante la erosión por impacto.
"""

from __future__ import annotations

from collections.abc import Callable
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

# Nombres válidos de filtros.
VALID_FILTERS: tuple[str, ...] = ("F1I", "F1C", "F2I", "F2C", "F3I", "F3C")


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


# ---------------------------------------------------------------------------
# Filtros individuales (sin Ovito)
# ---------------------------------------------------------------------------


def _filter_F1I(
    lf: pl.LazyFrame,
    bed_height: float,
    max_height: float,
    tol: float,
    prev_dump: pl.DataFrame | None = None,
    pipeline: Pipeline | None = None,
    freeze_height: float = 0.0,
) -> pl.DataFrame:
    """F1I: z > max_height (individual, sin Ovito).

    Args:
        lf: LazyFrame del dump.
        bed_height: Altura de la cama cruda [m] (no usado).
        max_height: Umbral de altura [m].
        tol: No usado.
        prev_dump: No usado.
        pipeline: No usado.
        freeze_height: No usado.

    Returns:
        DataFrame con partículas sobre max_height.
    """
    return lf.filter(pl.col("z") > max_height).collect()


def _filter_F2I(
    lf: pl.LazyFrame,
    bed_height: float,
    max_height: float,
    tol: float,
    prev_dump: pl.DataFrame | None = None,
    pipeline: Pipeline | None = None,
    freeze_height: float = 0.0,
) -> pl.DataFrame:
    """F2I: z > bed_height + vz > 0, con fallback freeze_height (individual).

    Args:
        lf: LazyFrame del dump.
        bed_height: Altura de la cama cruda [m].
        max_height: No usado.
        tol: No usado.
        prev_dump: No usado.
        pipeline: No usado.
        freeze_height: Altura de congelamiento [m].

    Returns:
        DataFrame con partículas eyectadas + proyectil.
    """
    return lf.filter(
        ((pl.col("z") > bed_height) & (pl.col("vz") > 0))
        | (pl.col("z") > freeze_height)
    ).collect()


def _filter_F3I(
    lf: pl.LazyFrame,
    bed_height: float,
    max_height: float,
    tol: float,
    prev_dump: pl.DataFrame | None = None,
    pipeline: Pipeline | None = None,
    freeze_height: float = 0.0,
) -> pl.DataFrame:
    """F3I: z > bed_height + vz > 0 + |Δv| < tol (individual).

    Args:
        lf: LazyFrame del dump.
        bed_height: Altura de la cama cruda [m].
        max_height: No usado.
        tol: Tolerancia de |Δv| [m/s].
        prev_dump: Dump anterior con columnas ["id", "vx", "vy", "vz"].
        pipeline: No usado.
        freeze_height: Altura de congelamiento [m].

    Returns:
        DataFrame con partículas balísticas + proyectil.
    """
    bed = lf.filter(
        ((pl.col("z") > bed_height) & (pl.col("vz") > 0))
        | (pl.col("z") > freeze_height)
    ).collect()

    if prev_dump is None or bed.is_empty():
        return bed

    prev = prev_dump.select(["id", "vx", "vy", "vz"]).rename(
        {"vx": "vx_prev", "vy": "vy_prev", "vz": "vz_prev"}
    )
    merged = bed.join(prev, on="id", how="inner")

    if merged.is_empty():
        return _empty_result()

    merged = merged.with_columns(
        (
            (pl.col("vx") - pl.col("vx_prev")) ** 2
            + (pl.col("vy") - pl.col("vy_prev")) ** 2
            + (pl.col("vz") - pl.col("vz_prev")) ** 2
        )
        .sqrt()
        .alias("delta_v")
    )

    filtered = merged.filter(
        (pl.col("delta_v") < tol) | (pl.col("z") > freeze_height)
    )
    return filtered.drop(["vx_prev", "vy_prev", "vz_prev", "delta_v"])


# ---------------------------------------------------------------------------
# Filtros de cluster (con Ovito)
# ---------------------------------------------------------------------------


def _filter_F1C(
    lf: pl.LazyFrame,
    bed_height: float,
    max_height: float,
    tol: float,
    prev_dump: pl.DataFrame | None = None,
    pipeline: Pipeline | None = None,
    freeze_height: float = 0.0,
) -> pl.DataFrame:
    """F1C: z > bed_height → Ovito → z_cm > max_height (cluster).

    Args:
        lf: LazyFrame del dump.
        bed_height: Altura de la cama cruda [m] (filtro inicial).
        max_height: Umbral CM del cluster [m].
        tol: No usado.
        prev_dump: No usado.
        pipeline: Pipeline de Ovito.
        freeze_height: Altura de congelamiento [m].

    Returns:
        DataFrame con clusters sobre max_height.
    """
    df = lf.filter(pl.col("z") > bed_height).collect()

    if pipeline is None or df.is_empty():
        return df

    return _apply_cluster_filter(df, pipeline, max_height, freeze_height, require_vz=False)


def _filter_F2C(
    lf: pl.LazyFrame,
    bed_height: float,
    max_height: float,
    tol: float,
    prev_dump: pl.DataFrame | None = None,
    pipeline: Pipeline | None = None,
    freeze_height: float = 0.0,
) -> pl.DataFrame:
    """F2C: z > bed_height → Ovito → z_cm > max_height AND vz_cm > 0 (cluster).

    Args:
        lf: LazyFrame del dump.
        bed_height: Altura de la cama cruda [m] (filtro inicial).
        max_height: Umbral CM del cluster [m].
        tol: No usado.
        prev_dump: No usado.
        pipeline: Pipeline de Ovito.
        freeze_height: Altura de congelamiento [m].

    Returns:
        DataFrame con clusters eyectados.
    """
    df = lf.filter(pl.col("z") > bed_height).collect()

    if pipeline is None or df.is_empty():
        return df

    return _apply_cluster_filter(df, pipeline, max_height, freeze_height, require_vz=True)


def _filter_F3C(
    lf: pl.LazyFrame,
    bed_height: float,
    max_height: float,
    tol: float,
    prev_dump: pl.DataFrame | None = None,
    pipeline: Pipeline | None = None,
    freeze_height: float = 0.0,
) -> pl.DataFrame:
    """F3C: z > bed_height → Ovito → z_cm > max_height AND vz_cm > 0 + |Δv_cm| < tol.

    Args:
        lf: LazyFrame del dump.
        bed_height: Altura de la cama cruda [m] (filtro inicial).
        max_height: Umbral CM del cluster [m].
        tol: Tolerancia de |Δv_cm| [m/s].
        prev_dump: Dump anterior con columnas ["id", "vx", "vy", "vz"].
        pipeline: Pipeline de Ovito.
        freeze_height: Altura de congelamiento [m].

    Returns:
        DataFrame con clusters balísticos + proyectil.
    """
    df = lf.filter(pl.col("z") > bed_height).collect()

    if pipeline is None or df.is_empty():
        return df

    df = _apply_cluster_filter(df, pipeline, max_height, freeze_height, require_vz=True)

    if prev_dump is None or df.is_empty():
        return df

    prev = prev_dump.select(["id", "vx", "vy", "vz"]).rename(
        {"vx": "vx_old", "vy": "vy_old", "vz": "vz_old"}
    )

    merged = df.join(prev, on="id", how="inner")

    if merged.is_empty():
        return _empty_result()

    cm_data = (
        merged.group_by("cluster")
        .agg(
            pl.col("vx_old").mean().alias("vx_prev"),
            pl.col("vy_old").mean().alias("vy_prev"),
            pl.col("vz_old").mean().alias("vz_prev"),
            pl.col("vx").mean().alias("vx_curr"),
            pl.col("vy").mean().alias("vy_curr"),
            pl.col("vz").mean().alias("vz_curr"),
            pl.col("z").mean().alias("z_cm"),
        )
        .with_columns(
            (
                (pl.col("vx_curr") - pl.col("vx_prev")) ** 2
                + (pl.col("vy_curr") - pl.col("vy_prev")) ** 2
                + (pl.col("vz_curr") - pl.col("vz_prev")) ** 2
            )
            .sqrt()
            .alias("delta_v_cm")
        )
    )

    valid_clusters = cm_data.filter(
        (pl.col("delta_v_cm") < tol) | (pl.col("z_cm") > freeze_height)
    )["cluster"].to_list()

    if valid_clusters:
        return df.filter(pl.col("cluster").is_in(valid_clusters))

    return pl.DataFrame()


# ---------------------------------------------------------------------------
# Dispatcher y funciones públicas
# ---------------------------------------------------------------------------

_FILTER_REGISTRY: dict[str, Callable] = {
    "F1I": _filter_F1I,
    "F1C": _filter_F1C,
    "F2I": _filter_F2I,
    "F2C": _filter_F2C,
    "F3I": _filter_F3I,
    "F3C": _filter_F3C,
}


def analyze_erosion(
    lf: pl.LazyFrame,
    pipeline: Pipeline,
    bed_height: float,
    max_height: float,
    prev_dump: pl.DataFrame | None,
    tolerancia_v_cm: float,
    freeze_height: float,
    filter_name: str = "F3C",
) -> pl.DataFrame:
    """Aplica el filtro de erosión configurado a un timestep.

    Args:
        lf: LazyFrame con las columnas del dump LAMMPS:
            ``["id", "type", "x", "y", "z", "vx", "vy", "vz", "fz"]``.
        pipeline: Pipeline de OVITO ya inicializado con ``build_ovito_pipeline``.
        bed_height: Altura cruda de la cama [m]. Filtro inicial de candidatas.
        max_height: Umbral de CM del cluster [m]. ``bed_height + margen``.
        prev_dump: Dump anterior completo con columnas ``["id", "vx", "vy", "vz"]``.
            None si es el primer dump.
        tolerancia_v_cm: Tolerancia de |Δv_cm| entre dumps [m/s].
        freeze_height: Altura de congelamiento para filtros FxC [m].
        filter_name: Nombre del filtro a aplicar (F1I, F1C, F2I, F2C, F3I, F3C).

    Returns:
        DataFrame con las partículas erosionadas + proyectil, incluyendo:

        - ``"coordinacion"`` (Float64): número de vecinos dentro del cutoff.
        - ``"cluster"`` (UInt32): índice de cluster asignado por OVITO.
        - ``"proyectil_erosion"`` (Boolean | None): ``True`` si el proyectil
          fue eyectado, ``False`` si no lo fue, ``None`` para partículas de
          la cama.

    Raises:
        ValueError: Si ``filter_name`` no es un filtro válido.

    Example:
        >>> pipeline = build_ovito_pipeline(cutoff=1.14e-6)
        >>> df = analyze_erosion(lf_timestep, pipeline, bed_height=5.0e-5,
        ...                      max_height=5.4e-5, prev_dump=None,
        ...                      tolerancia_v_cm=0.0, freeze_height=2.4e-4,
        ...                      filter_name="F3C")
    """
    if filter_name not in _FILTER_REGISTRY:
        raise ValueError(
            f"Filtro '{filter_name}' no válido. Opciones: {VALID_FILTERS}"
        )

    filter_fn = _FILTER_REGISTRY[filter_name]

    df = filter_fn(
        lf=lf,
        bed_height=bed_height,
        max_height=max_height,
        tol=tolerancia_v_cm,
        prev_dump=prev_dump,
        pipeline=pipeline,
        freeze_height=freeze_height,
    )

    return _add_result_columns(df)


def run_topology_analysis(
    dump_files: list[Path],
    pipeline: Pipeline,
    bed_height: float,
    max_height: float,
    sim_id: str,
    timestep_interval: int,
    tolerancia_v_cm: float,
    freeze_height: float,
    filter_name: str = "F3C",
) -> list[pl.DataFrame]:
    """Ejecuta el análisis topológico completo sobre una lista de archivos dump.

    Flujo de 2 fases:

    1. **Filtrado**: aplica el filtro a cada frame independientemente.
    2. **Tracking**: toma las IDs del último frame y filtra todos los frames
       para mantener solo las partículas que persisten hasta el final.

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
        freeze_height: Altura de congelamiento para filtros FxC [m].
        filter_name: Nombre del filtro de erosión (F1I, F1C, F2I, F2C, F3I, F3C).

    Returns:
        Lista de DataFrames, uno por cada archivo procesado. Cada DataFrame
        contiene solo las partículas que aparecen en el último frame,
        con sus propiedades topológicas y metadatos de simulación inyectados.
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
            lf=lf,
            pipeline=pipeline,
            bed_height=bed_height,
            max_height=max_height,
            prev_dump=prev_dump,
            tolerancia_v_cm=tolerancia_v_cm,
            freeze_height=freeze_height,
            filter_name=filter_name,
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

    return _apply_tracking(results)


def _apply_tracking(results: list[pl.DataFrame]) -> list[pl.DataFrame]:
    """Aplica tracking acumulativo filtrando por IDs del último frame.

    Toma las IDs del último frame y filtra todos los frames para mantener
    solo las partículas que persisten hasta el final.

    Args:
        results: Lista de DataFrames por frame, ya filtrados.

    Returns:
        Lista de DataFrames filtrados por IDs del último frame.
    """
    final_df = results[-1]
    eroded_ids = final_df["id"].to_list()

    if not eroded_ids:
        return results

    tracked: list[pl.DataFrame] = []
    for df in results:
        if not df.is_empty():
            tracked.append(df.filter(pl.col("id").is_in(eroded_ids)))

    return tracked


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------


def _run_ovito(df: pl.DataFrame, pipeline: Pipeline) -> pl.DataFrame:
    """Aplica el pipeline de OVITO a un DataFrame de partículas.

    Args:
        df: DataFrame con columnas ``["x", "y", "z"]``.
        pipeline: Pipeline de OVITO con modificadores adjuntos.

    Returns:
        DataFrame original con columnas ``"coordinacion"`` y ``"cluster"``.
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


def _apply_cluster_filter(
    df: pl.DataFrame,
    pipeline: Pipeline,
    max_height: float,
    freeze_height: float,
    require_vz: bool = False,
) -> pl.DataFrame:
    """Aplica filtro por CM del cluster tras ejecutar Ovito.

    Args:
        df: DataFrame con candidatas.
        pipeline: Pipeline de Ovito.
        max_height: Umbral CM del cluster [m].
        freeze_height: Altura de congelamiento [m]. Clusters por encima
            de este umbral se incluyen sin filtro adicional.
        require_vz: Si True, también filtra por vz_cm > 0.

    Returns:
        DataFrame con clusters válidos.
    """
    if df.is_empty():
        return df

    df = _run_ovito(df, pipeline)

    cluster_cm = df.group_by("cluster").agg(
        pl.col("z").mean().alias("z_cm"),
        pl.col("vz").mean().alias("vz_cm"),
    )

    mask = pl.col("z_cm") > max_height
    if require_vz:
        mask = mask & (pl.col("vz_cm") > 0) | (pl.col("z_cm") > freeze_height)

    valid = cluster_cm.filter(mask)["cluster"].to_list()

    return df.filter(pl.col("cluster").is_in(valid))


def _empty_result() -> pl.DataFrame:
    """Devuelve un DataFrame vacío con el esquema completo de salida.

    Returns:
        DataFrame vacío con columnas ``id``, ``type``, ``x``, ``y``, ``z``,
        ``vx``, ``vy``, ``vz``, ``fz``, ``coordinacion``, ``cluster``,
        ``proyectil_erosion``.
    """
    return pl.DataFrame(
        schema={
            "id": pl.UInt32,
            "type": pl.UInt32,
            "x": pl.Float64,
            "y": pl.Float64,
            "z": pl.Float64,
            "vx": pl.Float64,
            "vy": pl.Float64,
            "vz": pl.Float64,
            "fz": pl.Float64,
            "coordinacion": pl.Float64,
            "cluster": pl.UInt32,
            "proyectil_erosion": pl.Boolean,
        }
    )


def _add_result_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Agrega columnas de resultado estándar al DataFrame filtrado.

    Args:
        df: DataFrame resultado de un filtro.

    Returns:
        DataFrame con columnas ``coordinacion``, ``cluster`` y ``proyectil_erosion``.
    """
    if df.is_empty():
        return _empty_result()

    has_coordinacion = "coordinacion" in df.columns
    has_cluster = "cluster" in df.columns

    df = df.with_columns(
        pl.lit(None).cast(pl.Float64).alias("coordinacion")
        if not has_coordinacion
        else pl.col("coordinacion"),
        pl.lit(None).cast(pl.UInt32).alias("cluster")
        if not has_cluster
        else pl.col("cluster"),
    )

    return df.with_columns(
        pl.lit(None).cast(pl.Boolean).alias("proyectil_erosion")
    )
