"""
Análisis de erosión de partículas usando OVITO.

Calcula el número de coordinación y el índice de cluster para las partículas
que fueron eyectadas de la cama de sílice durante el impacto.

Física: La coordinación mide cuántos vecinos tiene cada partícula dentro del
radio de corte (cutoff = radio * factor_cutoff). El análisis de clusters agrupa
partículas eyectadas que forman agregados.
"""

from __future__ import annotations

import polars as pl
from numpy import empty as np_empty
from ovito.data import DataCollection, SimulationCell  # type: ignore[import-untyped]
from ovito.modifiers import (  # type: ignore[import-untyped]
    ClusterAnalysisModifier,
    CoordinationAnalysisModifier,
)
from ovito.pipeline import Pipeline, StaticSource  # type: ignore[import-untyped]

from .config_and_discovery import extraer_metadatos, print_progress
from .storage_and_viz import LammpsTrajectoryReader


def build_ovito_pipeline(cutoff: float) -> Pipeline:
    """Inicializa el pipeline de OVITO con los modificadores de coordinación y clusters.

    Args:
        cutoff: Distancia de corte para los modificadores [m].

    Returns:
        Pipeline de OVITO configurado.
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


def _filter_F3C(
    lf: pl.LazyFrame,
    bed_height: float,
    max_height: float,
    tol: float,
    prev_dump: pl.DataFrame | None = None,
    pipeline: Pipeline | None = None,
    freeze_height: float = 0.0,
) -> pl.DataFrame:
    """Filtro F3C: z > bed_height → Ovito → z_cm > max_height AND vz_cm > 0 + |Δv_cm| < tol."""
    df = lf.filter((pl.col("z") > bed_height) | (pl.col("type") == 2)).collect()

    if pipeline is None or df.is_empty():
        return df

    df = _apply_cluster_filter(df, pipeline, max_height, freeze_height, require_vz=True)

    if prev_dump is None or df.is_empty():
        return lf.collect().clear()

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
        return df.filter(pl.col("cluster").is_in(valid_clusters) | pl.col("type").eq(2))

    return lf.collect().clear()


def analyze_erosion(
    lf: pl.LazyFrame,
    pipeline: Pipeline,
    bed_height: float,
    max_height: float,
    prev_dump: pl.DataFrame | None,
    tolerancia_v_cm: float,
    freeze_height: float,
) -> pl.DataFrame:
    """Aplica el filtro F3C a un timestep.

    Returns:
        DataFrame con partículas erosionadas + proyectil, incluyendo:
        - coordinacion: número de vecinos dentro del cutoff
        - cluster: índice de cluster asignado por OVITO
        - proyectil_erosion: True/False/None según eyección del proyectil
    """
    df = _filter_F3C(
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
    reader: LammpsTrajectoryReader,
    cut_index: int,
    pipeline: Pipeline,
    bed_height: float,
    max_height: float,
    sim_id: str,
    timestep_interval: int,
    tolerancia_v_cm: float,
    freeze_height: float,
) -> list[pl.DataFrame]:
    """Ejecuta el análisis topológico completo sobre los frames del reader.

    Flujo de 2 fases:
    1. Filtrado: aplica el filtro F3C a cada frame independientemente.
    2. Tracking: filtra todos los frames por IDs del último frame.

    Args:
        reader: LammpsTrajectoryReader con los frames de la simulación.
        cut_index: Índice del frame donde cortar el análisis (para wave detection).
        pipeline: Pipeline de OVITO configurado.
        bed_height: Altura de la cama de partículas [m].
        max_height: Altura máxima de análisis [m].
        sim_id: Identificador de la simulación.
        timestep_interval: Intervalo de timesteps entre frames.
        tolerancia_v_cm: Tolerancia de |Δv_cm| [m/s].
        freeze_height: Altura de congelamiento [m].

    Returns:
        Lista de DataFrames, uno por cada frame procesado.
    """
    metadata = extraer_metadatos(sim_id)
    results: list[pl.DataFrame] = []
    prev_dump: pl.DataFrame | None = None
    total = cut_index

    for i in range(total):
        print_progress(i, total, label="erosión")

        lf = reader.get_frame_lazy(i)
        df_raw = lf.collect()

        df = analyze_erosion(
            lf=lf,
            pipeline=pipeline,
            bed_height=bed_height,
            max_height=max_height,
            prev_dump=prev_dump,
            tolerancia_v_cm=tolerancia_v_cm,
            freeze_height=freeze_height,
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
    """Aplica tracking acumulativo filtrando por IDs del último frame."""
    final_df = results[-1]
    eroded_ids = final_df["id"].to_list()

    if not eroded_ids:
        return [resultados.clear() for resultados in results]

    tracked: list[pl.DataFrame] = []
    for df in results:
        if not df.is_empty():
            tracked.append(df.filter(pl.col("id").is_in(eroded_ids)))

    return tracked


def _run_ovito(df: pl.DataFrame, pipeline: Pipeline) -> pl.DataFrame:
    """Aplica el pipeline de OVITO a un DataFrame de partículas."""
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
    """Aplica filtro por CM del cluster tras ejecutar Ovito."""
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

    erosion = df.filter(
        pl.col("cluster").is_in(valid) | pl.col("type").eq(2)
    ).with_columns(
        pl.when(
            pl.col("type").eq(2)
            & (
                (pl.col("z") > max_height) & (pl.col("vz") > 0)
                | (pl.col("z") > freeze_height)
            )
        )
        .then(True)
        .when(pl.col("type").eq(2))
        .then(False)
        .otherwise(pl.lit(None).cast(pl.Boolean))
        .alias("proyectil_erosion")
    )

    return erosion


def _empty_result() -> pl.DataFrame:
    """Devuelve un DataFrame vacío con el esquema completo de salida."""
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
            "coordinacion": pl.Float64,
            "cluster": pl.UInt32,
            "proyectil_erosion": pl.Boolean,
        }
    )


def _add_result_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Agrega columnas de resultado estándar al DataFrame filtrado."""
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

    return df
