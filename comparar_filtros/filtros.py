"""6 configuraciones de filtrado de erosión granular.

Filtros individuales (F1I, F2I, F3I):
  Aplican criterios a cada partícula individualmente.

Filtros de cluster (F1C, F2C, F3C):
  Ejecutan análisis de clusters con Ovito, calculan CM por cluster,
  y aplican criterios al CM del cluster.
"""

import polars as pl
from numpy import empty as np_empty
from ovito.data import DataCollection, SimulationCell  # type: ignore[import-untyped]
from ovito.modifiers import (  # type: ignore[import-untyped]
    ClusterAnalysisModifier,
    CoordinationAnalysisModifier,
)
from ovito.pipeline import Pipeline, StaticSource  # type: ignore[import-untyped]

_COLUMNS = ["id", "type", "x", "y", "z", "vx", "vy", "vz", "fz"]

# Nombres legibles de cada configuración
CONFIG_NAMES = ["F1I", "F1C", "F2I", "F2C", "F3I", "F3C"]


def build_ovito_pipeline(cutoff: float) -> Pipeline:
    """Inicializa pipeline de Ovito con modificadores de coordinación y clusters.

    Args:
        cutoff: Distancia de corte [m].

    Returns:
        Pipeline configurado y listo para reutilizar.
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


def _run_ovito(df: pl.DataFrame, pipeline: Pipeline) -> pl.DataFrame:
    """Ejecuta Ovito sobre un DataFrame y agrega columnas de cluster.

    Args:
        df: DataFrame con columnas ["x", "y", "z"].
        pipeline: Pipeline de Ovito configurado.

    Returns:
        DataFrame original con columnas "cluster" agregadas.
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
        pl.Series("cluster", computed.particles["Cluster"][:], dtype=pl.UInt32)
    )


# ---------------------------------------------------------------------------
# Filtros individuales (sin Ovito)
# ---------------------------------------------------------------------------


def filter_F1I(
    lf: pl.LazyFrame,
    bed_height: float,
    prev_dump: pl.DataFrame | None = None,
    tol: float = 0.0,
) -> pl.DataFrame:
    """F1I: Filtrado por z > bed_height (individual).

    Args:
        lf: LazyFrame del dump.
        bed_height: Altura de la cama [m].
        prev_dump: No usado (compatibilidad de firma).
        tol: No usado (compatibilidad de firma).

    Returns:
        DataFrame con partículas sobre la cama + proyectil.
    """
    df = lf.filter((pl.col("z") > bed_height) | (pl.col("type") == 2)).collect()
    return df


def filter_F2I(
    lf: pl.LazyFrame,
    bed_height: float,
    prev_dump: pl.DataFrame | None = None,
    tol: float = 0.0,
) -> pl.DataFrame:
    """F2I: Filtrado por z > bed_height AND vz > 0 (individual).

    Args:
        lf: LazyFrame del dump.
        bed_height: Altura de la cama [m].
        prev_dump: No usado.
        tol: No usado.

    Returns:
        DataFrame con partículas eyectadas + proyectil.
    """
    df = lf.filter(
        ((pl.col("z") > bed_height) & (pl.col("vz") > 0)) | (pl.col("type") == 2)
    ).collect()
    return df


def filter_F3I(
    lf: pl.LazyFrame,
    bed_height: float,
    prev_dump: pl.DataFrame | None = None,
    tol: float = 1e-6,
) -> pl.DataFrame:
    """F3I: Filtrado por z > bed_height AND vz > 0 AND |Δv| < tol (individual).

    Args:
        lf: LazyFrame del dump.
        bed_height: Altura de la cama [m].
        prev_dump: Dump anterior con columnas ["id", "vx", "vy", "vz"].
        tol: Tolerancia de |Δv| [m/s].

    Returns:
        DataFrame con partículas eyectadas balísticas + proyectil.
    """
    df = lf.filter(
        ((pl.col("z") > bed_height) & (pl.col("vz") > 0)) | (pl.col("type") == 2)
    ).collect()

    if prev_dump is None or df.is_empty():
        return df

    # Merge con dump anterior para comparar velocidades
    prev = prev_dump.select(["id", "vx", "vy", "vz"]).rename(
        {"vx": "vx_prev", "vy": "vy_prev", "vz": "vz_prev"}
    )
    merged = df.join(prev, on="id", how="left")

    # Calcular |Δv| solo para partículas que estaban en prev
    merged = merged.with_columns(
        pl.when(pl.col("vx_prev").is_not_null())
        .then(
            (
                (pl.col("vx") - pl.col("vx_prev")) ** 2
                + (pl.col("vy") - pl.col("vy_prev")) ** 2
                + (pl.col("vz") - pl.col("vz_prev")) ** 2
            ).sqrt()
        )
        .otherwise(pl.lit(-1.0))
        .alias("delta_v")
    )

    # Mantener: proyectil, partículas nuevas, o partículas con |Δv| < tol
    filtered = merged.filter(
        (pl.col("type") == 2)
        | (pl.col("vx_prev").is_null())
        | (pl.col("delta_v") < tol)
    )

    return filtered.drop(["vx_prev", "vy_prev", "vz_prev", "delta_v"])


# ---------------------------------------------------------------------------
# Filtros de cluster (con Ovito)
# ---------------------------------------------------------------------------


def filter_F1C(
    lf: pl.LazyFrame,
    bed_height: float,
    prev_dump: pl.DataFrame | None = None,
    tol: float = 0.0,
    pipeline: Pipeline | None = None,
) -> pl.DataFrame:
    """F1C: Filtrado por z_cm > bed_height (cluster).

    Args:
        lf: LazyFrame del dump.
        bed_height: Altura de la cama [m].
        prev_dump: No usado.
        tol: No usado.
        pipeline: Pipeline de Ovito.

    Returns:
        DataFrame con clusters cuyo CM está sobre la cama + proyectil.
    """
    df = lf.filter((pl.col("z") > bed_height) | (pl.col("type") == 2)).collect()

    if pipeline is None or df.is_empty():
        return df

    return _apply_cluster_filter(df, pipeline, bed_height)


def filter_F2C(
    lf: pl.LazyFrame,
    bed_height: float,
    prev_dump: pl.DataFrame | None = None,
    tol: float = 0.0,
    pipeline: Pipeline | None = None,
) -> pl.DataFrame:
    """F2C: Filtrado por z_cm > bed_height AND vz_cm > 0 (cluster).

    Args:
        lf: LazyFrame del dump.
        bed_height: Altura de la cama [m].
        prev_dump: No usado.
        tol: No usado.
        pipeline: Pipeline de Ovito.

    Returns:
        DataFrame con clusters eyectados + proyectil.
    """
    df = lf.filter(
        ((pl.col("z") > bed_height) & (pl.col("vz") > 0)) | (pl.col("type") == 2)
    ).collect()

    if pipeline is None or df.is_empty():
        return df

    return _apply_cluster_filter(df, pipeline, bed_height, require_vz=True)


def filter_F3C(
    lf: pl.LazyFrame,
    bed_height: float,
    prev_dump: pl.DataFrame | None = None,
    tol: float = 1e-6,
    pipeline: Pipeline | None = None,
) -> pl.DataFrame:
    """F3C: Filtrado por z_cm > bed_height AND vz_cm > 0 AND |Δv_cm| < tol (cluster).

    Args:
        lf: LazyFrame del dump.
        bed_height: Altura de la cama [m].
        prev_dump: Dump anterior con columnas ["id", "vx", "vy", "vz"].
        tol: Tolerancia de |Δv_cm| [m/s].
        pipeline: Pipeline de Ovito.

    Returns:
        DataFrame con clusters balísticos + proyectil.
    """
    df = lf.filter(
        ((pl.col("z") > bed_height) & (pl.col("vz") > 0)) | (pl.col("type") == 2)
    ).collect()

    if pipeline is None or df.is_empty():
        return df

    df = _apply_cluster_filter(df, pipeline, bed_height, require_vz=True)

    if prev_dump is None or df.is_empty():
        return df

    # Filtrar por |Δv_cm| entre dumps para cada cluster
    prev_ids = set(prev_dump["id"].to_list())
    bed = df.filter(pl.col("type") != 2)

    if bed.is_empty():
        return df

    cluster_delta_v = (
        bed.with_columns(pl.col("id").is_in(prev_ids).alias("in_prev"))
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
        .filter(pl.col("delta_v_cm") < tol)
    )

    valid_clusters = cluster_delta_v["cluster"].to_list()
    proj = df.filter(pl.col("type") == 2)

    if valid_clusters:
        filtered = df.filter(
            pl.col("cluster").is_in(valid_clusters) | (pl.col("type") == 2)
        )
    else:
        filtered = proj

    return filtered


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _apply_cluster_filter(
    df: pl.DataFrame,
    pipeline: Pipeline,
    bed_height: float,
    require_vz: bool = False,
) -> pl.DataFrame:
    """Aplica filtro por CM del cluster.

    Args:
        df: DataFrame con candidatas + proyectil.
        pipeline: Pipeline de Ovito.
        bed_height: Altura de la cama [m].
        require_vz: Si True, también filtra por vz_cm > 0.

    Returns:
        DataFrame con clusters válidos + proyectil.
    """
    df_eroded = df.filter(pl.col("type") != 2)
    proj = df.filter(pl.col("type") == 2)

    if df_eroded.is_empty():
        return df

    df_eroded = _run_ovito(df_eroded, pipeline)

    cluster_cm = df_eroded.group_by("cluster").agg(
        pl.col("z").mean().alias("z_cm"),
        pl.col("vz").mean().alias("vz_cm"),
    )

    mask = pl.col("z_cm") > bed_height
    if require_vz:
        mask = mask & (pl.col("vz_cm") > 0)

    valid = cluster_cm.filter(mask)["cluster"].to_list()
    df_eroded = df_eroded.filter(pl.col("cluster").is_in(valid))

    return (
        pl.concat([df_eroded, proj], how="diagonal")
        if not proj.is_empty()
        else df_eroded
    )
