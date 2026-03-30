"""6 configuraciones de filtrado de erosión granular.

Filtros individuales (F1I, F2I, F3I):
  z > bed_height → sin Ovito

Filtros de cluster (F1C, F2C, F3C):
  z > bed_height → Ovito → filtrar por CM del cluster

bed_height = altura cruda de la cama (sin margen)
max_height = bed_height + margen (para filtro CM del cluster)
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

CONFIG_NAMES = ["F1I", "F1C", "F2I", "F2C", "F3I", "F3C"]
_FREEZE_HEIGHT = 1e-4


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
    max_height: float,
    tol: float,
    prev_dump: pl.DataFrame | None = None,
    pipeline: Pipeline | None = None,
    eroded_ids: list = [],
) -> pl.DataFrame:
    """F1I: z > bed_height (individual).

    Args:
        lf: LazyFrame del dump.
        bed_height: Altura de la cama cruda [m].
        max_height: No usado (compatibilidad de firma).
        prev_dump: No usado.
        tol: No usado.

    Returns:
        DataFrame con partículas sobre la cama.
    """
    mask = pl.col("z") > bed_height
    if eroded_ids:
        mask = mask & (pl.col("id").is_in(eroded_ids))
    return lf.filter(mask).collect()


def filter_F2I(
    lf: pl.LazyFrame,
    bed_height: float,
    max_height: float,
    tol: float,
    prev_dump: pl.DataFrame | None = None,
    pipeline: Pipeline | None = None,
    eroded_ids: list = [],
) -> pl.DataFrame:
    """F2I: z > bed_height + post-filtro vz > 0 para bed (individual).

    Inicial: z > bed_height (captura candidatas sin discriminar vz).
    Post-filtro: vz > 0 para partículas de cama (proyectil sin filtro).

    Args:
        lf: LazyFrame del dump.
        bed_height: Altura de la cama cruda [m].
        max_height: No usado.
        prev_dump: No usado.
        tol: No usado.

    Returns:
        DataFrame con partículas eyectadas + proyectil.
    """
    mask = pl.col("z") > bed_height
    if eroded_ids:
        mask = mask & (pl.col("id").is_in(eroded_ids))
    return lf.filter(
        mask & (pl.col("vz") > 0) | (pl.col("z") > _FREEZE_HEIGHT)
    ).collect()


def filter_F3I(
    lf: pl.LazyFrame,
    bed_height: float,
    max_height: float,
    tol: float,
    prev_dump: pl.DataFrame | None = None,
    pipeline: Pipeline | None = None,
    eroded_ids: list = [],
) -> pl.DataFrame:
    """F3I: z > bed_height + vz > 0 + |Δv| < tol (individual).

    Inicial: z > bed_height (candidatas).
    Post-filtro: vz > 0 AND |Δv| < tol para bed.

    Args:
        lf: LazyFrame del dump.
        bed_height: Altura de la cama cruda [m].
        max_height: No usado.
        prev_dump: Dump anterior con columnas ["id", "vx", "vy", "vz"].
        tol: Tolerancia de |Δv| [m/s].

    Returns:
        DataFrame con partículas balísticas + proyectil.
    """
    mask = pl.col("z") > bed_height
    if eroded_ids:
        mask = mask & (pl.col("id").is_in(eroded_ids))

    bed = lf.filter(
        mask & (pl.col("vz") > 0) | (pl.col("z") > _FREEZE_HEIGHT)
    ).collect()

    if prev_dump is None or bed.is_empty():
        return bed

    prev = prev_dump.select(["id", "vx", "vy", "vz"]).rename(
        {"vx": "vx_prev", "vy": "vy_prev", "vz": "vz_prev"}
    )
    merged = bed.join(prev, on="id", how="inner")

    if merged.is_empty():
        return pl.DataFrame()

    merged = merged.with_columns(
        (
            (pl.col("vx") - pl.col("vx_prev")) ** 2
            + (pl.col("vy") - pl.col("vy_prev")) ** 2
            + (pl.col("vz") - pl.col("vz_prev")) ** 2
        )
        .sqrt()
        .alias("delta_v")
    )

    filtered_bed = merged.filter(
        (pl.col("delta_v") < tol) | (pl.col("z") > _FREEZE_HEIGHT)
    )
    filtered_bed = filtered_bed.drop(["vx_prev", "vy_prev", "vz_prev", "delta_v"])

    return filtered_bed


# ---------------------------------------------------------------------------
# Filtros de cluster (con Ovito)
# ---------------------------------------------------------------------------


def filter_F1C(
    lf: pl.LazyFrame,
    bed_height: float,
    max_height: float,
    tol: float,
    prev_dump: pl.DataFrame | None = None,
    pipeline: Pipeline | None = None,
    eroded_ids: list = [],
) -> pl.DataFrame:
    """F1C: z > bed_height → Ovito → z_cm > max_height (cluster).

    Args:
        lf: LazyFrame del dump.
        bed_height: Altura de la cama cruda [m] (filtro inicial).
        max_height: Umbral CM del cluster [m].
        prev_dump: No usado.
        tol: No usado.
        pipeline: Pipeline de Ovito.
        eroded_ids:

    Returns:
        DataFrame con clusters sobre max_height.
    """
    mask = pl.col("z") > bed_height
    if eroded_ids:
        mask = mask & (pl.col("id").is_in(eroded_ids))
    df = lf.filter(mask).collect()

    if pipeline is None or df.is_empty():
        return df

    return _apply_cluster_filter(df, pipeline, max_height)


def filter_F2C(
    lf: pl.LazyFrame,
    bed_height: float,
    max_height: float,
    tol: float,
    prev_dump: pl.DataFrame | None = None,
    pipeline: Pipeline | None = None,
    eroded_ids: list = [],
) -> pl.DataFrame:
    """F2C: z > bed_height → Ovito → z_cm > max_height AND vz_cm > 0 (cluster).

    Args:
        lf: LazyFrame del dump.
        bed_height: Altura de la cama cruda [m] (filtro inicial).
        max_height: Umbral CM del cluster [m].
        prev_dump: No usado.
        tol: No usado.
        pipeline: Pipeline de Ovito.

    Returns:
        DataFrame con clusters eyectados.
    """
    mask = pl.col("z") > bed_height
    if eroded_ids:
        mask = mask & (pl.col("id").is_in(eroded_ids))
    df = lf.filter(mask).collect()

    if pipeline is None or df.is_empty():
        return df

    return _apply_cluster_filter(df, pipeline, max_height, require_vz=True)


def filter_F3C(
    lf: pl.LazyFrame,
    bed_height: float,
    max_height: float,
    tol: float,
    prev_dump: pl.DataFrame | None = None,
    pipeline: Pipeline | None = None,
    eroded_ids: list = [],
) -> pl.DataFrame:

    mask = pl.col("z") > bed_height
    if eroded_ids:
        mask = mask & (pl.col("id").is_in(eroded_ids))
    df = lf.filter(mask).collect()

    if pipeline is None or df.is_empty():
        return df

    # Filtro de Ovito
    df = _apply_cluster_filter(df, pipeline, max_height, require_vz=True)

    if prev_dump is None or df.is_empty():
        return df

    # 1. YA NO FILTRAMOS type != 2. El proyectil pasa.
    # Preparamos las velocidades del dump anterior
    prev = prev_dump.select(["id", "vx", "vy", "vz"]).rename(
        {"vx": "vx_old", "vy": "vy_old", "vz": "vz_old"}
    )

    # 2. Cruce duro (inner join) de los IDs que sobrevivieron
    merged = df.join(prev, on="id", how="inner")

    if merged.is_empty():
        return pl.DataFrame()

    # 3. Cálculo directo del CM con la data unida
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

    # ==========================================
    # ZONA DE DEBUG: ESTO SE IMPRIME EN TU TERMINAL
    # ==========================================
    # print(f"\n[DEBUG F3C] Tolerancia actual: {tol}")
    # print(cm_data.select(["cluster", "delta_v_cm"]))
    # ==========================================

    # 4. Ahora sí, aplicamos tu filtro de tolerancia
    valid_clusters = cm_data.filter(
        (pl.col("delta_v_cm") < tol) | (pl.col("z_cm") > _FREEZE_HEIGHT)
    )["cluster"].to_list()

    if valid_clusters:
        return df.filter(pl.col("cluster").is_in(valid_clusters))

    return pl.DataFrame()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _apply_cluster_filter(
    df: pl.DataFrame,
    pipeline: Pipeline,
    max_height: float,
    require_vz: bool = False,
) -> pl.DataFrame:
    """Aplica filtro por CM del cluster.

    Args:
        df: DataFrame con candidatas.
        pipeline: Pipeline de Ovito.
        max_height: Umbral CM del cluster [m].
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
        mask = mask & (pl.col("vz_cm") > 0) | (pl.col("z_cm") > _FREEZE_HEIGHT)

    valid = cluster_cm.filter(mask)["cluster"].to_list()

    return df.filter(pl.col("cluster").is_in(valid))
