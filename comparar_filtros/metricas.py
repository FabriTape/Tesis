"""Métricas por step para comparación de filtros de erosión."""

import polars as pl
from numpy import empty as np_empty
from ovito.data import DataCollection, SimulationCell  # type: ignore[import-untyped]
from ovito.modifiers import ClusterAnalysisModifier  # type: ignore[import-untyped]
from ovito.pipeline import Pipeline, StaticSource  # type: ignore[import-untyped]


def _run_ovito(df: pl.DataFrame, pipeline: Pipeline) -> pl.DataFrame:
    """Ejecuta Ovito sobre un DataFrame y agrega columna de cluster.

    Args:
        df: DataFrame con columnas ["x", "y", "z"].
        pipeline: Pipeline de Ovito configurado.

    Returns:
        DataFrame original con columna "cluster" agregada.
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


def build_cluster_pipeline(cutoff: float) -> Pipeline:
    """Inicializa pipeline de Ovito solo para cluster analysis.

    Args:
        cutoff: Distancia de corte [m].

    Returns:
        Pipeline configurado.
    """
    data = DataCollection()
    particles = data.create_particles()
    particles.create_property("Position", data=np_empty((0, 3)))

    cell = SimulationCell()
    cell[:, :] = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
    data.objects.append(cell)

    pipeline = Pipeline(source=StaticSource(data=data))
    pipeline.modifiers.append(ClusterAnalysisModifier(cutoff=cutoff))

    return pipeline


def compute_metrics(
    df: pl.DataFrame,
    step_index: int,
    pipeline: Pipeline | None = None,
) -> dict:
    """Calcula métricas de un DataFrame de partículas filtradas.

    Ejecuta Ovito sobre las partículas de cama para obtener n_clusters.

    Args:
        df: DataFrame con partículas filtradas (incluye proyectil).
        step_index: Índice del step/dump.
        pipeline: Pipeline de Ovito para cluster analysis. Si es None,
            n_clusters será 0.

    Returns:
        Diccionario con métricas por step.
    """
    if df.is_empty() or "type" not in df.columns:
        return {
            "step": step_index,
            "n_bed": 0,
            "n_proj": 0,
            "n_total": 0,
            "n_clusters": 0,
            "z_max": 0.0,
            "n_unique_ids": 0,
        }

    bed = df.filter(pl.col("type") != 2)
    proj = df.filter(pl.col("type") == 2)

    n_clusters = 0
    if pipeline is not None and not bed.is_empty():
        bed_with_cluster = _run_ovito(bed, pipeline)
        n_clusters = bed_with_cluster["cluster"].n_unique()

    return {
        "step": step_index,
        "n_bed": bed.shape[0],
        "n_proj": proj.shape[0],
        "n_total": df.shape[0],
        "n_clusters": n_clusters,
        "z_max": bed["z"].max() if not bed.is_empty() else 0.0,
        "n_unique_ids": bed["id"].n_unique(),
    }


def compute_gaps_per_step(df: pl.DataFrame) -> list[dict]:
    """Calcula gaps acumulados por step.

    Para cada step, cuenta cuántos IDs detectados en pasos anteriores
    no están presentes en el step actual.

    Args:
        df: DataFrame concatenado de todos los steps con columna "i".

    Returns:
        Lista de dicts con "step" y "gaps_acumulados".
    """
    bed = df.filter(pl.col("type") != 2)
    if bed.is_empty():
        return []

    max_step = bed["i"].max()
    seen_ids: set[int] = set()
    result = []

    for step in range(max_step + 1):
        step_ids = set(bed.filter(pl.col("i") == step)["id"].to_list())
        gaps = len(seen_ids - step_ids)
        result.append({"step": step, "gaps_acumulados": gaps})
        seen_ids.update(step_ids)

    return result


def compute_gaps_per_step_raw(df: pl.DataFrame) -> list[dict]:
    """Calcula gaps POR step (no acumulados).

    Para cada step, cuenta cuántos IDs NO están presentes en el step siguiente
    (IDs que desaparecieron hacia adelante).

    Args:
        df: DataFrame concatenado de todos los steps con columna "i".

    Returns:
        Lista de dicts con "step" y "gaps".
    """
    bed = df.filter(pl.col("type") != 2)
    if bed.is_empty():
        return []

    max_step = bed["i"].max()
    result = []

    for step in range(max_step + 1):
        step_ids = set(bed.filter(pl.col("i") == step)["id"].to_list())
        if step < max_step:
            next_ids = set(bed.filter(pl.col("i") == step + 1)["id"].to_list())
            gaps = len(step_ids - next_ids)
        else:
            gaps = 0
        result.append({"step": step, "gaps": gaps})

    return result


def compute_gaps(df: pl.DataFrame) -> int:
    """Cuenta partículas con gaps (entran y salen entre steps).

    Args:
        df: DataFrame concatenado de todos los steps con columna "i".

    Returns:
        Número de partículas con gaps.
    """
    bed = df.filter(pl.col("type") != 2)
    if bed.is_empty():
        return 0

    gaps = (
        bed.group_by("id")
        .agg(
            pl.col("i").min().alias("ini"),
            pl.col("i").max().alias("fin"),
            pl.col("i").count().alias("n"),
        )
        .with_columns((pl.col("fin") - pl.col("ini") + 1).alias("rango"))
        .filter(pl.col("n") != pl.col("rango"))
    )

    return gaps.shape[0]


def compute_fluctuations(metrics_list: list[dict]) -> int:
    """Cuenta fluctuaciones (disminuciones en n_bed entre steps).

    Args:
        metrics_list: Lista de diccionarios de métricas por step.

    Returns:
        Número de veces que n_bed disminuyó.
    """
    counts = [m["n_bed"] for m in metrics_list]
    return sum(1 for j in range(1, len(counts)) if counts[j] < counts[j - 1])
