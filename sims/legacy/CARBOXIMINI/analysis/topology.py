"""
Análisis de clusters y física computacional para partículas eyectadas.

Aplica cluster analysis de OVITO sobre las partículas del dump de aire
(eyectadas + proyectil), computa fuerza Z neta por cluster y filtra
aquellos en vuelo balístico libre (fuerza neta ≈ 0).

Física: En ausencia de gravedad, un cluster libre de contacto con la cama
tiene fuerza Z neta cercana a cero. La tolerancia configurable define el
umbral de esta aproximación.
"""

from pathlib import Path

import polars as pl
from numpy import empty as np_empty
from ovito.data import DataCollection, SimulationCell  # type: ignore[import-untyped]
from ovito.modifiers import ClusterAnalysisModifier  # type: ignore[import-untyped]
from ovito.pipeline import Pipeline, StaticSource  # type: ignore[import-untyped]

from ..io.lammps_reader import read_air_files
from ..utils import print_progress


def build_ovito_pipeline(cutoff: float) -> Pipeline:
    """Inicializa pipeline OVITO con ClusterAnalysisModifier.

    Crea una estructura de datos vacía como fuente estática y adjunta
    el modificador de clusters. El pipeline se reutiliza en todos los
    timesteps reemplazando el source para evitar overhead.

    Args:
        cutoff: Distancia de corte para cluster analysis [m].
            Calculado como ``radio_m * factor_cutoff``.

    Returns:
        Pipeline de OVITO configurado, listo para recibir datos.

    Example:
        >>> pipeline = build_ovito_pipeline(cutoff=7.6e-7 * 2)
    """
    data = DataCollection()
    particles = data.create_particles()
    particles.create_property("Position", data=np_empty((0, 3)))

    cell = SimulationCell()
    cell[:, :] = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
    data.objects.append(cell)

    pipeline = Pipeline(source=StaticSource(data=data))
    pipeline.modifiers.append(
        ClusterAnalysisModifier(cutoff=cutoff, sort_by_size=True)
    )

    return pipeline


def run_cluster_analysis(
    df: pl.DataFrame, pipeline: Pipeline
) -> pl.DataFrame:
    """Ejecuta cluster analysis de OVITO sobre un DataFrame de partículas.

    Extrae las coordenadas [x, y, z] como arreglo NumPy, ejecuta el pipeline
    de OVITO y agrega la columna ``Cluster_ID`` al DataFrame original.

    Args:
        df: DataFrame con columnas ``"x"``, ``"y"``, ``"z"``. Debe tener
            al menos una fila.
        pipeline: Pipeline de OVITO inicializado con ``build_ovito_pipeline``.

    Returns:
        DataFrame original con columna adicional ``"Cluster_ID"`` (UInt32).
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
            "Cluster_ID",
            computed.particles["Cluster"][:],
            dtype=pl.UInt32,
        )
    )


def compute_cluster_physics(
    df: pl.DataFrame, tolerancia_fz: float
) -> pl.DataFrame:
    """Computa fuerza Z neta por cluster y filtra los de vuelo balístico.

    Agrupa por ``Cluster_ID`` y calcula agregaciones nativas de Polars:
    fuerza Z neta (suma de ``fz``), cantidad de partículas y presencia
    del proyectil (type == 2).

    Un cluster se considera en vuelo balístico libre si la fuerza Z neta
    es cercana a cero: ``|fz_sum| < tolerancia_fz``.

    Args:
        df: DataFrame con ``Cluster_ID``, ``fz``, ``type``.
        tolerancia_fz: Umbral absoluto de fuerza Z neta [N].

    Returns:
        DataFrame filtrado con columnas: ``Cluster_ID``, ``fz_sum``,
        ``n_particles``, ``proyectil_erosion``.
        Solo contiene clusters que pasan el filtro balístico.
    """
    cluster_stats = df.group_by("Cluster_ID").agg(
        pl.col("fz").sum().alias("fz_sum"),
        pl.col("id").len().alias("n_particles"),
        pl.col("type").eq(2).any().alias("has_projectile"),
    )

    ballistic = cluster_stats.filter(
        pl.col("fz_sum").abs() < tolerancia_fz
    )

    has_any_projectile = df.select(
        pl.col("type").eq(2).any()
    ).item()

    if has_any_projectile:
        projectile_cluster_ids = (
            df.filter(pl.col("type") == 2)
            .select("Cluster_ID")
            .unique()
            .to_series()
        )

        ballistic = ballistic.with_columns(
            pl.col("Cluster_ID")
            .is_in(projectile_cluster_ids)
            .alias("proyectil_erosion")
        )
    else:
        ballistic = ballistic.with_columns(
            pl.lit(None).cast(pl.Boolean).alias("proyectil_erosion")
        )

    return ballistic


def analyze_timestep(
    file_paths: list[Path],
    pipeline: Pipeline,
    tolerancia_fz: float,
) -> pl.DataFrame | None:
    """Procesa un paso temporal completo: lectura, clusters, física.

    Args:
        file_paths: Archivos de aire para este timestep (uno por core).
        pipeline: Pipeline de OVITO.
        tolerancia_fz: Umbral para filtro balístico [N].

    Returns:
        DataFrame con clusters balísticos y sus propiedades, o None
        si no hay partículas o no hay clusters que pasen el filtro.
    """
    lf = read_air_files(file_paths)
    df = lf.collect()

    if df.is_empty():
        return None

    df = run_cluster_analysis(df, pipeline)

    result = compute_cluster_physics(df, tolerancia_fz)

    if result.is_empty():
        return None

    return result


def run_topology_analysis(
    air_files_by_step: dict[int, list[Path]],
    pipeline: Pipeline,
    tolerancia_fz: float,
    sim_id: str,
    metadata: dict[str, str],
    cutoff_step: int,
    timestep_interval: int,
) -> list[pl.DataFrame]:
    """Ejecuta análisis topológico sobre todos los pasos hasta el cutoff.

    Itera los pasos ordenados, ejecuta ``analyze_timestep()`` por cada uno,
    inyecta metadatos de simulación como columnas literales y calcula
    la columna ``step = i * timestep_interval``.

    Args:
        air_files_by_step: Diccionario ``{step: [archivos]}`` ordenado.
        pipeline: Pipeline de OVITO.
        tolerancia_fz: Umbral de filtro balístico [N].
        sim_id: Identificador de la simulación.
        metadata: Metadatos extraídos del nombre de carpeta.
        cutoff_step: Último paso a procesar (inclusive). Pasos mayores
            se descartan (freno de emergencia por onda de choque).
        timestep_interval: Frecuencia de dump (diferencia entre pasos
            consecutivos en timesteps LAMMPS).

    Returns:
        Lista de DataFrames, uno por paso procesado. Cada DataFrame
        incluye columnas de metadatos inyectadas y ``step = i * paso``.
        Lista vacía si ningún paso produjo resultados.
    """
    steps_to_process = [
        s for s in air_files_by_step if s <= cutoff_step
    ]

    if not steps_to_process:
        return []

    results: list[pl.DataFrame] = []
    total = len(steps_to_process)

    for i, step in enumerate(steps_to_process):
        print_progress(i, total, label=f"clusters {sim_id}")

        df = analyze_timestep(
            air_files_by_step[step], pipeline, tolerancia_fz
        )

        if df is not None:
            df = df.with_columns(
                **{k: pl.lit(v) for k, v in metadata.items()},
                paso=pl.lit(timestep_interval),
                i=pl.lit(i),
            )
            results.append(df)

    print_progress(total, total, label=f"clusters {sim_id}")
    print()

    return results
