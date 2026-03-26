"""Métricas por step para comparación de filtros de erosión."""

import polars as pl


def compute_metrics(df: pl.DataFrame, step_index: int) -> dict:
    """Calcula métricas de un DataFrame de partículas filtradas.

    Args:
        df: DataFrame con partículas filtradas (incluye proyectil).
        step_index: Índice del step/dump.

    Returns:
        Diccionario con métricas por step.
    """
    bed = df.filter(pl.col("type") != 2)
    proj = df.filter(pl.col("type") == 2)

    return {
        "step": step_index,
        "n_bed": bed.shape[0],
        "n_proj": proj.shape[0],
        "n_total": df.shape[0],
        "z_cm": bed["z"].mean() if not bed.is_empty() else 0.0,
        "vz_cm": bed["vz"].mean() if not bed.is_empty() else 0.0,
        "z_max": bed["z"].max() if not bed.is_empty() else 0.0,
        "n_unique_ids": bed["id"].n_unique(),
    }


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
