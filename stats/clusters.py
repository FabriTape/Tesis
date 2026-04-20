"""Funciones de cálculo de clusters para análisis de erosión.

Contiene funciones para calcular tamaños de clusters, distribuciones y estadísticas.
Usa Polars para operaciones vectorizadas.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def filter_erosion(df: pl.DataFrame, ff: float) -> pl.DataFrame:
    """Filtra exclusión de erosión: partículas tipo 1 con z > threshold o vz > 0.

    Args:
        df: DataFrame con columnas: ff, type, z, vz.
        ff: Fraction fill (fraction fill).

    Returns:
        DataFrame con partículas erosionadas.
    """
    return df.filter(
        (pl.col("ff") == ff)
        & (pl.col("i") == pl.col("i").max())
        & (pl.col("type") == 2)
        & ((pl.col("z") > 1e-4) | (pl.col("vz") > 0))
    )


def compute_unique_ids_by_velocity(
    df: pl.DataFrame, ff: float
) -> pl.DataFrame:
    """Calcula promedio de IDs únicos por velocidad.

    Args:
        df: DataFrame con columnas: vel0, angulo, posx, posy, id.
        ff: Fraction fill.

    Returns:
        DataFrame con estadísticas por vel0.
    """
    df_filt = filter_erosion(df, ff)

    grouped = (
        df_filt.group_by(["vel0", "angulo", "posx", "posy"])
        .agg(pl.col("id").n_unique().alias("unique_ids"))
    )

    result = (
        grouped.group_by("vel0")
        .agg(
            avg=pl.col("unique_ids").mean(),
            std=pl.col("unique_ids").std(),
            min=pl.col("unique_ids").min(),
            max=pl.col("unique_ids").max(),
        )
        .with_columns(
            (pl.col("avg") - pl.col("std").fill_null(0)).clip(0).alias("lower_std"),
            (pl.col("avg") + pl.col("std").fill_null(0)).alias("upper_std"),
        )
        .sort("vel0")
    )
    return result


def compute_unique_ids_raw_with_positions(
    df: pl.DataFrame, ff: float
) -> dict[float, pl.DataFrame]:
    """Calcula IDs únicos por cada simulación individual con posiciones.

    Args:
        df: DataFrame con columnas: vel0, angulo, posx, posy, id.
        ff: Fraction fill.

    Returns:
        Dict con vel0 -> DataFrame.
    """
    df_filt = filter_erosion(df, ff)

    grouped = (
        df_filt.group_by(["vel0", "angulo", "posx", "posy"])
        .agg(pl.col("id").n_unique().alias("unique_ids"))
    )

    result: dict[float, pl.DataFrame] = {}
    for vel0 in grouped["vel0"].unique().to_list():
        result[vel0] = grouped.filter(pl.col("vel0") == vel0).select(
            ["vel0", "angulo", "posx", "posy", "unique_ids"]
        )
    return result


def compute_erosion_counts_with_zeros(
    df: pl.DataFrame, ff: float, grid_pairs: list[tuple[float, float]]
) -> dict[float, np.ndarray]:
    """Calcula conteos de erosión por posición de forma optimizada.

    Args:
        df: DataFrame con columnas: vel0, posx, posy, id.
        ff: Fraction fill.
        grid_pairs: Lista de tuplas (posx, posy) para la grilla.

    Returns:
        Dict con vel0 -> np.ndarray de conteos.
    """
    df_filt = filter_erosion(df, ff)

    grouped = df_filt.group_by(["vel0", "posx", "posy"]).agg(
        pl.col("id").n_unique()
    )

    group_dict = {
        (row["vel0"], row["posx"], row["posy"]): row["id"]
        for row in grouped.iter_rows(named=True)
    }

    result: dict[float, np.ndarray] = {}
    vel0_unique = df_filt["vel0"].unique().to_list()
    for vel0 in vel0_unique:
        counts = [
            group_dict.get((vel0, px, py), 0) for px, py in grid_pairs
        ]
        result[vel0] = np.array(counts, dtype=float)

    return result


def compute_unique_ids_raw(df: pl.DataFrame, ff: float) -> dict[float, list[int]]:
    """Calcula IDs únicos por cada simulación individual.

    Args:
        df: DataFrame con columnas: vel0, angulo, posx, posy, id.
        ff: Fraction fill.

    Returns:
        Dict con vel0 -> lista de IDs únicos.
    """
    df_filt = filter_erosion(df, ff)

    grouped = (
        df_filt.group_by(["vel0", "angulo", "posx", "posy"])
        .agg(pl.col("id").n_unique().alias("unique_ids"))
    )

    result: dict[float, list[int]] = {}
    for vel0 in grouped["vel0"].unique().to_list():
        result[vel0] = grouped.filter(pl.col("vel0") == vel0)[
            "unique_ids"
        ].to_list()
    return result


def compute_cluster_frequency(df: pl.DataFrame, ff: float) -> pl.DataFrame:
    """Calcula frecuencia de tamaños de cluster por ff.

    Args:
        df: DataFrame con columnas: cluster, id.
        ff: Fraction fill.

    Returns:
        DataFrame con frecuencia por tamaño.
    """
    df_filt = filter_erosion(df, ff)

    sizes_per_sim = []
    for _key, group in df_filt.group_by(["vel0", "angulo", "posx", "posy"]):
        cluster_sizes = group.group_by("cluster").agg(
            pl.col("id").n_unique()
        )["id"]
        for size in cluster_sizes:
            sizes_per_sim.append({"cluster_size": size})

    if not sizes_per_sim:
        return pl.DataFrame()

    df_sizes = pl.DataFrame(sizes_per_sim)

    result = (
        df_sizes.group_by("cluster_size")
        .agg(pl.len().alias("frecuencia"))
        .sort("cluster_size")
    )
    return result


def compute_cluster_sizes(
    df: pl.DataFrame, ff: float, vel0: float
) -> np.ndarray:
    """Extrae tamaños de cluster para un ff y vel0 dados.

    Args:
        df: DataFrame con columnas: vel0, cluster, id.
        ff: Fraction fill.
        vel0: Velocidad inicial.

    Returns:
        Array de tamaños de cluster.
    """
    df_filt = df.filter(pl.col("vel0") == vel0)
    df_filt = filter_erosion(df_filt, ff)

    sizes = (
        df_filt.group_by(["angulo", "posx", "posy", "cluster"])
        .agg(pl.col("id").n_unique())
    )["id"].to_numpy()
    return sizes


def compute_cluster_size_stats(
    df: pl.DataFrame, ff: float, vel0: float
) -> dict[str, float]:
    """Calcula estadísticas de tamaños de cluster.

    Args:
        df: DataFrame con columnas: cluster, id.
        ff: Fraction fill.
        vel0: Velocidad inicial.

    Returns:
        Dict con estadísticas (min, max, mean, std, n_clusters).
    """
    sizes = compute_cluster_sizes(df, ff, vel0)
    if len(sizes) == 0:
        return {}

    return {
        "min": float(sizes.min()),
        "max": float(sizes.max()),
        "mean": float(sizes.mean()),
        "std": float(sizes.std()) if len(sizes) > 1 else 0.0,
        "n_clusters": len(sizes),
    }