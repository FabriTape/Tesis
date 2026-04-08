"""Módulo de cálculos de clusters.

Contiene funciones para calcular tamaños de clusters, distribuciones y estadísticas.
"""

import pandas as pd
import numpy as np

from data_loader import (
    filter_by_ff,
    filter_by_vel0,
    filter_by_i_value,
    filter_exclude_erosion,
)
from constants import I_VALUE


def compute_unique_ids_by_velocity(
    df: pd.DataFrame, ff: float
) -> pd.DataFrame:
    """Calcula promedio de IDs únicos por velocidad.

    Args:
        df: DataFrame con datos.
        ff: Factor de fricción.

    Returns:
        DataFrame con estadísticas por vel0.
    """
    df_filt = filter_by_ff(df, ff)
    df_filt = filter_by_i_value(df_filt, I_VALUE)
    df_filt = filter_exclude_erosion(df_filt)

    grouped = (
        df_filt.groupby(["vel0", "angulo", "posx", "posy"])["id"]
        .nunique()
        .reset_index()
    )
    grouped.rename(columns={"id": "unique_ids"}, inplace=True)

    result = (
        grouped.groupby("vel0")["unique_ids"]
        .agg(avg="mean", std="std", min="min", max="max")
        .reset_index()
    )
    result["lower_std"] = result["avg"] - result["std"].fillna(0)
    result["lower_std"] = result["lower_std"].clip(lower=0)
    result["upper_std"] = result["avg"] + result["std"].fillna(0)
    return result.sort_values("vel0")


def compute_unique_ids_raw_with_positions(
    df: pd.DataFrame, ff: float, i_value: int
) -> dict[float, pd.DataFrame]:
    """Calcula IDs únicos por cada simulación individual con posiciones.

    Args:
        df: DataFrame con datos.
        ff: Factor de fricción.
        i_value: Paso temporal a filtrar.

    Returns:
        Dict con vel0 -> DataFrame con columnas [vel0, angulo, posx, posy, unique_ids].
    """
    df_filt = filter_by_ff(df, ff)
    df_filt = filter_by_i_value(df_filt, i_value)
    df_filt = filter_exclude_erosion(df_filt)

    grouped = (
        df_filt.groupby(["vel0", "angulo", "posx", "posy"])["id"]
        .nunique()
        .reset_index()
    )
    grouped.rename(columns={"id": "unique_ids"}, inplace=True)

    result = {}
    for vel0 in grouped["vel0"].unique():
        result[vel0] = grouped[grouped["vel0"] == vel0][
            ["vel0", "angulo", "posx", "posy", "unique_ids"]
        ].reset_index(drop=True)
    return result


def compute_unique_ids_raw(df: pd.DataFrame, ff: float) -> dict[float, list[int]]:
    """Calcula IDs únicos por cada simulación individual.

    Args:
        df: DataFrame con datos.
        ff: Factor de fricción.

    Returns:
        Dict con vel0 -> lista de IDs únicos por movimiento.
    """
    df_filt = filter_by_ff(df, ff)
    df_filt = filter_by_i_value(df_filt, I_VALUE)
    df_filt = filter_exclude_erosion(df_filt)

    grouped = (
        df_filt.groupby(["vel0", "angulo", "posx", "posy"])["id"]
        .nunique()
        .reset_index()
    )
    grouped.rename(columns={"id": "unique_ids"}, inplace=True)

    result = {}
    for vel0 in grouped["vel0"].unique():
        result[vel0] = grouped[grouped["vel0"] == vel0]["unique_ids"].tolist()
    return result


def compute_cluster_frequency(df: pd.DataFrame, ff: float) -> pd.DataFrame:
    """Calcula frecuencia de tamaños de cluster por ff.

    Args:
        df: DataFrame con datos.
        ff: Factor de fricción.

    Returns:
        DataFrame con frecuencia por tamaño de cluster.
    """
    df_filt = filter_by_ff(df, ff)
    df_filt = filter_by_i_value(df_filt, I_VALUE)
    df_filt = filter_exclude_erosion(df_filt)

    sizes_per_sim = []
    for (_, _, _, _), group in df_filt.groupby(
        ["vel0", "angulo", "posx", "posy"]
    ):
        cluster_sizes = group.groupby("cluster")["id"].nunique()
        for size in cluster_sizes:
            sizes_per_sim.append({"cluster_size": size})

    if not sizes_per_sim:
        return pd.DataFrame()

    df_sizes = pd.DataFrame(sizes_per_sim)

    result = (
        df_sizes.groupby("cluster_size").size().reset_index(name="frecuencia")
    )
    return result.sort_values("cluster_size")


def compute_cluster_sizes(df: pd.DataFrame, ff: float, vel0: float) -> np.ndarray:
    """Extrae tamaños de cluster para un ff y vel0 dados.

    Args:
        df: DataFrame con datos.
        ff: Factor de fricción.
        vel0: Velocidad inicial.

    Returns:
        Array de tamaños de cluster.
    """
    df_filt = filter_by_ff(df, ff)
    df_filt = filter_by_vel0(df_filt, vel0)
    df_filt = filter_by_i_value(df_filt, I_VALUE)
    df_filt = filter_exclude_erosion(df_filt)

    return (
        df_filt.groupby(["angulo", "posx", "posy", "cluster"])["id"]
        .nunique()
        .values
    )


def compute_cluster_size_stats(
    df: pd.DataFrame, ff: float, vel0: float
) -> dict[str, float]:
    """Calcula estadísticas de tamaños de cluster.

    Args:
        df: DataFrame con datos.
        ff: Factor de fricción.
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