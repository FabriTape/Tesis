"""Módulo de cálculos de clusters.

Contiene funciones para calcular tamaños de clusters, distribuciones y estadísticas.
"""

import numpy as np
import pandas as pd
from constants import I_VALUE


def filter_erosion(df: pd.DataFrame, ff: float) -> pd.DataFrame:
    """Filtra exclusión de erosión: partículas tipo 1 con z > threshold o vz > 0.

    Args:
        df: DataFrame de datos.
        ff: Factor de llenado.

    Returns:
        DataFrame con partículas erosionadas.
    """
    df = df[(df["ff"] == ff) & (df["i"] == I_VALUE)]
    return df[((df["type"] == 1) & ((df["z"] > 1e-4) | (df["vz"] > 0)))]


def compute_unique_ids_by_velocity(df: pd.DataFrame, ff: float) -> pd.DataFrame:
    """Calcula promedio de IDs únicos por velocidad.

    Args:
        df: DataFrame con datos.
        ff: Factor de fricción.

    Returns:
        DataFrame con estadísticas por vel0.
    """
    df_filt = filter_erosion(df, ff)

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
    df: pd.DataFrame, ff: float
) -> dict[float, pd.DataFrame]:
    """Calcula IDs únicos por cada simulación individual con posiciones.

    Args:
        df: DataFrame con datos.
        ff: Factor de fricción.

    Returns:
        Dict con vel0 -> DataFrame con columnas [vel0, angulo, posx, posy, unique_ids].
    """
    df_filt = filter_erosion(df, ff)

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


def compute_erosion_counts_with_zeros(
    df: pd.DataFrame,
    ff: float,
    grid_pairs: list[tuple[float, float]],
) -> dict[float, np.ndarray]:
    """Calcula conteos de erosión por posición, incluyendo ceros.

    Args:
        df: DataFrame completo.
        ff: Factor de fricción.
        grid_pairs: Lista de posiciones válidas (posx, posy).

    Returns:
        dict[vel0] = array numpy con conteos (algunos pueden ser 0).
    """

    df_filt = filter_erosion(df, ff)

    existing_pairs = set(zip(df_filt["posx"], df_filt["posy"]))

    result = {}
    for vel0 in df_filt["vel0"].unique():
        df_vel = df_filt[df_filt["vel0"] == vel0]

        counts = []
        for posx, posy in grid_pairs:
            if (posx, posy) in existing_pairs:
                count = df_vel[(df_vel["posx"] == posx) & (df_vel["posy"] == posy)][
                    "id"
                ].nunique()
                counts.append(count)
            else:
                counts.append(0)

        result[vel0] = np.array(counts, dtype=float)

    return result


def compute_unique_ids_raw(df: pd.DataFrame, ff: float) -> dict[float, list[int]]:
    """Calcula IDs únicos por cada simulación individual.

    Args:
        df: DataFrame con datos.
        ff: Factor de fricción.

    Returns:
        Dict con vel0 -> lista de IDs únicos por movimiento.
    """
    df_filt = filter_erosion(df, ff)

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
    df_filt = filter_erosion(df, ff)

    sizes_per_sim = []
    for (_, _, _, _), group in df_filt.groupby(["vel0", "angulo", "posx", "posy"]):
        cluster_sizes = group.groupby("cluster")["id"].nunique()
        for size in cluster_sizes:
            sizes_per_sim.append({"cluster_size": size})

    if not sizes_per_sim:
        return pd.DataFrame()

    df_sizes = pd.DataFrame(sizes_per_sim)

    result = df_sizes.groupby("cluster_size").size().reset_index(name="frecuencia")
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
    df_filt = df[df["vel0"] == vel0]
    df_filt = filter_erosion(df_filt, ff)

    return df_filt.groupby(["angulo", "posx", "posy", "cluster"])["id"].nunique().values


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
