"""Módulo de cálculos de clusters.

Contiene funciones para calcular tamaños de clusters, distribuciones y estadísticas.
"""

import numpy as np
import pandas as pd


def filter_erosion(df: pd.DataFrame, ff: float) -> pd.DataFrame:
    """Filtra exclusión de erosión: partículas tipo 1 con z > threshold o vz > 0.

    Args:
        df: DataFrame de datos.
        ff: Factor de llenado.

    Returns:
        DataFrame con partículas erosionadas.
    """
    df = df[(df["ff"] == ff) & (df["i"] == df["i"].max())]
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
    df: pd.DataFrame, ff: float, grid_pairs: list[tuple[float, float]]
) -> dict[float, np.ndarray]:
    """Calcula conteos de erosión por posición de forma optimizada."""
    df_filt = filter_erosion(df, ff)

    # Pre-calcular conteos por (vel0, posx, posy) en un diccionario rápido
    grouped = df_filt.groupby(["vel0", "posx", "posy"])["id"].nunique().to_dict()

    result = {}
    for vel0 in df_filt["vel0"].unique():
        # Busca en el diccionario en O(1), si no existe pone 0
        counts = [grouped.get((vel0, px, py), np.nan) for px, py in grid_pairs]
        result[vel0] = np.array(counts, dtype=float)

    return result


def compute_unique_ids_raw(
    df: pd.DataFrame, ff: float, global_grid: pd.DataFrame, max_data: int
) -> dict[float, np.ndarray]:
    """Calcula IDs únicos por cada simulación individual con padding y NaN.

    Args:
        df: DataFrame con datos.
        ff: Factor de fricción.
        global_grid: DataFrame con la grilla global (posx, posy).
        max_data: Máximo de datos reales entre todas las simulaciones.

    Returns:
        Dict con vel0 -> array de tamaño max_data (datos reales + padding 0).
    """
    df_filt = filter_erosion(df, ff)

    grouped = (
        df_filt.groupby(["vel0", "angulo", "posx", "posy"])["id"]
        .nunique()
        .reset_index()
    )
    grouped.rename(columns={"id": "unique_ids"}, inplace=True)

    grid_set = set(zip(global_grid["posx"], global_grid["posy"]))

    result = {}
    for vel0 in grouped["vel0"].unique():
        sim_data = grouped[grouped["vel0"] == vel0]

        sim_positions = set(zip(sim_data["posx"], sim_data["posy"]))

        has_data = sim_positions & grid_set
        missing = grid_set - sim_positions

        values = []
        for posx, posy in has_data:
            val = sim_data[
                (sim_data["posx"] == posx) & (sim_data["posy"] == posy)
            ]["unique_ids"].values[0]
            values.append(val)

        padding_count = max(0, max_data - len(values))
        values.extend([0] * padding_count)

        missing_positions = [(px, py, np.nan) for px, py in missing]
        if missing_positions:
            arr = np.array(values + [np.nan] * len(missing_positions))
        else:
            arr = np.array(values)

        result[vel0] = arr

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
