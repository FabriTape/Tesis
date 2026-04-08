"""Módulo de cálculos cinemáticos.

Contiene funciones para calcular energía cinética, ángulos de velocidad,
penetración y otros parámetros cinemáticos.
"""

import pandas as pd
import numpy as np

from data_loader import (
    filter_by_ff,
    filter_by_vel0,
    filter_by_i_value,
    filter_exclude_erosion,
    filter_is_erosion,
    filter_penetration,
)
from constants import I_VALUE


def compute_kinetic_energy(df: pd.DataFrame, ff: float) -> pd.DataFrame:
    """Calcula energía cinética promedio K/m del CM de clusters.

    Args:
        df: DataFrame con datos.
        ff: Factor de fricción.

    Returns:
        DataFrame con estadísticas de energía por vel0.
    """
    df_filt = filter_by_ff(df, ff)
    df_filt = filter_by_i_value(df_filt, I_VALUE)
    df_filt = filter_exclude_erosion(df_filt)

    cm_data = (
        df_filt.groupby(["vel0", "angulo", "posx", "posy", "cluster"])
        .agg(
            avg_vx=("vx", "mean"),
            avg_vy=("vy", "mean"),
            avg_vz=("vz", "mean"),
        )
        .reset_index()
    )

    cm_data["ek"] = 0.5 * (
        cm_data["avg_vx"] ** 2 + cm_data["avg_vy"] ** 2 + cm_data["avg_vz"] ** 2
    )

    result = (
        cm_data.groupby("vel0")["ek"]
        .agg(avg="mean", std="std", min="min", max="max")
        .reset_index()
    )
    result["lower_std"] = (result["avg"] - result["std"].fillna(0)).clip(lower=0)
    result["upper_std"] = result["avg"] + result["std"].fillna(0)
    return result.sort_values("vel0")


def compute_velocity_angle(df: pd.DataFrame, ff: float, vel0: float) -> pd.DataFrame:
    """Calcula distribución del ángulo de velocidad del CM por cluster.

    Args:
        df: DataFrame con datos.
        ff: Factor de fricción.
        vel0: Velocidad inicial.

    Returns:
        DataFrame con distribución por bins de ángulo.
    """
    df_filt = filter_by_ff(df, ff)
    df_filt = filter_by_vel0(df_filt, vel0)
    df_filt = filter_by_i_value(df_filt, I_VALUE)
    df_filt = filter_exclude_erosion(df_filt)

    cm_data = (
        df_filt.groupby(["angulo", "posx", "posy", "cluster"])
        .agg(
            avg_vx=("vx", "mean"),
            avg_vy=("vy", "mean"),
            avg_vz=("vz", "mean"),
        )
        .reset_index()
    )

    cm_data["angle"] = np.degrees(
        np.arctan2(
            cm_data["avg_vz"], np.sqrt(cm_data["avg_vx"] ** 2 + cm_data["avg_vy"] ** 2)
        )
    )
    cm_data["bin"] = (cm_data["angle"] // 10) * 10
    freq = cm_data.groupby("bin").size().reset_index(name="frecuencia")
    return freq.sort_values("bin")


def compute_penetration_depth(df: pd.DataFrame, ff: float) -> pd.DataFrame:
    """Calcula profundidad de penetración promedio.

    Args:
        df: DataFrame con datos.
        ff: Factor de fricción.

    Returns:
        DataFrame con estadísticas de penetración por vel0.
    """
    df_filt = filter_by_ff(df, ff)
    df_filt = filter_by_i_value(df_filt, I_VALUE)
    df_filt = filter_penetration(df_filt).copy()

    df_filt["penetracion"] = abs(df_filt["z"] - 1e-4) * 1e6

    result = (
        df_filt.groupby("vel0")["penetracion"]
        .agg(avg="mean", std="std", min="min", max="max")
        .reset_index()
    )
    result["lower_std"] = (result["avg"] - result["std"].fillna(0)).clip(lower=0)
    result["upper_std"] = result["avg"] + result["std"].fillna(0)
    return result.sort_values("vel0")


GRID_POSITIONS = [-1.52, 0.0, 1.52, 3.04]


def compute_heatmap_data(df: pd.DataFrame, ff: float, vel0: float) -> pd.DataFrame:
    """Calcula partículas erosionadas por posición usando el último time step.

    Args:
        df: DataFrame con datos.
        ff: Factor de fricción.
        vel0: Velocidad inicial.

    Returns:
        DataFrame pivot con IDs por posición, índices y columnas normalizados por 1.52.
    """
    df_filt = filter_by_ff(df, ff)
    df_filt = filter_by_vel0(df_filt, vel0)
    max_i = df_filt["i"].max()
    df_filt = filter_by_i_value(df_filt, max_i)

    grouped = df_filt.groupby(["posx", "posy"]).size().reset_index(name="id")

    pivot = grouped.pivot(index="posy", columns="posx", values="id")

    full_index = pd.Index(GRID_POSITIONS, name="posy")
    full_columns = pd.Index(GRID_POSITIONS, name="posx")

    pivot = pivot.reindex(index=full_index, columns=full_columns)

    pivot.index = (pivot.index / 1.52).round(2)
    pivot.columns = (pivot.columns / 1.52).round(2)

    return pivot


def compute_cluster_km(df: pd.DataFrame, ff: float, vel0: float) -> pd.DataFrame:
    """Calcula K/m (energía cinética por unidad de masa) del CM de cada cluster.

    Args:
        df: DataFrame con datos.
        ff: Factor de fricción.
        vel0: Velocidad inicial.

    Returns:
        DataFrame con tamaño de cluster y K/m del CM.
    """
    df_filt = filter_by_ff(df, ff)
    df_filt = filter_by_vel0(df_filt, vel0)
    df_filt = filter_by_i_value(df_filt, I_VALUE)
    df_filt = filter_exclude_erosion(df_filt)

    cluster_data = (
        df_filt.groupby(["angulo", "posx", "posy", "cluster"])
        .agg(
            avg_vx=("vx", "mean"),
            avg_vy=("vy", "mean"),
            avg_vz=("vz", "mean"),
            cluster_size=("id", "nunique"),
        )
        .reset_index()
    )

    cluster_data["km"] = 0.5 * (
        cluster_data["avg_vx"] ** 2
        + cluster_data["avg_vy"] ** 2
        + cluster_data["avg_vz"] ** 2
    )

    return cluster_data[["cluster_size", "km"]]


def compute_cluster_km_binned(
    df: pd.DataFrame, ff: float, vel0: float, n_bins: int = 10
) -> pd.DataFrame:
    """Calcula K/m vs tamaño de cluster con log-binning.

    Args:
        df: DataFrame con datos.
        ff: Factor de fricción.
        vel0: Velocidad inicial.
        n_bins: Número de bins logarítmicos (default 10).

    Returns:
        DataFrame con columnas: s_center, km_mean, km_std, n_clusters.
    """
    raw_data = compute_cluster_km(df, ff, vel0)

    if raw_data.empty or len(raw_data) < 2:
        return pd.DataFrame()

    s_values = raw_data["cluster_size"].values
    km_values = raw_data["km"].values

    s_min, s_max = s_values.min(), s_values.max()

    if s_min <= 0 or s_max <= 0:
        return pd.DataFrame()

    bin_edges = np.logspace(np.log10(s_min), np.log10(s_max), n_bins + 1)

    results = []

    for i in range(len(bin_edges) - 1):
        s_low, s_high = bin_edges[i], bin_edges[i + 1]

        mask = (s_values >= s_low) & (s_values < s_high)
        if not mask.any():
            mask = (s_values >= s_low) & (s_values <= s_high)

        if not mask.any():
            continue

        km_in_bin = km_values[mask]

        s_center = np.sqrt(s_low * s_high)
        km_mean = np.mean(km_in_bin)
        n_clusters = len(km_in_bin)

        km_std = np.std(km_in_bin, ddof=1) if n_clusters > 1 else np.nan

        results.append({
            "s_center": s_center,
            "km_mean": km_mean,
            "km_std": km_std,
            "n_clusters": n_clusters,
        })

    if not results:
        return pd.DataFrame()

    binned_df = pd.DataFrame(results)

    valid_stds = binned_df["km_std"].dropna()
    if valid_stds.empty:
        return pd.DataFrame()

    max_std = valid_stds.max()

    binned_df.loc[binned_df["km_std"].isna(), "km_std"] = max_std

    return binned_df


def compute_cluster_rg(df: pd.DataFrame, ff: float, vel0: float) -> pd.DataFrame:
    """Calcula el radio de giro Rg para cada cluster.

    Args:
        df: DataFrame con datos.
        ff: Factor de fricción.
        vel0: Velocidad inicial.

    Returns:
        DataFrame con columnas: cluster_size, rg.
    """
    df_filt = filter_by_ff(df, ff)
    df_filt = filter_by_vel0(df_filt, vel0)
    df_filt = filter_by_i_value(df_filt, I_VALUE)
    df_filt = filter_exclude_erosion(df_filt)

    cols_grupo = ["vel0", "angulo", "posx", "posy", "cluster"]

    df_cm = (
        df_filt.groupby(cols_grupo)
        .agg(
            x_cm=("x", "mean"),
            y_cm=("y", "mean"),
            z_cm=("z", "mean"),
            cluster_size=("id", "nunique"),
        )
        .reset_index()
    )

    df_merged = df_filt.merge(
        df_cm[cols_grupo + ["x_cm", "y_cm", "z_cm", "cluster_size"]],
        on=cols_grupo,
        how="left",
    )

    df_merged["dr2"] = (
        (df_merged["x"] - df_merged["x_cm"]) ** 2
        + (df_merged["y"] - df_merged["y_cm"]) ** 2
        + (df_merged["z"] - df_merged["z_cm"]) ** 2
    )

    rg_data = (
        df_merged.groupby(cols_grupo)
        .agg(
            cluster_size=("cluster_size", "first"),
            dr2_mean=("dr2", "mean"),
        )
        .reset_index()
    )

    rg_data["rg"] = np.sqrt(rg_data["dr2_mean"])

    rg_data = rg_data[rg_data["cluster_size"] > 1]

    return rg_data[["cluster_size", "rg"]]


def compute_rg_binned(
    df: pd.DataFrame, ff: float, vel0: float, n_bins: int = 12
) -> pd.DataFrame:
    """Calcula Rg vs tamaño de cluster con log-binning.

    Args:
        df: DataFrame con datos.
        ff: Factor de fricción.
        vel0: Velocidad inicial.
        n_bins: Número de bins logarítmicos (default 12).

    Returns:
        DataFrame con columnas: s_center, log_rg_mean, log_rg_std, n_clusters.
    """
    raw_data = compute_cluster_rg(df, ff, vel0)

    if raw_data.empty or len(raw_data) < 2:
        return pd.DataFrame()

    s_values = raw_data["cluster_size"].values
    rg_values = raw_data["rg"].values

    log_rg = np.log10(rg_values)

    s_min, s_max = s_values.min(), s_values.max()

    if s_min <= 0 or s_max <= 0:
        return pd.DataFrame()

    bin_edges = np.logspace(np.log10(s_min), np.log10(s_max), n_bins + 1)

    results = []

    for i in range(len(bin_edges) - 1):
        s_low, s_high = bin_edges[i], bin_edges[i + 1]

        mask = (s_values >= s_low) & (s_values < s_high)
        if not mask.any():
            mask = (s_values >= s_low) & (s_values <= s_high)

        if not mask.any():
            continue

        log_rg_in_bin = log_rg[mask]

        s_center = np.sqrt(s_low * s_high)
        log_rg_mean = np.mean(log_rg_in_bin)
        n_clusters = len(log_rg_in_bin)

        if n_clusters == 1:
            log_rg_std = 0.0
        else:
            log_rg_std = np.std(log_rg_in_bin, ddof=1)

        results.append({
            "s_center": s_center,
            "log_rg_mean": log_rg_mean,
            "log_rg_std": log_rg_std,
            "n_clusters": n_clusters,
        })

    if not results:
        return pd.DataFrame()

    binned_df = pd.DataFrame(results)

    if len(binned_df) > 0:
        max_std = binned_df[binned_df["n_clusters"] > 1]["log_rg_std"].max()
        if pd.isna(max_std) or max_std == 0:
            max_std = binned_df["log_rg_std"].max()
        binned_df["log_rg_std"] = binned_df["log_rg_std"].replace(0, max_std)

    return binned_df