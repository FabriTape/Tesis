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


def compute_heatmap_data(df: pd.DataFrame, ff: float, vel0: float) -> pd.DataFrame:
    """Calcula IDs únicos erosionadas por posición (posx, posy).

    Args:
        df: DataFrame con datos.
        ff: Factor de fricción.
        vel0: Velocidad inicial.

    Returns:
        DataFrame pivot con IDs por posición, índices y columnas normalizados por 1.52.
    """
    df_filt = filter_by_ff(df, ff)
    df_filt = filter_by_vel0(df_filt, vel0)
    df_filt = filter_by_i_value(df_filt, I_VALUE)
    df_filt = filter_exclude_erosion(df_filt)

    grouped = df_filt.groupby(["posx", "posy"])["id"].nunique().reset_index()

    pivot = grouped.pivot(index="posy", columns="posx", values="id").fillna(0)

    pivot.index = (pivot.index / 1.52).round(2)
    pivot.columns = (pivot.columns / 1.52).round(2)

    return pivot