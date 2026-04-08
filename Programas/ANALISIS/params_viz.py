"""Módulo de análisis de parámetros vs energía.

Genera gráficos de evolución de parámetros de distribuciones vs K/m usando MLE.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from data_loader import filter_by_ff, filter_by_vel0, filter_by_i_value, filter_exclude_erosion
from constants import I_VALUE, COLORS
from utils import compute_alpha_mle


def compute_params_mle(df: pd.DataFrame, ff: float) -> pd.DataFrame:
    """Calcula parámetros MLE de distribuciones vs K/m.

    Args:
        df: DataFrame con datos.
        ff: Factor de fricción.

    Returns:
        DataFrame con parámetros por vel0.
    """
    valores_vel0 = sorted(df["vel0"].unique())
    results = []

    for vel0 in valores_vel0:
        df_filt = filter_by_ff(df, ff)
        df_filt = filter_by_vel0(df_filt, vel0)
        df_filt = filter_by_i_value(df_filt, I_VALUE)
        df_filt = filter_exclude_erosion(df_filt)

        if df_filt.empty:
            continue

        sizes = df_filt.groupby(["posx", "posy", "cluster"])["id"].nunique().values

        if len(np.unique(sizes)) < 2:
            continue

        km = (vel0**2) / 2
        xm = int(sizes.min())

        alpha = compute_alpha_mle(sizes, xm)

        ccdf_emp = 1 - np.arange(1, len(sizes) + 1) / len(sizes)
        ccdf_pl = (sizes[np.argsort(sizes)] / xm) ** (-alpha + 1)
        ccdf_pl = np.clip(ccdf_pl, 1e-15, 1.0)

        D = np.max(np.abs(ccdf_emp[:len(ccdf_pl)] - ccdf_pl[:len(ccdf_emp)]))

        results.append({
            "vel0": vel0,
            "km": km,
            "alpha": alpha,
            "xmin": xm,
            "D": D,
        })

    return pd.DataFrame(results) if results else pd.DataFrame()


def plot_params_vs_energy(df: pd.DataFrame, ff: float) -> Figure:
    """Genera gráfico de parámetros vs K/m.

    Args:
        df: DataFrame con datos.
        ff: Factor de fricción.

    Returns:
        Figura matplotlib.
    """
    res_df = compute_params_mle(df, ff)
    if res_df.empty:
        return plt.figure()

    res_df = res_df.sort_values("km")
    fs = 14
    marker = "o-"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(res_df["km"], res_df["alpha"], marker, color=COLORS["power_law"], markersize=8)
    axes[0].set_xlabel("K/m", fontsize=fs)
    axes[0].set_ylabel(r"$\alpha$", fontsize=fs)
    axes[0].set_title(r"Exponente $\alpha$ (MLE)", fontsize=fs * 0.9)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(res_df["km"], res_df["xmin"], marker, color=COLORS["primary"], markersize=8)
    axes[1].set_xlabel("K/m", fontsize=fs)
    axes[1].set_ylabel(r"$x_{min}$", fontsize=fs)
    axes[1].set_title(r"$x_{min}$", fontsize=fs * 0.9)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(res_df["km"], res_df["D"], marker, color=COLORS["exponential"], markersize=8)
    axes[2].set_xlabel("K/m", fontsize=fs)
    axes[2].set_ylabel("D", fontsize=fs)
    axes[2].set_title("Estadístico KS", fontsize=fs * 0.9)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"Parámetros MLE vs K/m (ff={ff}, i={I_VALUE})", fontsize=fs)
    plt.tight_layout()
    return fig