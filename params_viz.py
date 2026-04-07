"""Módulo de análisis de parámetros vs energía.

Genera gráficos de evolución de parámetros de distribuciones vs K/m.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import powerlaw
from matplotlib.figure import Figure

from data_loader import filter_by_ff, filter_by_vel0, filter_by_i_value, filter_exclude_erosion
from constants import I_VALUE, COLORS
from utils import compute_alpha_mle


def compute_params_vs_energy(df: pd.DataFrame, ff: float) -> pd.DataFrame:
    """Calcula parámetros de distribuciones vs K/m.

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

        raw_sizes = (
            df_filt.groupby(["angulo", "posx", "posy", "cluster"])["id"]
            .nunique()
            .values
        )

        if len(np.unique(raw_sizes)) < 2:
            continue

        cm_data = (
            df_filt.groupby(["angulo", "posx", "posy", "cluster"])
            .agg(
                avg_vx=("vx", "mean"),
                avg_vy=("vy", "mean"),
                avg_vz=("vz", "mean"),
            )
            .reset_index()
        )

        if cm_data.empty:
            continue

        ek = 0.5 * (cm_data["avg_vx"]**2 + cm_data["avg_vy"]**2 + cm_data["avg_vz"]**2)
        km = ek.mean()

        # MLE para alpha
        xmin_options = [1, int(raw_sizes.min())]
        best_alpha, best_xmin, best_D = None, None, float("inf")

        for xm in xmin_options:
            alpha = compute_alpha_mle(raw_sizes, xm)
            try:
                fit_test = powerlaw.Fit(raw_sizes, discrete=True, verbose=False, xmin=xm)
                D = fit_test.power_law.D
                if D < best_D:
                    best_D, best_alpha, best_xmin = D, alpha, xm
            except Exception:
                continue

        if best_alpha is None or best_xmin is None:
            continue

        try:
            fit = powerlaw.Fit(
                raw_sizes, discrete=True, verbose=False, xmin=best_xmin,
                parameter_ranges={
                    "power_law": {"alpha": (1, 10)},
                    "truncated_power_law": {"alpha": (1, 10), "Lambda": (1e-10, 10)},
                    "exponential": {"Lambda": (1e-10, 10)},
                    "lognormal": {"mu": (-1, 10), "sigma": (1e-10, 10)},
                },
            )

            results.append({
                "vel0": vel0,
                "km": km,
                "alpha_pl": best_alpha,
                "xmin_pl": best_xmin,
                "alpha_tpl": fit.truncated_power_law.alpha,
                "lambda_tpl": fit.truncated_power_law.Lambda,
                "lambda_exp": fit.exponential.Lambda,
                "mu_ln": fit.lognormal.mu,
                "sigma_ln": fit.lognormal.sigma,
            })
        except Exception:
            continue

    return pd.DataFrame(results) if results else pd.DataFrame()


def plot_params_vs_energy(df: pd.DataFrame, ff: float) -> Figure:
    """Genera gráfico de parámetros de distribuciones vs K/m.

    Args:
        df: DataFrame con datos.
        ff: Factor de fricción.

    Returns:
        Figura matplotlib.
    """
    res_df = compute_params_vs_energy(df, ff)
    if res_df.empty:
        return plt.figure()

    res_df = res_df.sort_values("km")
    fs = 1.8 * 12

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # α PL
    axes[0, 0].plot(res_df["km"], res_df["alpha_pl"], "o-", color=COLORS["power_law"], markersize=8)
    axes[0, 0].set_xlabel("K/m", fontsize=fs)
    axes[0, 0].set_ylabel(r"$\alpha$", fontsize=fs)
    axes[0, 0].set_title("Power Law: α", fontsize=fs * 0.9)
    axes[0, 0].grid(True, alpha=0.3)

    # xmin PL
    axes[0, 1].plot(res_df["km"], res_df["xmin_pl"], "o-", color=COLORS["power_law"], markersize=8)
    axes[0, 1].set_xlabel("K/m", fontsize=fs)
    axes[0, 1].set_ylabel(r"$x_{min}$", fontsize=fs)
    axes[0, 1].set_title("Power Law: xmin", fontsize=fs * 0.9)
    axes[0, 1].grid(True, alpha=0.3)

    # α TPL
    axes[0, 2].plot(res_df["km"], res_df["alpha_tpl"], "o-", color=COLORS["truncated"], markersize=8)
    axes[0, 2].set_xlabel("K/m", fontsize=fs)
    axes[0, 2].set_ylabel(r"$\alpha$", fontsize=fs)
    axes[0, 2].set_title("Truncated PL: α", fontsize=fs * 0.9)
    axes[0, 2].grid(True, alpha=0.3)

    # λ TPL
    axes[1, 0].plot(res_df["km"], res_df["lambda_tpl"], "o-", color=COLORS["truncated"], markersize=8)
    axes[1, 0].set_xlabel("K/m", fontsize=fs)
    axes[1, 0].set_ylabel(r"$\lambda$", fontsize=fs)
    axes[1, 0].set_title("Truncated PL: λ", fontsize=fs * 0.9)
    axes[1, 0].grid(True, alpha=0.3)

    # λ Exponential
    axes[1, 1].plot(res_df["km"], res_df["lambda_exp"], "o-", color=COLORS["exponential"], markersize=8)
    axes[1, 1].set_xlabel("K/m", fontsize=fs)
    axes[1, 1].set_ylabel(r"$\lambda$", fontsize=fs)
    axes[1, 1].set_title("Exponential: λ", fontsize=fs * 0.9)
    axes[1, 1].grid(True, alpha=0.3)

    # Lognormal
    valid_ln = res_df[res_df["mu_ln"] >= -1]
    axes[1, 2].plot(valid_ln["km"], valid_ln["mu_ln"], "o-", color=COLORS["lognormal"], markersize=8, label="μ")
    axes[1, 2].plot(valid_ln["km"], valid_ln["sigma_ln"], "s-", color=COLORS["secondary"], markersize=8, label="σ")
    axes[1, 2].set_xlabel("K/m", fontsize=fs)
    axes[1, 2].set_ylabel("Valor", fontsize=fs)
    axes[1, 2].set_title("Lognormal (μ ≥ -1)", fontsize=fs * 0.9)
    axes[1, 2].legend(fontsize=fs * 0.6)
    axes[1, 2].grid(True, alpha=0.3)

    fig.suptitle(f"Parámetros vs K/m (ff={ff}, i={I_VALUE})", fontsize=fs)
    plt.tight_layout()
    return fig