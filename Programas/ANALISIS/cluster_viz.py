"""Módulo de visualizaciones de distribución de clusters.

Genera gráficos de frecuencia y CCDF usando solo MLE + estadístico KS.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.optimize import curve_fit

from cluster_calc import compute_cluster_frequency, compute_cluster_sizes
from constants import I_VALUE, COLORS
from kinematics_calc import compute_cluster_km, compute_cluster_km_binned, compute_cluster_rg, compute_rg_binned
from utils import compute_alpha_mle, compute_ccdf_empirical, compute_ccdf_from_pdf


def power_law(x: np.ndarray, C: float, beta: float) -> np.ndarray:
    """Modelo Power Law: f(s) = C * s^(-beta)"""
    return C * np.power(x, -beta)


def exponential(x: np.ndarray, C: float, lam: float) -> np.ndarray:
    """Modelo Exponential: f(s) = C * exp(-lambda * s)"""
    return C * np.exp(-lam * x)


def plot_cluster_frequency(df: pd.DataFrame, ff: float) -> Figure:
    """Genera gráfico de barras de frecuencia vs tamaño de cluster."""
    data = compute_cluster_frequency(df, ff)
    if data.empty:
        return Figure()

    fs = 14
    sizes = data["cluster_size"].values
    freq = data["frecuencia"].values

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(sizes)), freq, color=COLORS["primary"], alpha=0.7)

    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([str(int(s)) for s in sizes])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Tamaño de cluster", fontsize=fs)
    ax.set_ylabel("Frecuencia", fontsize=fs)
    ax.set_title(f"Frecuencia vs Tamaño (ff={ff}, i={I_VALUE})", fontsize=fs * 0.8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def _ks_statistic(data: np.ndarray, alpha: float, xmin: int) -> float:
    """Calcula estadístico KS."""
    sizes_sorted = np.sort(data)
    n = len(sizes_sorted)

    ccdf_emp = 1 - np.arange(1, n + 1) / n
    ccdf_theory = (sizes_sorted / xmin) ** (-alpha + 1)
    ccdf_theory = np.clip(ccdf_theory, 1e-15, 1.0)

    return np.max(np.abs(ccdf_emp - ccdf_theory))


def plot_cluster_ccdf(df: pd.DataFrame, ff: float, vel0: float) -> list[Figure]:
    """Genera gráfico CCDF log-log usando solo MLE + KS."""
    raw_sizes = compute_cluster_sizes(df, ff, vel0)

    if len(raw_sizes) < 2 or len(np.unique(raw_sizes)) < 2:
        return []

    fs = 14
    xm = int(raw_sizes.min())
    alpha = compute_alpha_mle(raw_sizes, xm)
    D = _ks_statistic(raw_sizes, alpha, xm)

    fig, ax = plt.subplots(figsize=(10, 6))

    unique_sizes, ccdf_empirical = compute_ccdf_empirical(raw_sizes)
    ax.bar(unique_sizes, ccdf_empirical, color=COLORS["primary"], alpha=0.7, label="Datos")

    x_int = np.arange(1, int(raw_sizes.max()) + 1)

    pdf_pl = x_int ** (-alpha)
    pdf_pl = pdf_pl / np.sum(pdf_pl)
    ccdf_pl = compute_ccdf_from_pdf(pdf_pl)
    ax.step(x_int, ccdf_pl, where="post", color=COLORS["power_law"],
            label=f"PL (α={alpha:.2f}, D={D:.3f})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Tamaño de cluster", fontsize=fs)
    ax.set_ylabel("CCDF", fontsize=fs)
    ax.set_title(f"CCDF (ff={ff}, v0={vel0})", fontsize=fs * 0.8)
    ax.legend(fontsize=fs * 0.6, loc="lower left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return [fig]


def _create_table(data: list[list[str]], titulo: str = "") -> Figure:
    """Crea una tabla."""
    fs = 10
    num_cols = len(data[0]) if data else 2
    fig, ax = plt.subplots(figsize=(10, max(2, len(data) * 0.4)))
    ax.axis("off")

    col_widths = [0.3, 0.6] if num_cols == 2 else [0.25, 0.35, 0.35]
    table = ax.table(cellText=data, loc="center", cellLoc="center", colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(fs)
    table.scale(1.2, 1.2)

    for i in range(len(data)):
        for j in range(num_cols):
            cell = table[(i, j)]
            cell.set_edgecolor("#cccccc")
            if i == 0:
                cell.set_facecolor("#3b82f6")
                cell.set_text_props(color="white", weight="bold")

    if titulo:
        ax.set_title(titulo, fontsize=fs + 2, pad=10)
    plt.tight_layout()
    return fig


def fit_km_curves(binned_df: pd.DataFrame) -> dict:
    """Ajusta modelos a datos binned."""
    x = binned_df["s_center"].values
    y = binned_df["km_mean"].values
    sigma = binned_df["km_std"].values
    sigma = np.where(sigma <= 0, 1e-10, sigma)

    results = {}

    try:
        popt, _ = curve_fit(power_law, x, y, sigma=sigma, absolute_sigma=True, maxfev=5000)
        results["power_law"] = {"C": popt[0], "beta": popt[1]}
    except Exception:
        results["power_law"] = None

    try:
        popt, _ = curve_fit(exponential, x, y, sigma=sigma, absolute_sigma=True, maxfev=5000)
        results["exponential"] = {"C": popt[0], "lam": popt[1]}
    except Exception:
        results["exponential"] = None

    return results


def plot_cluster_km_scatter(df: pd.DataFrame, ff: float, vel0: float) -> Figure:
    """Genera gráfico log-log de K/m vs tamaño."""
    data = compute_cluster_km(df, ff, vel0)
    if data.empty or len(data) < 2:
        return Figure()

    fs = 14
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(data["cluster_size"], data["km"], color=COLORS["primary"], alpha=0.5, s=30)

    ax.set_xscale("log")
    ax.set_yscale("log")

    km = (vel0**2) / 2
    ax.set_xlabel("Tamaño de cluster", fontsize=fs)
    ax.set_ylabel("K/m cluster", fontsize=fs)
    ax.set_title(f"K/m vs Tamaño (ff={ff}, K/m={km:.1f})", fontsize=fs * 0.8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_cluster_km_logfit(df: pd.DataFrame, ff: float, vel0: float) -> list[Figure]:
    """Genera gráfico log-log de K/m con ajustes."""
    raw_data = compute_cluster_km(df, ff, vel0)
    binned_data = compute_cluster_km_binned(df, ff, vel0, n_bins=10)

    if raw_data.empty or binned_data.empty:
        return []

    fs = 14
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(raw_data["cluster_size"], raw_data["km"], color=COLORS["primary"],
             alpha=0.3, s=20, label="Datos")

    ax.errorbar(binned_data["s_center"], binned_data["km_mean"], yerr=binned_data["km_std"],
              fmt="o", color=COLORS["power_law"], markersize=8, capsize=4,
              label="Binning")

    x_fit = np.logspace(np.log10(binned_data["s_center"].min()),
                    np.log10(binned_data["s_center"].max()), 100)

    fit_results = fit_km_curves(binned_data)

    if fit_results.get("power_law"):
        pl = fit_results["power_law"]
        ax.plot(x_fit, power_law(x_fit, pl["C"], pl["beta"]), "-",
               color=COLORS["power_law"], linewidth=2, label=f"PL: β={pl['beta']:.3f}")

    km = (vel0**2) / 2
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Tamaño de cluster", fontsize=fs)
    ax.set_ylabel("K/m cluster", fontsize=fs)
    ax.set_title(f"K/m vs Tamaño (ff={ff}, K/m={km:.1f})", fontsize=fs * 0.8)
    ax.legend(fontsize=fs * 0.6, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    table_data = [["Modelo", "β"]]
    if fit_results.get("power_law"):
        table_data.append(["PL", f"{fit_results['power_law']['beta']:.3f}"])

    return [fig, _create_table(table_data, "Parámetros")]


def fit_rg_linear(binned_df: pd.DataFrame) -> dict:
    """Ajusta regresión lineal a log10(Rg) vs log10(s)."""
    if binned_df.empty or len(binned_df) < 2:
        return {}

    x = np.log10(binned_df["s_center"].values)
    y = binned_df["log_rg_mean"].values

    m, C = np.polyfit(x, y, 1)
    y_pred = m * x + C
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return {"df": 1 / m if m != 0 else 0, "intercept": C, "r_squared": r_squared}


def plot_cluster_rg_logfit(df: pd.DataFrame, ff: float, vel0: float) -> list[Figure]:
    """Genera gráfico log-log de Rg vs tamaño."""
    raw_data = compute_cluster_rg(df, ff, vel0)
    binned_data = compute_rg_binned(df, ff, vel0, n_bins=12)

    if raw_data.empty or binned_data.empty:
        return []

    fs = 14
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(raw_data["cluster_size"], raw_data["rg"], color=COLORS["primary"],
             alpha=0.3, s=20, label="Datos")

    ax.errorbar(binned_data["s_center"], 10 ** binned_data["log_rg_mean"],
              yerr=10 ** binned_data["log_rg_std"], fmt="o", color=COLORS["power_law"],
              markersize=8, capsize=4, label="Binning")

    fit = fit_rg_linear(binned_data)

    x_fit = np.logspace(np.log10(binned_data["s_center"].min()),
                      np.log10(binned_data["s_center"].max()), 100)
    y_fit = 10 ** (fit.get("slope", 1) * np.log10(x_fit) + fit.get("intercept", 0))
    ax.plot(x_fit, y_fit, "-", color=COLORS["truncated"], linewidth=2,
            label=f"df={fit.get('df', 0):.2f}")

    km = (vel0**2) / 2
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Tamaño de cluster", fontsize=fs)
    ax.set_ylabel("Rg", fontsize=fs)
    ax.set_title(f"Rg vs Tamaño (ff={ff}, K/m={km:.1f})", fontsize=fs * 0.8)
    ax.legend(fontsize=fs * 0.6, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    table_data = [["Parámetro", "Valor"], ["df", f"{fit.get('df', 0):.2f}"], ["R2", f"{fit.get('r_squared', 0):.3f}"]]
    return [fig, _create_table(table_data, "Dimensión Fractal")]