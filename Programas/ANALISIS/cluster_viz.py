"""Módulo de visualizaciones de distribución de clusters.

Genera gráficos de frecuencia y CCDF para análisis de distribución de tamaños.
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import powerlaw
from matplotlib.figure import Figure
from scipy.optimize import curve_fit
from uncertainties import ufloat

from cluster_calc import compute_cluster_frequency, compute_cluster_sizes
from constants import I_VALUE, COLORS
from kinematics_calc import (
    compute_cluster_km,
    compute_cluster_km_binned,
    compute_cluster_rg,
    compute_rg_binned,
)
from utils import compute_alpha_mle, compute_ccdf_empirical, compute_ccdf_from_pdf


def format_uncertainty(val: float, err: float) -> str:
    """Formatea valor con incertidumbre usando 1 cifra significativa."""
    if err <= 0 or np.isnan(err) or np.isinf(err):
        return f"{val:.1f}"
    
    try:
        uval = ufloat(val, err)
        return f"{uval:.1u}"  # 1 significant figure in uncertainty
    except Exception:
        return f"{val:.1f}±{err:.1f}"


def power_law(s: np.ndarray, C: float, beta: float) -> np.ndarray:
    """Modelo Power Law: f(s) = C * s^(-beta)"""
    return C * np.power(s, -beta)


def exponential(s: np.ndarray, C: float, lam: float) -> np.ndarray:
    """Modelo Exponential: f(s) = C * exp(-lambda * s)"""
    return C * np.exp(-lam * s)


def truncated_pl(s: np.ndarray, C: float, beta: float, lam: float) -> np.ndarray:
    """Modelo Truncated Power Law: f(s) = C * s^(-beta) * exp(-lambda * s)"""
    return C * np.power(s, -beta) * np.exp(-lam * s)


def fit_km_curves(binned_df: pd.DataFrame) -> dict[str, Any]:
    """Ajusta tres modelos a los datos binned.

    Args:
        binned_df: DataFrame con columnas s_center, km_mean, km_std.

    Returns:
        Dict con parámetros ajustados de cada modelo.
    """
    x = binned_df["s_center"].values
    y = binned_df["km_mean"].values
    sigma = binned_df["km_std"].values

    sigma = np.where(sigma <= 0, 1e-10, sigma)

    results = {}

    try:
        popt_pl, pcov_pl = curve_fit(
            power_law, x, y, sigma=sigma, absolute_sigma=True, maxfev=5000
        )
        results["power_law"] = {
            "C": popt_pl[0],
            "C_err": np.sqrt(pcov_pl[0, 0]),
            "beta": popt_pl[1],
            "beta_err": np.sqrt(pcov_pl[1, 1]),
        }
    except Exception:
        results["power_law"] = None

    try:
        popt_exp, pcov_exp = curve_fit(
            exponential, x, y, sigma=sigma, absolute_sigma=True, maxfev=5000
        )
        results["exponential"] = {
            "C": popt_exp[0],
            "C_err": np.sqrt(pcov_exp[0, 0]),
            "lam": popt_exp[1],
            "lam_err": np.sqrt(pcov_exp[1, 1]),
        }
    except Exception:
        results["exponential"] = None

    try:
        popt_tpl, pcov_tpl = curve_fit(
            truncated_pl, x, y, sigma=sigma, absolute_sigma=True, maxfev=5000, p0=[1, 1, 0.1]
        )
        results["truncated"] = {
            "C": popt_tpl[0],
            "C_err": np.sqrt(pcov_tpl[0, 0]),
            "beta": popt_tpl[1],
            "beta_err": np.sqrt(pcov_tpl[1, 1]),
            "lam": popt_tpl[2],
            "lam_err": np.sqrt(pcov_tpl[2, 2]),
        }
    except Exception:
        results["truncated"] = None

    return results


def plot_cluster_frequency(df: Any, ff: float) -> Figure:
    """Genera gráfico de barras de frecuencia vs tamaño de cluster.

    Args:
        df: DataFrame con datos.
        ff: Factor de fricción.

    Returns:
        Figura matplotlib.
    """
    data = compute_cluster_frequency(df, ff)
    if data.empty:
        return Figure()

    fs = 1.8 * 12
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
    ax.set_title(f"Frecuencia vs Tamaño de cluster (ff={ff}, i={I_VALUE})", fontsize=fs * 0.8)
    ax.margins(x=0.1, y=0.3)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=fs * 0.7)
    plt.tight_layout()
    return fig


def plot_cluster_ccdf(df: Any, ff: float, vel0: float) -> list[Figure]:
    """Genera gráfico CCDF log-log de tamaños de cluster.

    Args:
        df: DataFrame con datos.
        ff: Factor de fricción.
        vel0: Velocidad inicial.

    Returns:
        Lista de figuras (gráfico + tablas).
    """
    raw_sizes = compute_cluster_sizes(df, ff, vel0)

    if len(raw_sizes) < 2 or len(np.unique(raw_sizes)) < 2:
        return []

    fs = 1.8 * 12

    # Fit powerlaw
    try:
        fit = powerlaw.Fit(
            raw_sizes,
            discrete=True,
            verbose=False,
            xmin=1,
            parameter_ranges={
                "power_law": {"alpha": (1, 10)},
                "truncated_power_law": {"alpha": (1, 10), "Lambda": (1e-10, 10)},
                "exponential": {"Lambda": (1e-10, 10)},
                "lognormal": {"mu": (None, None), "sigma": (1e-10, 10)},
            },
        )
    except Exception as e:
        print(f"Error en ajuste powerlaw CCDF vel0={vel0}: {e}")
        return []

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

    if best_alpha is None:
        return []

    alpha_mle = best_alpha
    xmin_calc = best_xmin

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    unique_sizes, ccdf_empirical = compute_ccdf_empirical(raw_sizes)
    ax.bar(unique_sizes, ccdf_empirical, color=COLORS["primary"], alpha=0.7, label="Datos")

    x_int = np.arange(1, int(raw_sizes.max()) + 1)

    # Power Law
    pdf_pl = x_int ** (-alpha_mle)
    pdf_pl = pdf_pl / np.sum(pdf_pl)
    ccdf_pl = compute_ccdf_from_pdf(pdf_pl)
    ax.step(x_int, ccdf_pl, where="post", color=COLORS["power_law"],
            label=f"Power Law (α={alpha_mle:.2f})")

    # Truncated PL
    try:
        tpl = fit.truncated_power_law
        alpha_tpl, lam_tpl = tpl.alpha, tpl.Lambda
        pdf_tpl = (x_int ** (-alpha_tpl)) * np.exp(-lam_tpl * (x_int - 1))
        pdf_tpl = pdf_tpl / np.sum(pdf_tpl)
        ccdf_tpl = compute_ccdf_from_pdf(pdf_tpl)
        ax.step(x_int, ccdf_tpl, where="post", color=COLORS["truncated"],
                label=f"Truncated PL (α={alpha_tpl:.2f}, λ={lam_tpl:.2e})")
    except Exception:
        pass

    # Exponential
    try:
        exp_dist = fit.exponential
        lam_exp = exp_dist.Lambda
        pdf_exp = lam_exp * np.exp(-lam_exp * x_int)
        pdf_exp = pdf_exp / np.sum(pdf_exp)
        ccdf_exp = compute_ccdf_from_pdf(pdf_exp)
        ax.step(x_int, ccdf_exp, where="post", color=COLORS["exponential"],
                label=f"Exponential (λ={lam_exp:.2e})")
    except Exception:
        pass

    # Lognormal
    try:
        ln_dist = fit.lognormal
        mu_ln, sigma_ln = ln_dist.mu, ln_dist.sigma
        pdf_ln = (1 / (x_int * sigma_ln * np.sqrt(2 * np.pi))) * \
                 np.exp(-((np.log(x_int) - mu_ln) ** 2) / (2 * sigma_ln ** 2))
        pdf_ln = pdf_ln / np.sum(pdf_ln)
        ccdf_ln = compute_ccdf_from_pdf(pdf_ln)
        ax.step(x_int, ccdf_ln, where="post", color=COLORS["lognormal"],
                label=f"Lognormal (μ={mu_ln:.2f}, σ={sigma_ln:.2f})")
    except Exception:
        pass

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\log_{10}$(Tamaño de cluster)", fontsize=fs)
    ax.set_ylabel(r"$\log_{10}$($\bar{F}$(x))", fontsize=fs)
    ax.set_title(rf"CCDF log-log (ff={ff}, $v_0$={vel0}, i={I_VALUE})", fontsize=fs * 0.8)
    ax.legend(fontsize=fs * 0.6, loc="lower left")
    ax.margins(x=0.1, y=0.3)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    figuras = [fig]

    # Tablas
    try:
        comp_exp = fit.distribution_compare("power_law", "exponential")
        comp_ln = fit.distribution_compare("power_law", "lognormal")
        comp_trunc = fit.distribution_compare("power_law", "truncated_power_law")

        table_data = [
            ["Modelo", "R (likelihood ratio)", "p-value"],
            ["PL vs Trunc. PL", f"{comp_trunc[0]:.3f}", f"{comp_trunc[1]:.3f}"],
            ["PL vs Exponential", f"{comp_exp[0]:.3f}", f"{comp_exp[1]:.3f}"],
            ["PL vs Lognormal", f"{comp_ln[0]:.3f}", f"{comp_ln[1]:.3f}"],
        ]
        figuras.append(_create_table(table_data, "Comparación de distribuciones"))

        param_data = [
            ["Modelo", "Parámetros"],
            ["Power Law", f"α={alpha_mle:.4f}, xmin={xmin_calc:.0f}, D={fit.power_law.D:.4f}"],
        ]
        try:
            param_data.append(["Truncated PL", f"α={fit.truncated_power_law.alpha:.4f}, "
                              f"λ={fit.truncated_power_law.Lambda:.2e}, D={fit.truncated_power_law.D:.4f}"])
        except Exception:
            pass
        try:
            param_data.append(["Exponential", f"λ={fit.exponential.Lambda:.2e}, D={fit.exponential.D:.4f}"])
        except Exception:
            pass
        try:
            param_data.append(["Lognormal", f"μ={fit.lognormal.mu:.4f}, σ={fit.lognormal.sigma:.4f}, "
                              f"D={fit.lognormal.D:.4f}"])
        except Exception:
            pass
        figuras.append(_create_table(param_data, "Parámetros ajustados"))
    except Exception:
        pass

    return figuras


def _create_table(data: list[list[str]], titulo: str = "") -> Figure:
    """Crea una tabla con formato mejorado.

    Args:
        data: Datos de la tabla.
        titulo: Título opcional.

    Returns:
        Figura matplotlib.
    """
    fs = 10
    num_cols = len(data[0]) if data else 2
    fig, ax = plt.subplots(figsize=(10, max(2, len(data) * 0.4)))
    ax.axis("off")

    if num_cols == 3:
        col_widths = [0.25, 0.35, 0.35]
    elif num_cols == 2:
        col_widths = [0.3, 0.6]
    else:
        col_widths = [0.35, 0.35, 0.3]

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
            elif data[i][0].endswith(":"):
                cell.set_facecolor("#e5e7eb")
                cell.set_text_props(weight="bold")
            else:
                cell.set_facecolor("#f9fafb")

    if titulo:
        ax.set_title(titulo, fontsize=fs + 2, pad=10)

    plt.tight_layout()
    return fig


def plot_cluster_km_scatter(df: Any, ff: float, vel0: float) -> Figure:
    """Genera gráfico log-log de K/m del CM vs tamaño de cluster.

    Args:
        df: DataFrame con datos.
        ff: Factor de fricción.
        vel0: Velocidad inicial.

    Returns:
        Figura matplotlib.
    """
    data = compute_cluster_km(df, ff, vel0)
    if data.empty or len(data) < 2:
        return Figure()

    fs = 1.8 * 12

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(
        data["cluster_size"],
        data["km"],
        color=COLORS["primary"],
        alpha=0.5,
        s=30,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    km = (vel0**2) / 2
    ax.set_xlabel(r"$\log_{10}$(Tamaño de cluster)", fontsize=fs)
    ax.set_ylabel(r"$\log_{10}$($K$/$m$ cluster) [J/kg]", fontsize=fs)
    ax.set_title(
        rf"K/m cluster vs Tamaño (ff={ff}, K/m={km:.1f} J/kg, i={I_VALUE})",
        fontsize=fs * 0.8,
    )
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=fs * 0.7)
    plt.tight_layout()
    return fig


def plot_cluster_km_logfit(df: Any, ff: float, vel0: float) -> list[Figure]:
    """Genera gráfico log-log de K/m vs tamaño con ajustes ponderados.

    Args:
        df: DataFrame con datos.
        ff: Factor de fricción.
        vel0: Velocidad inicial.

    Returns:
        Lista de figuras (gráfico + tabla de parámetros).
    """
    raw_data = compute_cluster_km(df, ff, vel0)
    binned_data = compute_cluster_km_binned(df, ff, vel0, n_bins=10)

    if raw_data.empty or binned_data.empty:
        return []

    fs = 1.8 * 12

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(
        raw_data["cluster_size"],
        raw_data["km"],
        color=COLORS["primary"],
        alpha=0.3,
        s=20,
        label="Datos crudos",
    )

    ax.errorbar(
        binned_data["s_center"],
        binned_data["km_mean"],
        yerr=binned_data["km_std"],
        fmt="o",
        color=COLORS["power_law"],
        markersize=8,
        capsize=4,
        label="Log-binning (10 bins)",
    )

    x_fit = np.logspace(
        np.log10(binned_data["s_center"].min()),
        np.log10(binned_data["s_center"].max()),
        100
    )

    fit_results = fit_km_curves(binned_data)

    if fit_results.get("power_law"):
        pl = fit_results["power_law"]
        y_pl = power_law(x_fit, pl["C"], pl["beta"])
        ax.plot(x_fit, y_pl, "-", color=COLORS["power_law"], linewidth=2,
                label=f"Power Law: β={pl['beta']:.3f}")

    if fit_results.get("exponential"):
        exp = fit_results["exponential"]
        y_exp = exponential(x_fit, exp["C"], exp["lam"])
        ax.plot(x_fit, y_exp, "--", color=COLORS["exponential"], linewidth=2,
                label=f"Exponential: λ={exp['lam']:.3f}")

    if fit_results.get("truncated"):
        tpl = fit_results["truncated"]
        y_tpl = truncated_pl(x_fit, tpl["C"], tpl["beta"], tpl["lam"])
        ax.plot(x_fit, y_tpl, "-.", color=COLORS["truncated"], linewidth=2,
                label=f"Trunc. PL: β={tpl['beta']:.3f}, λ={tpl['lam']:.3f}")

    km = (vel0**2) / 2
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Tamaño de cluster", fontsize=fs)
    ax.set_ylabel("K/m cluster [J/kg]", fontsize=fs)
    ax.set_title(f"K/m vs Tamaño (ff={ff}, K/m={km:.1f} J/kg)", fontsize=fs * 0.8)
    ax.legend(fontsize=fs * 0.6, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=fs * 0.7)
    plt.tight_layout()

    figuras = [fig]

    table_data = [["Modelo", "β", "λ"]]

    if fit_results.get("power_law"):
        pl = fit_results["power_law"]
        table_data.append([
            "Power Law",
            format_uncertainty(pl['beta'], pl['beta_err']),
            "-"
        ])

    if fit_results.get("exponential"):
        exp = fit_results["exponential"]
        table_data.append([
            "Exponential",
            "-",
            format_uncertainty(exp['lam'], exp['lam_err'])
        ])

    if fit_results.get("truncated"):
        tpl = fit_results["truncated"]
        table_data.append([
            "Trunc. PL",
            format_uncertainty(tpl['beta'], tpl['beta_err']),
            format_uncertainty(tpl['lam'], tpl['lam_err'])
        ])

    figuras.append(_create_table(table_data, "Parámetros ajustados"))

    return figuras


def fit_rg_linear(binned_df: pd.DataFrame) -> dict[str, Any]:
    """Ajusta regresión lineal a log10(Rg) vs log10(s).

    Args:
        binned_df: DataFrame con columnas s_center, log_rg_mean.

    Returns:
        Dict con d_f, intercept C, y R².
    """
    if binned_df.empty or len(binned_df) < 2:
        return {}

    x = np.log10(binned_df["s_center"].values)
    y = binned_df["log_rg_mean"].values

    m, C = np.polyfit(x, y, 1)

    y_pred = m * x + C
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    df = 1 / m if m != 0 else 0

    return {
        "df": df,
        "intercept": C,
        "r_squared": r_squared,
        "slope": m,
    }


def plot_cluster_rg_logfit(df: Any, ff: float, vel0: float) -> list[Figure]:
    """Genera gráfico log-log de Rg vs tamaño con ajuste lineal.

    Args:
        df: DataFrame con datos.
        ff: Factor de fricción.
        vel0: Velocidad inicial.

    Returns:
        Lista de figuras (gráfico + tabla).
    """
    raw_data = compute_cluster_rg(df, ff, vel0)
    binned_data = compute_rg_binned(df, ff, vel0, n_bins=12)

    if raw_data.empty or binned_data.empty:
        return []

    fs = 1.8 * 12

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(
        raw_data["cluster_size"],
        raw_data["rg"],
        color=COLORS["primary"],
        alpha=0.3,
        s=20,
        label="Datos crudos",
    )

    ax.errorbar(
        binned_data["s_center"],
        10 ** binned_data["log_rg_mean"],
        yerr=10 ** binned_data["log_rg_std"],
        fmt="o",
        color=COLORS["power_law"],
        markersize=8,
        capsize=4,
        label="Log-binning (12 bins)",
    )

    fit = fit_rg_linear(binned_data)

    x_fit = np.logspace(
        np.log10(binned_data["s_center"].min()),
        np.log10(binned_data["s_center"].max()),
        100
    )
    y_fit = 10 ** (fit["slope"] * np.log10(x_fit) + fit["intercept"])
    ax.plot(x_fit, y_fit, "-", color=COLORS["truncated"], linewidth=2,
            label=f"Ajuste: d_f={fit['df']:.2f}")

    km = (vel0**2) / 2
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Tamaño de cluster (s)", fontsize=fs)
    ax.set_ylabel("Radio de giro R_g", fontsize=fs)
    ax.set_title(f"R_g vs Tamaño (ff={ff}, K/m={km:.1f} J/kg)", fontsize=fs * 0.8)
    ax.legend(fontsize=fs * 0.6, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=fs * 0.7)
    plt.tight_layout()

    figuras = [fig]

    table_data = [["Parámetro", "Valor"]]

    if fit:
        table_data.append(["d_f", f"{fit['df']:.2f}"])
        table_data.append(["C", f"{fit['intercept']:.3f}"])
        table_data.append(["R²", f"{fit['r_squared']:.3f}"])

    figuras.append(_create_table(table_data, "Dimensión Fractal"))

    return figuras