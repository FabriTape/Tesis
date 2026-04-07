"""Módulo de visualizaciones de distribución de clusters.

Genera gráficos de frecuencia y CCDF para análisis de distribución de tamaños.
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import powerlaw
from matplotlib.figure import Figure

from cluster_calc import compute_cluster_frequency, compute_cluster_sizes
from constants import I_VALUE, COLORS
from utils import compute_alpha_mle, compute_ccdf_empirical, compute_ccdf_from_pdf


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
    ax.set_xlabel("Tamaño de cluster", fontsize=fs)
    ax.set_ylabel("Frecuencia", fontsize=fs)
    ax.set_title(f"Frecuencia vs Tamaño de cluster (ff={ff}, i={I_VALUE})", fontsize=fs * 0.8)
    ax.margins(x=0.1, y=0.3)
    ax.grid(True, alpha=0.3, axis="y")
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

    col_widths = [0.35, 0.35, 0.3] if num_cols == 3 else [0.4, 0.2]

    table = ax.table(cellText=data, loc="center", cellLoc="center", colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(fs)
    table.scale(1.2, 1.5)

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