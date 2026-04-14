"""Script de visualizaciones removidas del pipeline principal.

Genera figuras en PDF: frecuencia cluster, CCDF, K/m, Rg, ángulo, parámetros.
Ejecutar: python pdf_quitadas.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from scipy.optimize import curve_fit
from uncertainties import ufloat

try:
    import powerlaw
    POWERLAW_AVAILABLE = True
except ImportError:
    POWERLAW_AVAILABLE = False


# Rutas
RESULTADOS_DIR = Path("/home/tutturuu/Documentos/Universidad/Tesis/Documentos/Archive/Resultados NP 1/Resultados 0r v1")
CSV_PATH = RESULTADOS_DIR / "merge_resultados.csv"
SALIDA_DIR = RESULTADOS_DIR / "figuras_legacy"
I_VALUE = 80

# Colores
COLORS = {
    "primary": "#3b82f6",
    "power_law": "#e74c3c",
    "truncated": "#2ecc71",
    "exponential": "#f39c12",
    "gamma": "#9b59b6",
}


def load_data() -> pd.DataFrame:
    """Carga datos desde CSV."""
    df = pd.read_csv(CSV_PATH)
    return df[(df["i"] == I_VALUE) & (df["type"].isin([1, 2]))]


def format_uncertainty(val: float, err: float) -> str:
    """Formatea valor con incertidumbre."""
    if err <= 0 or np.isnan(err) or np.isinf(err):
        return f"{val:.1f}"
    try:
        uval = ufloat(val, err)
        return f"{uval:.1u}"
    except Exception:
        return f"{val:.1f}±{err:.1f}"


# ============ CLUSTER CALC FUNCTIONS ============

def compute_cluster_sizes(df: pd.DataFrame, ff: float, vel0: float) -> np.ndarray:
    """Calcula tamaños de clusters para ff y vel0."""
    mask = (df["ff"] == ff) & (df["vel0"] == vel0) & (df["type"] == 2) & (df["z"] > 1e-4)
    return df[mask]["cluster"].value_counts().values


def compute_cluster_frequency(df: pd.DataFrame, ff: float) -> pd.DataFrame:
    """Calcula frecuencia de tamaños de cluster."""
    mask = (df["ff"] == ff) & (df["type"] == 2) & (df["z"] > 1e-4)
    sizes = df[mask]["cluster"].value_counts().values
    unique, counts = np.unique(sizes, return_counts=True)
    return pd.DataFrame({"cluster_size": unique, "frecuencia": counts})


def compute_kinetic_energy(df: pd.DataFrame, ff: float) -> pd.DataFrame:
    """Calcula energía cinética promedio K/m del CM de clusters."""
    mask = (df["ff"] == ff) & (df["type"] == 2) & (df["z"] > 1e-4)
    
    if mask.sum() == 0:
        return pd.DataFrame()

    df_filt = df[mask].copy()
    
    cm_data = df_filt.groupby(["vel0", "angulo", "posx", "posy", "cluster"]).agg(
        avg_vx=("vx", "mean"),
        avg_vy=("vy", "mean"),
        avg_vz=("vz", "mean"),
    ).reset_index()

    cm_data["ek"] = 0.5 * (cm_data["avg_vx"]**2 + cm_data["avg_vy"]**2 + cm_data["avg_vz"]**2)

    result = cm_data.groupby("vel0")["ek"].agg(["mean", "std", "min", "max"]).reset_index()
    result.columns = ["vel0", "avg", "std", "min", "max"]
    return result.sort_values("vel0")


def compute_cluster_km(df: pd.DataFrame, ff: float, vel0: float) -> pd.DataFrame:
    """Calcula K/m por cluster."""
    mask = (df["ff"] == ff) & (df["vel0"] == vel0) & (df["type"] == 2) & (df["z"] > 1e-4)
    clusters = df[mask].groupby("cluster")

    data = []
    for cid, grp in clusters:
        mass = len(grp)
        vx, vy, vz = grp["vx"].mean(), grp["vy"].mean(), grp["vz"].mean()
        km = 0.5 * (vx**2 + vy**2 + vz**2) / mass
        data.append({"cluster_id": cid, "cluster_size": mass, "km": km})

    return pd.DataFrame(data)


def compute_cluster_km_binned(df: pd.DataFrame, ff: float, vel0: float, n_bins: int = 10) -> pd.DataFrame:
    """Calcula K/m binned por tamaño."""
    raw = compute_cluster_km(df, ff, vel0)
    if raw.empty or len(raw) < 2:
        return pd.DataFrame()

    raw_log = np.log10(raw["cluster_size"])
    bins = np.logspace(raw_log.min(), raw_log.max(), n_bins + 1)

    results = []
    for i in range(n_bins):
        mask = (raw["cluster_size"] >= bins[i]) & (raw["cluster_size"] < bins[i + 1])
        bin_data = raw[mask]
        if len(bin_data) > 0:
            results.append({
                "s_center": 10 ** ((np.log10(bins[i]) + np.log10(bins[i + 1])) / 2),
                "km_mean": bin_data["km"].mean(),
                "km_std": bin_data["km"].std(),
            })

    return pd.DataFrame(results)


def compute_cluster_rg(df: pd.DataFrame, ff: float, vel0: float) -> pd.DataFrame:
    """Calcula radio de giro por cluster."""
    mask = (df["ff"] == ff) & (df["vel0"] == vel0) & (df["type"] == 2) & (df["z"] > 1e-4)
    clusters = df[mask].groupby("cluster")

    data = []
    for cid, grp in clusters:
        mass = len(grp)
        if mass < 2:
            continue

        cm = grp[["x", "y", "z"]].mean()
        rg = np.sqrt(((grp["x"] - cm["x"])**2 + (grp["y"] - cm["y"])**2 + (grp["z"] - cm["z"])**2).mean())

        data.append({"cluster_id": cid, "cluster_size": mass, "rg": rg})

    return pd.DataFrame(data)


def compute_rg_binned(df: pd.DataFrame, ff: float, vel0: float, n_bins: int = 12) -> pd.DataFrame:
    """Calcula Rg binned."""
    raw = compute_cluster_rg(df, ff, vel0)
    if raw.empty or len(raw) < 2:
        return pd.DataFrame()

    raw_log = np.log10(raw["cluster_size"])
    bins = np.logspace(raw_log.min(), raw_log.max(), n_bins + 1)

    results = []
    for i in range(n_bins):
        mask = (raw["cluster_size"] >= bins[i]) & (raw["cluster_size"] < bins[i + 1])
        bin_data = raw[mask]
        if len(bin_data) > 0:
            log_rg = np.log10(bin_data["rg"])
            results.append({
                "s_center": 10 ** ((np.log10(bins[i]) + np.log10(bins[i + 1])) / 2),
                "log_rg_mean": log_rg.mean(),
                "log_rg_std": log_rg.std(),
            })

    return pd.DataFrame(results)


def compute_velocity_angle(df: pd.DataFrame, ff: float, vel0: float) -> pd.DataFrame:
    """Calcula histograma de ángulo de velocidad."""
    mask = (df["ff"] == ff) & (df["vel0"] == vel0) & (df["type"] == 2) & (df["z"] > 1e-4)
    clusters = df[mask].groupby("cluster")

    angles = []
    for cid, grp in clusters:
        vx, vy, vz = grp["vx"].mean(), grp["vy"].mean(), grp["vz"].mean()
        v_mag = np.sqrt(vx**2 + vy**2 + vz**2)
        if v_mag > 0:
            angle = np.degrees(np.arccos(np.clip(vz / v_mag, -1, 1)))
            angles.append(angle)

    if not angles:
        return pd.DataFrame()

    bins = np.arange(0, 190, 10)
    hist, _ = np.histogram(angles, bins=bins)

    return pd.DataFrame({"bin": bins[:-1], "frecuencia": hist})


def compute_params_mle(df: pd.DataFrame, ff: float, vel0: float) -> dict:
    """Calcula parámetros MLE de distribución de tamaños."""
    sizes = compute_cluster_sizes(df, ff, vel0)
    if len(sizes) < 2:
        return {}

    if not POWERLAW_AVAILABLE:
        return {}

    try:
        fit = powerlaw.Fit(sizes, discrete=True, verbose=False, xmin=1)
        return {
            "alpha": fit.power_law.alpha,
            "xmin": fit.power_law.xmin,
            "sigma": fit.power_law.sigma,
        }
    except Exception:
        return {}


def compute_penetration_depth(df: pd.DataFrame, ff: float) -> pd.DataFrame:
    """Calcula profundidad de penetración promedio."""
    mask = (df["ff"] == ff) & (df["type"] == 2) & (df["z"] < 1e-4) & (df["vz"] < 0)
    
    if mask.sum() == 0:
        return pd.DataFrame()

    df_filt = df[mask].copy()
    df_filt["penetracion"] = abs(df_filt["z"] - 1e-4) * 1e6

    result = df_filt.groupby("vel0")["penetracion"].agg(["mean", "std", "min", "max"]).reset_index()
    result.columns = ["vel0", "avg", "std", "min", "max"]
    return result.sort_values("vel0")


# ============ FIT FUNCTIONS ============

def power_law(s: np.ndarray, C: float, beta: float) -> np.ndarray:
    return C * np.power(s, -beta)


def exponential(s: np.ndarray, C: float, lam: float) -> np.ndarray:
    return C * np.exp(-lam * s)


def truncated_pl(s: np.ndarray, C: float, beta: float, lam: float) -> np.ndarray:
    return C * np.power(s, -beta) * np.exp(-lam * s)


def fit_km_curves(binned_df: pd.DataFrame) -> dict:
    """Ajusta tres modelos a los datos binned."""
    x = binned_df["s_center"].values
    y = binned_df["km_mean"].values
    sigma = binned_df["km_std"].values
    sigma = np.where(sigma <= 0, 1e-10, sigma)

    results = {}

    try:
        popt_pl, pcov_pl = curve_fit(power_law, x, y, sigma=sigma, absolute_sigma=True, maxfev=5000)
        results["power_law"] = {
            "C": popt_pl[0], "C_err": np.sqrt(pcov_pl[0, 0]),
            "beta": popt_pl[1], "beta_err": np.sqrt(pcov_pl[1, 1]),
        }
    except Exception:
        results["power_law"] = None

    try:
        popt_exp, pcov_exp = curve_fit(exponential, x, y, sigma=sigma, absolute_sigma=True, maxfev=5000)
        results["exponential"] = {
            "C": popt_exp[0], "C_err": np.sqrt(pcov_exp[0, 0]),
            "lam": popt_exp[1], "lam_err": np.sqrt(pcov_exp[1, 1]),
        }
    except Exception:
        results["exponential"] = None

    try:
        popt_tpl, pcov_tpl = curve_fit(truncated_pl, x, y, sigma=sigma, absolute_sigma=True, maxfev=5000, p0=[1, 1, 0.1])
        results["truncated"] = {
            "C": popt_tpl[0], "C_err": np.sqrt(pcov_tpl[0, 0]),
            "beta": popt_tpl[1], "beta_err": np.sqrt(pcov_tpl[1, 1]),
            "lam": popt_tpl[2], "lam_err": np.sqrt(pcov_tpl[2, 2]),
        }
    except Exception:
        results["truncated"] = None

    return results


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

    df = 1 / m if m != 0 else 0

    return {"df": df, "intercept": C, "r_squared": r_squared, "slope": m}


# ============ PLOT FUNCTIONS ============

def plot_cluster_frequency(df: pd.DataFrame, ff: float) -> plt.Figure:
    """Gráfico de barras de frecuencia vs tamaño de cluster."""
    data = compute_cluster_frequency(df, ff)
    if data.empty:
        return plt.figure()

    fs = 12
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
    ax.set_title(f"Frecuencia vs Tamaño de cluster (ff={ff}, i={I_VALUE})", fontsize=fs * 0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_cluster_ccdf(df: pd.DataFrame, ff: float, vel0: float) -> list:
    """Gráfico CCDF log-log de tamaños de cluster."""
    raw_sizes = compute_cluster_sizes(df, ff, vel0)

    if len(raw_sizes) < 2 or len(np.unique(raw_sizes)) < 2:
        return []

    fs = 12

    if not POWERLAW_AVAILABLE:
        return []

    try:
        fit = powerlaw.Fit(
            raw_sizes, discrete=True, verbose=False, xmin=1,
            parameter_ranges={
                "power_law": {"alpha": (1, 10)},
                "truncated_power_law": {"alpha": (1, 10), "Lambda": (1e-10, 10)},
                "exponential": {"Lambda": (1e-10, 10)},
            },
        )
    except Exception as e:
        print(f"Error en ajuste powerlaw CCDF vel0={vel0}: {e}")
        return []

    alpha_mle = fit.power_law.alpha
    xmin = fit.power_law.xmin

    fig, ax = plt.subplots(figsize=(10, 6))

    unique_sizes = np.unique(raw_sizes)
    ccdf_empirical = np.array([np.mean(raw_sizes >= s) for s in unique_sizes])
    ax.bar(unique_sizes, ccdf_empirical, color=COLORS["primary"], alpha=0.7, label="Datos")

    x_int = np.arange(1, int(raw_sizes.max()) + 1)
    pdf_pl = x_int ** (-alpha_mle)
    pdf_pl = pdf_pl / np.sum(pdf_pl)
    ccdf_pl = np.cumsum(pdf_pl[::-1])[::-1]
    ax.step(x_int, ccdf_pl, where="post", color=COLORS["power_law"],
            label=f"Power Law (α={alpha_mle:.2f})")

    try:
        tpl = fit.truncated_power_law
        alpha_tpl, lam_tpl = tpl.alpha, tpl.Lambda
        pdf_tpl = (x_int ** (-alpha_tpl)) * np.exp(-lam_tpl * (x_int - 1))
        pdf_tpl = pdf_tpl / np.sum(pdf_tpl)
        ccdf_tpl = np.cumsum(pdf_tpl[::-1])[::-1]
        ax.step(x_int, ccdf_tpl, where="post", color=COLORS["truncated"],
                label=f"Truncated PL (α={alpha_tpl:.2f}, λ={lam_tpl:.2e})")
    except Exception:
        pass

    try:
        exp_dist = fit.exponential
        lam_exp = exp_dist.Lambda
        pdf_exp = lam_exp * np.exp(-lam_exp * x_int)
        pdf_exp = pdf_exp / np.sum(pdf_exp)
        ccdf_exp = np.cumsum(pdf_exp[::-1])[::-1]
        ax.step(x_int, ccdf_exp, where="post", color=COLORS["exponential"],
                label=f"Exponential (λ={lam_exp:.2e})")
    except Exception:
        pass

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Tamaño de cluster", fontsize=fs)
    ax.set_ylabel("CCDF", fontsize=fs)
    ax.set_title(f"CCDF log-log (ff={ff}, v0={vel0}, i={I_VALUE})", fontsize=fs * 0.9)
    ax.legend(fontsize=fs * 0.6, loc="lower left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    figuras = [fig]

    try:
        comp_exp = fit.distribution_compare("power_law", "exponential")
        comp_trunc = fit.distribution_compare("power_law", "truncated_power_law")

        table_data = [
            ["Modelo", "R (likelihood ratio)", "p-value"],
            ["PL vs Trunc. PL", f"{comp_trunc[0]:.3f}", f"{comp_trunc[1]:.3f}"],
            ["PL vs Exponential", f"{comp_exp[0]:.3f}", f"{comp_exp[1]:.3f}"],
        ]
        figuras.append(_create_table(table_data, "Comparación de distribuciones"))

        param_data = [
            ["Modelo", "Parámetros"],
            ["Power Law", f"α={alpha_mle:.4f}, xmin={xmin:.0f}, D={fit.power_law.D:.4f}"],
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
        figuras.append(_create_table(param_data, "Parámetros ajustados"))
    except Exception:
        pass

    return figuras


def _create_table(data: list, titulo: str = "") -> plt.Figure:
    """Crea una tabla con formato mejorado."""
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

    if titulo:
        ax.set_title(titulo, fontsize=fs + 2, pad=10)
    plt.tight_layout()
    return fig


def plot_cluster_km_scatter(df: pd.DataFrame, ff: float, vel0: float) -> plt.Figure:
    """Gráfico scatter de K/m vs tamaño."""
    data = compute_cluster_km(df, ff, vel0)
    if data.empty or len(data) < 2:
        return plt.figure()

    fs = 12
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(data["cluster_size"], data["km"], color=COLORS["primary"], alpha=0.3, s=20)

    km = (vel0**2) / 2
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Tamaño de cluster", fontsize=fs)
    ax.set_ylabel("K/m cluster [J/kg]", fontsize=fs)
    ax.set_title(f"K/m vs Tamaño (ff={ff}, K/m={km:.1f} J/kg)", fontsize=fs * 0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_cluster_km_logfit(df: pd.DataFrame, ff: float, vel0: float) -> list:
    """Gráfico log-log de K/m con ajustes."""
    raw_data = compute_cluster_km(df, ff, vel0)
    binned_data = compute_cluster_km_binned(df, ff, vel0, n_bins=10)

    if raw_data.empty or binned_data.empty:
        return []

    fs = 12
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(raw_data["cluster_size"], raw_data["km"], color=COLORS["primary"], alpha=0.3, s=20, label="Datos crudos")

    ax.errorbar(binned_data["s_center"], binned_data["km_mean"], yerr=binned_data["km_std"],
                fmt="o", color=COLORS["power_law"], markersize=8, capsize=4, label="Log-binning (10 bins)")

    x_fit = np.logspace(np.log10(binned_data["s_center"].min()), np.log10(binned_data["s_center"].max()), 100)

    fit_results = fit_km_curves(binned_data)

    if fit_results.get("power_law"):
        pl = fit_results["power_law"]
        y_pl = power_law(x_fit, pl["C"], pl["beta"])
        ax.plot(x_fit, y_pl, "-", color=COLORS["power_law"], linewidth=2, label=f"Power Law: β={pl['beta']:.3f}")

    if fit_results.get("exponential"):
        exp = fit_results["exponential"]
        y_exp = exponential(x_fit, exp["C"], exp["lam"])
        ax.plot(x_fit, y_exp, "--", color=COLORS["exponential"], linewidth=2, label=f"Exponential: λ={exp['lam']:.3f}")

    if fit_results.get("truncated"):
        tpl = fit_results["truncated"]
        y_tpl = truncated_pl(x_fit, tpl["C"], tpl["beta"], tpl["lam"])
        ax.plot(x_fit, y_tpl, "-.", color=COLORS["truncated"], linewidth=2, label=f"Trunc. PL: β={tpl['beta']:.3f}, λ={tpl['lam']:.3f}")

    km = (vel0**2) / 2
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Tamaño de cluster", fontsize=fs)
    ax.set_ylabel("K/m cluster [J/kg]", fontsize=fs)
    ax.set_title(f"K/m vs Tamaño (ff={ff}, K/m={km:.1f} J/kg)", fontsize=fs * 0.9)
    ax.legend(fontsize=fs * 0.6, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    figuras = [fig]

    table_data = [["Modelo", "β", "λ"]]

    if fit_results.get("power_law"):
        pl = fit_results["power_law"]
        table_data.append(["Power Law", format_uncertainty(pl['beta'], pl['beta_err']), "-"])

    if fit_results.get("exponential"):
        exp = fit_results["exponential"]
        table_data.append(["Exponential", "-", format_uncertainty(exp['lam'], exp['lam_err'])])

    if fit_results.get("truncated"):
        tpl = fit_results["truncated"]
        table_data.append(["Trunc. PL", format_uncertainty(tpl['beta'], tpl['beta_err']), format_uncertainty(tpl['lam'], tpl['lam_err'])])

    figuras.append(_create_table(table_data, "Parámetros ajustados"))

    return figuras


def plot_cluster_rg_logfit(df: pd.DataFrame, ff: float, vel0: float) -> list:
    """Gráfico log-log de Rg vs tamaño con ajuste lineal."""
    raw_data = compute_cluster_rg(df, ff, vel0)
    binned_data = compute_rg_binned(df, ff, vel0, n_bins=12)

    if raw_data.empty or binned_data.empty:
        return []

    fs = 12
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(raw_data["cluster_size"], raw_data["rg"], color=COLORS["primary"], alpha=0.3, s=20, label="Datos crudos")

    ax.errorbar(binned_data["s_center"], 10 ** binned_data["log_rg_mean"],
                yerr=10 ** binned_data["log_rg_std"], fmt="o", color=COLORS["power_law"],
                markersize=8, capsize=4, label="Log-binning (12 bins)")

    fit = fit_rg_linear(binned_data)

    x_fit = np.logspace(np.log10(binned_data["s_center"].min()), np.log10(binned_data["s_center"].max()), 100)
    y_fit = 10 ** (fit["slope"] * np.log10(x_fit) + fit["intercept"])
    ax.plot(x_fit, y_fit, "-", color=COLORS["truncated"], linewidth=2, label=f"Ajuste: d_f={fit['df']:.2f}")

    km = (vel0**2) / 2
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Tamaño de cluster (s)", fontsize=fs)
    ax.set_ylabel("Radio de giro R_g", fontsize=fs)
    ax.set_title(f"R_g vs Tamaño (ff={ff}, K/m={km:.1f} J/kg)", fontsize=fs * 0.9)
    ax.legend(fontsize=fs * 0.6, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    figuras = [fig]

    table_data = [["Parámetro", "Valor"]]
    if fit:
        table_data.append(["d_f", f"{fit['df']:.2f}"])
        table_data.append(["C", f"{fit['intercept']:.3f}"])
        table_data.append(["R²", f"{fit['r_squared']:.3f}"])

    figuras.append(_create_table(table_data, "Dimensión Fractal"))

    return figuras


def plot_velocity_angle(df: pd.DataFrame, ff: float, vel0: float) -> plt.Figure:
    """Gráfico de distribución del ángulo de velocidad del CM."""
    data = compute_velocity_angle(df, ff, vel0)
    if data.empty:
        return plt.figure()

    fs = 12
    labels = [f"{int(b)}° a {int(b) + 10}°" for b in data["bin"]]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(labels, data["frecuencia"], color=COLORS["primary"])

    ax.set_xlabel("Ángulo (°)", fontsize=fs)
    ax.set_ylabel("Frecuencia", fontsize=fs)
    ax.set_title(f"Ángulo velocidad CM (ff={ff}, vel0={vel0}, i={I_VALUE})", fontsize=fs * 0.9)
    plt.xticks(rotation=45, ha="right", fontsize=fs * 0.7)
    ax.margins(x=0.1, y=0.3)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=fs * 0.7)
    plt.tight_layout()
    return fig


def plot_params_vs_energy(df: pd.DataFrame, ff: float) -> plt.Figure:
    """Gráfico de parámetros MLE vs energía cinética."""
    vel0_values = sorted(df[df["ff"] == ff]["vel0"].unique())

    alpha_vals, xmin_vals, sigma_vals = [], [], []

    for v in vel0_values:
        params = compute_params_mle(df, ff, v)
        alpha_vals.append(params.get("alpha", np.nan))
        xmin_vals.append(params.get("xmin", np.nan))
        sigma_vals.append(params.get("sigma", np.nan))

    km_values = [(v**2) / 2 for v in vel0_values]

    fs = 10
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(km_values, alpha_vals, "o-", color=COLORS["power_law"])
    axes[0, 0].set_xlabel(r"$K$/$m$ [J/kg]", fontsize=fs)
    axes[0, 0].set_ylabel(r"$\alpha$", fontsize=fs)
    axes[0, 0].set_title(r"$\alpha$ vs $K$/$m$", fontsize=fs)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(km_values, xmin_vals, "o-", color=COLORS["truncated"])
    axes[0, 1].set_xlabel(r"$K$/$m$ [J/kg]", fontsize=fs)
    axes[0, 1].set_ylabel(r"$x_{min}$", fontsize=fs)
    axes[0, 1].set_title(r"$x_{min}$ vs $K$/$m$", fontsize=fs)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(km_values, sigma_vals, "o-", color=COLORS["exponential"])
    axes[1, 0].set_xlabel(r"$K$/$m$ [J/kg]", fontsize=fs)
    axes[1, 0].set_ylabel(r"$\sigma$", fontsize=fs)
    axes[1, 0].set_title(r"$\sigma$ vs $K$/$m$", fontsize=fs)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].axis("off")
    axes[1, 1].text(0.5, 0.5, "Parámetros MLE\n(ff={ff}, i={I_VALUE})",
                    ha="center", va="center", fontsize=fs * 1.5)

    fig.suptitle(f"Parámetros de ajuste Power Law (ff={ff})", fontsize=fs * 1.2)
    plt.tight_layout()
    return fig


def plot_penetration(df: pd.DataFrame, ff: float) -> plt.Figure:
    """Gráfico de profundidad de penetración vs vel0."""
    data = compute_penetration_depth(df, ff)
    if data.empty:
        return plt.figure()

    fs = 12
    unique_vel0 = sorted(data["vel0"].unique())
    avg_values = data.set_index("vel0").loc[unique_vel0, "avg"].values

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(unique_vel0)), avg_values, color=COLORS["primary"])

    km_values = [(v**2) / 2 for v in unique_vel0]

    ax.set_xticks(range(len(unique_vel0)))
    ax.set_xticklabels([f"{km:.1f}" for km in km_values])
    ax.set_xlabel(r"$K$/$m$ [J/kg]", fontsize=fs)
    ax.set_ylabel("Profundidad de penetración (µm)", fontsize=fs)
    ax.set_title(f"Penetración (ff={ff}, i={I_VALUE})", fontsize=fs * 0.9)
    ax.margins(x=0.1, y=0.3)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=fs * 0.7)
    plt.tight_layout()
    return fig


def plot_kinetic_energy(df: pd.DataFrame, ff: float) -> plt.Figure:
    """Gráfico de K/m promedio del CM de clusters salientes."""
    data = compute_kinetic_energy(df, ff)
    if data.empty:
        return plt.figure()

    fs = 12
    unique_vel0 = sorted(data["vel0"].unique())
    avg_values = data.set_index("vel0").loc[unique_vel0, "avg"].values

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(unique_vel0)), avg_values, color=COLORS["primary"])

    km_values = [(v**2) / 2 for v in unique_vel0]

    ax.set_xticks(range(len(unique_vel0)))
    ax.set_xticklabels([f"{km:.1f}" for km in km_values])
    ax.set_xlabel(r"$K$/$m$ [J/kg]", fontsize=fs)
    ax.set_ylabel(r"$K$/$m$ cluster [J/kg]", fontsize=fs)
    ax.set_title(f"K/m cluster (ff={ff}, i={I_VALUE})", fontsize=fs * 0.9)
    ax.margins(x=0.1, y=0.3)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=fs * 0.7)
    plt.tight_layout()
    return fig


# ============ MAIN ============

def run_all_legacy_plots() -> None:
    """Ejecuta todas las visualizaciones y genera PDF."""
    print("Cargando datos...")
    df = load_data()
    print(f"Datos cargados: {len(df)} filas")

    SALIDA_DIR.mkdir(parents=True, exist_ok=True)

    figuras = []
    FF_VALUES = [0.15, 0.35]
    valores_vel0 = sorted(df["vel0"].unique())

    for ff in FF_VALUES:
        print(f"Procesando ff={ff}...")

        # Frecuencia vs tamaño
        fig_freq = plot_cluster_frequency(df, ff)
        if fig_freq.get_axes():
            figuras.append(fig_freq)

        # Por cada vel0
        for vel0 in valores_vel0:
            print(f"  vel0={vel0}...")

            # CCDF
            figs_ccdf = plot_cluster_ccdf(df, ff, vel0)
            for fig_cc in figs_ccdf:
                if fig_cc.get_axes():
                    figuras.append(fig_cc)

            # Ángulo de velocidad
            fig_ang = plot_velocity_angle(df, ff, vel0)
            if fig_ang.get_axes():
                figuras.append(fig_ang)

            # K/m scatter
            fig_km = plot_cluster_km_scatter(df, ff, vel0)
            if fig_km.get_axes():
                figuras.append(fig_km)

            # K/m fit
            figs_km_fit = plot_cluster_km_logfit(df, ff, vel0)
            for fig_kmf in figs_km_fit:
                if fig_kmf.get_axes():
                    figuras.append(fig_kmf)

            # Rg fit
            figs_rg = plot_cluster_rg_logfit(df, ff, vel0)
            for fig_rg in figs_rg:
                if fig_rg.get_axes():
                    figuras.append(fig_rg)

        # Parámetros vs energía
        fig_params = plot_params_vs_energy(df, ff)
        if fig_params.get_axes():
            figuras.append(fig_params)

        # Penetración
        fig_pen = plot_penetration(df, ff)
        if fig_pen.get_axes():
            figuras.append(fig_pen)

        # Energía cinética
        fig_ek = plot_kinetic_energy(df, ff)
        if fig_ek.get_axes():
            figuras.append(fig_ek)

    # PDF final
    pdf_path = SALIDA_DIR / "figuras_quitadas.pdf"
    print(f"Generando PDF: {pdf_path}")
    with PdfPages(pdf_path) as pdf:
        for fig in figuras:
            if fig.get_axes():
                pdf.savefig(fig)

    print(f"¡Listo! PDF en {pdf_path}")


if __name__ == "__main__":
    run_all_legacy_plots()