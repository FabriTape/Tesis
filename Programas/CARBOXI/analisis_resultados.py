"""Script de análisis de merge_resultados.csv para CARBOXI.

Genera gráficos de análisis de erosión: IDs únicos por velocidad, heatmaps,
distribución de clusters, ángulo de velocidad, energía cinética y profundidad
de penetración.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import powerlaw
import seaborn as sns
from matplotlib.figure import Figure

CARBOXI_DIR = Path("/home/tutturuu/Documentos/Universidad/Tesis/Programas/CARBOXI")
RESULTADOS_DIR = Path("/home/tutturuu/Documentos/resultados0r")
CSV_PATH = RESULTADOS_DIR / "merge_resultados.csv"
SALIDA_DIR = RESULTADOS_DIR / "figuras"

I_VALUE = 80
EXCLUIR_EROSION = "NOT (type = 2 AND (z < 1e-4 OR vz < 0))"
FILTRO_PENETRACION = "type = 2 AND (z < 1e-4 OR vz < 0)"


def crear_tabla(data: list[list[str]], titulo: str = "") -> Figure:
    """Crea una tabla con formato mejorado.

    Args:
        data: Datos de la tabla (lista de filas).
        titulo: Título opcional.

    Returns:
        Figura matplotlib.
    """
    fs = 10
    num_cols = len(data[0]) if data else 2
    fig, ax = plt.subplots(figsize=(10, max(2, len(data) * 0.4)))
    ax.axis("off")

    col_widths = [0.35, 0.35, 0.3] if num_cols == 3 else [0.4, 0.2]

    table = ax.table(
        cellText=data,
        loc="center",
        cellLoc="center",
        colWidths=col_widths,
    )
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


def cargar_datos() -> pd.DataFrame:
    """Carga el CSV de resultados de simulaciones LAMMPS.

    Returns:
        DataFrame con los datos de merge_resultados.csv.
    """
    return pd.read_csv(CSV_PATH)


def aplicar_filtros(
    df: pd.DataFrame,
    ff: float,
    vel0: float | None = None,
    i_value: int = I_VALUE,
    excluir_erosion: bool = True,
) -> pd.DataFrame:
    """Aplica filtros estándar a los datos.

    Args:
        df: DataFrame de datos.
        ff: Factor de fricción a filtrar.
        vel0: Velocidad inicial opcional.
        i_value: Paso temporal a filtrar.
        excluir_erosion: Si True, aplica filtro de exclusión de erosión.

    Returns:
        DataFrame filtrado.
    """
    mask = (df["i"] == i_value) & (df["ff"] == ff)
    if vel0 is not None:
        mask &= df["vel0"] == vel0
    if excluir_erosion:
        mask &= ~((df["type"] == 2) & ((df["z"] < 1e-4) | (df["vz"] < 0)))
    return df[mask].copy()


def calcular_ids_por_vel0(df: pd.DataFrame, ff: float) -> pd.DataFrame:
    """Calcula promedio de IDs únicos por vel0.

    Args:
        df: DataFrame con datos.
        ff: Factor de fricción.

    Returns:
        DataFrame con estadísticas por vel0.
    """
    df_filt = aplicar_filtros(df, ff)
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


def calcular_ids_raw_por_vel0(df: pd.DataFrame, ff: float) -> dict[float, list[int]]:
    """Calcula IDs únicos por cada simulación individual (movimiento).

    Args:
        df: DataFrame con datos.
        ff: Factor de fricción.

    Returns:
        Dict con vel0 -> lista de IDs únicos por movimiento.
    """
    df_filt = aplicar_filtros(df, ff)
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


def calcular_frec_cluster(df: pd.DataFrame, ff: float) -> pd.DataFrame:
    """Calcula frecuencia de tamaños de cluster por ff.

    Args:
        df: DataFrame con datos.
        ff: Factor de fricción.

    Returns:
        DataFrame con frecuencia promedio por tamaño de cluster.
    """
    df_filt = aplicar_filtros(df, ff)

    sizes_per_sim = []
    for (vel0, angulo, posx, posy), group in df_filt.groupby(
        ["vel0", "angulo", "posx", "posy"]
    ):
        cluster_sizes = group.groupby("cluster")["id"].nunique()
        for size in cluster_sizes:
            sizes_per_sim.append({"vel0": vel0, "cluster_size": size})

    if not sizes_per_sim:
        return pd.DataFrame()

    df_sizes = pd.DataFrame(sizes_per_sim)

    result = (
        df_sizes.groupby("cluster_size")["vel0"].count().reset_index(name="frecuencia")
    )

    freq_by_vel0 = (
        df_sizes.groupby(["cluster_size", "vel0"]).size().unstack(fill_value=0)
    )
    freq_stats = freq_by_vel0.mean(axis=1).reset_index(name="avg")
    freq_stats["std"] = freq_by_vel0.std(axis=1)

    result = result.merge(freq_stats, on="cluster_size")
    return result.sort_values("cluster_size")


def grafico_ids_por_vel0(df: pd.DataFrame, ff: float) -> Figure:
    """Genera gráfico de barras de frecuencia vs tamaño de cluster.

    Args:
        df: DataFrame con datos.
        ff: Factor de fricción.

    Returns:
        Figura matplotlib.
    """
    data = calcular_frec_cluster(df, ff)
    if data.empty:
        return Figure()

    fs = 1.8 * 12
    sizes = data["cluster_size"].values
    freq = data["frecuencia"].values

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(range(len(sizes)), freq, color="#3b82f6", capsize=3, alpha=0.7)

    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([str(int(s)) for s in sizes])
    ax.set_xlabel("Tamaño de cluster", fontsize=fs)
    ax.set_ylabel("Frecuencia", fontsize=fs)
    ax.set_title(
        f"Frecuencia vs Tamaño de cluster (ff={ff}, i={I_VALUE})", fontsize=fs * 0.8
    )
    ax.margins(x=0.1, y=0.3)
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(labelsize=fs * 0.7)
    plt.tight_layout()
    return fig


def grafico_ids_boxplot(df: pd.DataFrame, ff: float) -> Figure:
    """Genera gráfico de cajas y bigotes de IDs únicos por K/m.

    Muestra la distribución de partículas erosionadas a través de todos
    los movimientos (combinaciones de angulo, posx, posy) para cada K/m.

    Args:
        df: DataFrame con datos.
        ff: Factor de fricción.

    Returns:
        Figura matplotlib.
    """
    data = calcular_ids_raw_por_vel0(df, ff)
    if not data:
        return Figure()

    fs = 1.8 * 12
    unique_vel0 = sorted(data.keys())

    km_values = [(v**2) / 2 for v in unique_vel0]
    box_data = [data[v] for v in unique_vel0]

    fig, ax = plt.subplots(figsize=(10, 6))

    bp = ax.boxplot(
        box_data,
        labels=[f"{km:.1f}" for km in km_values],
        patch_artist=True,
        medianprops={"color": "#e74c3c", "linewidth": 2},
        flierprops={
            "marker": "o",
            "markerfacecolor": "#3b82f6",
            "markersize": 6,
            "alpha": 0.5,
        },
    )

    for patch in bp["boxes"]:
        patch.set_facecolor("#3b82f6")
        patch.set_alpha(0.3)

    ax.set_xlabel(r"$K$/$m$ [J/kg]", fontsize=fs)
    ax.set_ylabel("Y promedio", fontsize=fs)
    ax.set_title(
        f"Distribución de erosión por K/m (ff={ff}, i={I_VALUE})", fontsize=fs * 0.8
    )
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(labelsize=fs * 0.7)
    plt.tight_layout()
    return fig


def grafico_cluster_ccdf(df: pd.DataFrame, ff: float, vel0: float) -> list[Figure]:
    """Genera gráfico CCDF log-log de tamaños de cluster.

    Args:
        df: DataFrame con datos.
        ff: Factor de fricción.
        vel0: Velocidad inicial.

    Returns:
        Lista de figuras (gráfico + tablas).
    """
    df_filt = aplicar_filtros(df, ff, vel0)
    if df_filt.empty:
        return []

    raw_sizes = (
        df_filt.groupby(["angulo", "posx", "posy", "cluster"])["id"].nunique().values
    )

    if len(np.unique(raw_sizes)) < 2:
        return []

    fs = 1.8 * 12

    try:
        fit = powerlaw.Fit(
            raw_sizes,
            discrete=True,
            verbose=False,
            parameter_ranges={
                "power_law": {"alpha": (1, 10)},
                "truncated_power_law": {"alpha": (1, 10), "Lambda": (1e-10, 10)},
                "exponential": {"Lambda": (1e-10, 10)},
                "lognormal": {"mu": (None, None), "sigma": (1e-10, 10)},
            },
        )
        if np.isnan(fit.power_law.alpha):
            raise ValueError("Alpha is NaN")
    except Exception as e:
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
            if np.isnan(fit.power_law.alpha):
                raise ValueError("Alpha is NaN")
        except Exception as e2:
            print(f"Error en ajuste powerlaw CCDF vel0={vel0}: {e} -> {e2}")
            return []

    def hurwitz_zeta(s, q, n=100):
        result = 0.0
        for k in range(n):
            result += 1.0 / ((k + q) ** s)
        return result

    def hurwitz_zeta_derivative(s, q, n=100):
        result = 0.0
        for k in range(n):
            result -= np.log(k + q) / ((k + q) ** s)
        return result

    def compute_alpha_mle(data, xmin):
        n = len(data)
        log_sum = np.sum(np.log(data))
        target = -log_sum / n

        def objective(alpha):
            if alpha <= 1:
                return 1e10
            zeta = hurwitz_zeta(alpha, xmin)
            zeta_prime = hurwitz_zeta_derivative(alpha, xmin)
            if zeta <= 0 or np.isnan(zeta) or np.isinf(zeta):
                return 1e10
            return (zeta_prime / zeta) - target

        from scipy.optimize import brentq

        try:
            alpha_opt = brentq(objective, 1.01, 20)
            return alpha_opt
        except:
            return 1 + n / np.sum(np.log(data[data >= xmin] / xmin))

    xmin_options = [1, int(raw_sizes.min())]
    best_alpha = None
    best_xmin = None
    best_D = float("inf")

    for xm in xmin_options:
        alpha = compute_alpha_mle(raw_sizes, xm)
        zeta = hurwitz_zeta(alpha, xm)
        cdf_theoretical = np.array(
            [
                np.sum((raw_sizes >= x) * (x ** (-alpha)))
                / (zeta * len(raw_sizes[raw_sizes >= xm]))
                for x in np.sort(raw_sizes)
            ]
        )
        cdf_empirical = np.arange(1, len(raw_sizes) + 1) / len(raw_sizes)
        D = np.max(np.abs(cdf_empirical - cdf_theoretical[: len(cdf_empirical)]))
        if D < best_D:
            best_D = D
            best_alpha = alpha
            best_xmin = xm

    alpha_mle = best_alpha
    xmin_calc = best_xmin

    fig, ax = plt.subplots(figsize=(10, 6))

    unique_sizes, counts = np.unique(raw_sizes, return_counts=True)
    ccdf_empirical = np.cumsum(counts[::-1])[::-1] / len(raw_sizes)
    ccdf_empirical = np.clip(ccdf_empirical, 1e-15, 1.0)

    ax.bar(unique_sizes, ccdf_empirical, color="#3b82f6", alpha=0.7, label="Datos")

    x_int = np.arange(1, int(raw_sizes.max()) + 1)

    pdf_pl = x_int ** (-alpha_mle)
    pdf_pl = pdf_pl / np.sum(pdf_pl)
    ccdf_pl = np.cumsum(pdf_pl[::-1])[::-1]
    ccdf_pl = np.clip(ccdf_pl, 1e-15, 1.0)
    ax.step(
        x_int,
        ccdf_pl,
        where="post",
        color="#e74c3c",
        label=f"Power Law (α={alpha_mle:.2f})",
    )

    try:
        tpl = fit.truncated_power_law
        alpha_tpl = tpl.alpha
        lam_tpl = tpl.Lambda
        pdf_tpl = (x_int ** (-alpha_tpl)) * np.exp(-lam_tpl * (x_int - 1))
        pdf_tpl = pdf_tpl / np.sum(pdf_tpl)
        ccdf_tpl = np.cumsum(pdf_tpl[::-1])[::-1]
        ccdf_tpl = np.clip(ccdf_tpl, 1e-15, 1.0)
        ax.step(
            x_int,
            ccdf_tpl,
            where="post",
            color="#2ecc71",
            label=f"Truncated PL (α={alpha_tpl:.2f}, λ={lam_tpl:.2e})",
        )
    except Exception as e:
        print(f"Truncated PL error vel0={vel0}: {e}")

    try:
        exp_dist = fit.exponential
        lam_exp = exp_dist.Lambda
        pdf_exp = lam_exp * np.exp(-lam_exp * x_int)
        pdf_exp = pdf_exp / np.sum(pdf_exp)
        ccdf_exp = np.cumsum(pdf_exp[::-1])[::-1]
        ccdf_exp = np.clip(ccdf_exp, 1e-15, 1.0)
        ax.step(
            x_int,
            ccdf_exp,
            where="post",
            color="#f39c12",
            label=f"Exponential (λ={lam_exp:.2e})",
        )
    except Exception as e:
        print(f"Exponential error vel0={vel0}: {e}")

    try:
        ln_dist = fit.lognormal
        mu_ln = ln_dist.mu
        sigma_ln = ln_dist.sigma
        pdf_ln = (1 / (x_int * sigma_ln * np.sqrt(2 * np.pi))) * np.exp(-((np.log(x_int) - mu_ln) ** 2) / (2 * sigma_ln ** 2))
        pdf_ln = pdf_ln / np.sum(pdf_ln)
        ccdf_ln = np.cumsum(pdf_ln[::-1])[::-1]
        ccdf_ln = np.clip(ccdf_ln, 1e-15, 1.0)
        ax.step(
            x_int,
            ccdf_ln,
            where="post",
            color="#9b59b6",
            label=f"Lognormal (μ={mu_ln:.2f}, σ={sigma_ln:.2f})",
        )
    except Exception as e:
        print(f"Lognormal error vel0={vel0}: {e}")

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(r"$\log_{10}$(Tamaño de cluster)", fontsize=fs)
    ax.set_ylabel(r"$\log_{10}$($\bar{F}$(x))", fontsize=fs)
    ax.set_title(
        rf"CCDF log-log (ff={ff}, $v_0$={vel0}, i={I_VALUE})", fontsize=fs * 0.8
    )
    ax.legend(fontsize=fs * 0.6, loc="lower left")
    ax.margins(x=0.1, y=0.3)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    figuras = [fig]

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
        figuras.append(crear_tabla(table_data, "Comparación de distribuciones"))

        param_data = [
            ["Modelo", "Parámetros"],
            [
                "Power Law",
                f"α={alpha_mle:.4f}, xmin={xmin_calc:.0f}, D={fit.power_law.D:.4f}",
            ],
        ]

        try:
            tpl = fit.truncated_power_law
            param_data.append(
                [
                    "Truncated PL",
                    f"α={tpl.alpha:.4f}, λ={tpl.Lambda:.2e}, D={tpl.D:.4f}",
                ]
            )
        except Exception:
            pass

        try:
            exp_dist = fit.exponential
            param_data.append(
                ["Exponential", f"λ={exp_dist.Lambda:.2e}, D={exp_dist.D:.4f}"]
            )
        except Exception:
            pass

        try:
            ln_dist = fit.lognormal
            param_data.append(
                [
                    "Lognormal",
                    f"μ={ln_dist.mu:.4f}, σ={ln_dist.sigma:.4f}, D={ln_dist.D:.4f}",
                ]
            )
        except Exception:
            pass

        figuras.append(crear_tabla(param_data, "Parámetros ajustados"))
    except Exception:
        pass

    return figuras


def calcular_angulo_vel_cm(df: pd.DataFrame, ff: float, vel0: float) -> pd.DataFrame:
    """Calcula distribución del ángulo de velocidad del CM por cluster.

    Args:
        df: DataFrame con datos.
        ff: Factor de fricción.
        vel0: Velocidad inicial.

    Returns:
        DataFrame con distribución por bins de ángulo.
    """
    df_filt = aplicar_filtros(df, ff, vel0)
    cm_data = (
        df_filt.groupby(["angulo", "posx", "posy", "cluster"])
        .agg(
            avg_vx=("vx", "mean"),
            avg_vy=("vy", "mean"),
            avg_vz=("vz", "mean"),
        )
        .reset_index()
    )

    import numpy as np

    cm_data["angle"] = np.degrees(
        np.arctan2(
            cm_data["avg_vz"], np.sqrt(cm_data["avg_vx"] ** 2 + cm_data["avg_vy"] ** 2)
        )
    )
    cm_data["bin"] = (cm_data["angle"] // 10) * 10
    freq = cm_data.groupby("bin").size().reset_index(name="frecuencia")
    return freq.sort_values("bin")


def grafico_angulo_vel_cm(df: pd.DataFrame, ff: float, vel0: float) -> Figure:
    """Genera gráfico de distribución del ángulo de velocidad del CM.

    Args:
        df: DataFrame con datos.
        ff: Factor de fricción.
        vel0: Velocidad inicial.

    Returns:
        Figura matplotlib.
    """
    data = calcular_angulo_vel_cm(df, ff, vel0)
    if data.empty:
        return Figure()

    fs = 1.8 * 12
    labels = [f"{int(b)}° a {int(b) + 10}°" for b in data["bin"]]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(labels, data["frecuencia"], color="#3b82f6")

    ax.set_xlabel("Ángulo (°)", fontsize=fs)
    ax.set_ylabel("Frecuencia", fontsize=fs)
    ax.set_title(
        f"Ángulo velocidad CM (ff={ff}, vel0={vel0}, i={I_VALUE})", fontsize=fs * 0.8
    )
    plt.xticks(rotation=45, ha="right", fontsize=fs * 0.7)
    ax.margins(x=0.1, y=0.3)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=fs * 0.7)
    plt.tight_layout()
    return fig


def calcular_heatmap_erosion(df: pd.DataFrame, ff: float, vel0: float) -> pd.DataFrame:
    """Calcula количecUnique IDs erosionadas por posición (posx, posy).

    Args:
        df: DataFrame con datos.
        ff: Factor de fricción.
        vel0: Velocidad inicial.

    Returns:
        DataFrame pivot con promedio de IDs por posición.
    """
    df_filt = aplicar_filtros(df, ff, vel0)
    grouped = df_filt.groupby(["posx", "posy"])["id"].nunique().reset_index()
    return grouped.pivot(index="posy", columns="posx", values="id").fillna(0)


def grafico_heatmap_erosion(df: pd.DataFrame, ff: float, vel0: float) -> Figure:
    """Genera heatmap square de erosión por posición.

    Args:
        df: DataFrame con datos.
        ff: Factor de fricción.
        vel0: Velocidad inicial.

    Returns:
        Figura matplotlib.
    """
    data = calcular_heatmap_erosion(df, ff, vel0)
    if data.empty:
        return Figure()

    fs = 1.8 * 12

    fig, ax = plt.subplots(figsize=(8, 6))

    data_sorted = data.sort_index(ascending=False)

    sns.heatmap(
        data_sorted,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        cbar_kws={"label": "Y promedio", "shrink": 0.8},
        ax=ax,
        linewidths=0.5,
        linecolor="white",
        square=True,
    )

    km = (vel0**2) / 2
    ax.set_xlabel("posx (σ)", fontsize=fs)
    ax.set_ylabel("posy (σ)", fontsize=fs)
    ax.set_title(
        f"Erosión por posición (ff={ff}, K/m={km:.1f} J/kg, i={I_VALUE})",
        fontsize=fs * 0.8,
    )
    ax.tick_params(labelsize=fs * 0.7)
    plt.tight_layout()
    return fig


def calcular_energia_km(df: pd.DataFrame, ff: float) -> pd.DataFrame:
    """Calcula energía cinética promedio K/m del CM de clusters.

    Args:
        df: DataFrame con datos.
        ff: Factor de fricción.

    Returns:
        DataFrame con estadísticas de energía por vel0.
    """
    df_filt = aplicar_filtros(df, ff)
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
        .agg(
            avg="mean",
            std="std",
            min="min",
            max="max",
        )
        .reset_index()
    )
    result["lower_std"] = result["avg"] - result["std"].fillna(0)
    result["lower_std"] = result["lower_std"].clip(lower=0)
    result["upper_std"] = result["avg"] + result["std"].fillna(0)
    return result.sort_values("vel0")


def grafico_energia_km(df: pd.DataFrame, ff: float) -> Figure:
    """Genera gráfico de K/m promedio del CM de clusters salientes.

    Args:
        df: DataFrame con datos.
        ff: Factor de fricción.

    Returns:
        Figura matplotlib.
    """
    data = calcular_energia_km(df, ff)
    if data.empty:
        return Figure()

    fs = 1.8 * 12
    unique_vel0 = sorted(data["vel0"].unique())
    avg_values = data.set_index("vel0").loc[unique_vel0, "avg"].values

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(unique_vel0)), avg_values, color="#3b82f6")

    km_values = [(v**2) / 2 for v in unique_vel0]

    ax.set_xticks(range(len(unique_vel0)))
    ax.set_xticklabels([f"{km:.1f}" for km in km_values])
    ax.set_xlabel(r"$K$/$m$ [J/kg]", fontsize=fs)
    ax.set_ylabel(r"$K$/$m$ cluster [J/kg]", fontsize=fs)
    ax.set_title(f"K/m cluster (ff={ff}, i={I_VALUE})", fontsize=fs * 0.8)
    ax.margins(x=0.1, y=0.3)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=fs * 0.7)
    plt.tight_layout()
    return fig


def calcular_penetracion(df: pd.DataFrame, ff: float) -> pd.DataFrame:
    """Calcula profundidad de penetración promedio.

    Args:
        df: DataFrame con datos.
        ff: Factor de fricción.

    Returns:
        DataFrame con estadísticas de penetración por vel0.
    """
    df_filt = df[(df["i"] == I_VALUE) & (df["ff"] == ff)]
    df_filt = df_filt[
        (df_filt["type"] == 2) & ((df_filt["z"] < 1e-4) | (df_filt["vz"] < 0))
    ]
    df_filt = df_filt.copy()
    df_filt["penetracion"] = abs(df_filt["z"] - 1e-4) * 1e6

    result = (
        df_filt.groupby("vel0")["penetracion"]
        .agg(
            avg="mean",
            std="std",
            min="min",
            max="max",
        )
        .reset_index()
    )
    result["lower_std"] = result["avg"] - result["std"].fillna(0)
    result["lower_std"] = result["lower_std"].clip(lower=0)
    result["upper_std"] = result["avg"] + result["std"].fillna(0)
    return result.sort_values("vel0")


def grafico_penetracion(df: pd.DataFrame, ff: float) -> Figure:
    """Genera gráfico de profundidad de penetración vs vel0.

    Args:
        df: DataFrame con datos.
        ff: Factor de fricción.

    Returns:
        Figura matplotlib.
    """
    data = calcular_penetracion(df, ff)

    if data.empty:
        return plt.figure()

    fs = 1.8 * 12
    fig, ax = plt.subplots(figsize=(10, 6))

    unique_vel0 = sorted(data["vel0"].unique())
    avg_values = [data[data["vel0"] == v]["avg"].values[0] for v in unique_vel0]

    ax.bar(range(len(unique_vel0)), avg_values, color="#3b82f6")

    km_values = [(v**2) / 2 for v in unique_vel0]

    ax.set_xticks(range(len(unique_vel0)))
    ax.set_xticklabels([f"{km:.1f}" for km in km_values])
    ax.set_xlabel(r"$K$/$m$ [J/kg]", fontsize=fs)
    ax.set_ylabel("Profundidad de penetración (µm)", fontsize=fs)
    ax.set_title(f"Penetración (ff={ff}, i={I_VALUE})", fontsize=fs * 0.8)
    ax.margins(x=0.1, y=0.3)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=fs * 0.7)
    plt.tight_layout()
    return fig


def grafico_parametros_vs_energia(df: pd.DataFrame, ff: float) -> Figure:
    """Genera gráfico de parámetros de distribuciones vs K/m.

    Args:
        df: DataFrame con datos.
        ff: Factor de fricción.

    Returns:
        Figura matplotlib.
    """
    fs = 1.8 * 12
    valores_vel0 = sorted(df["vel0"].unique())

    def hurwitz_zeta(s, q, n=100):
        result = 0.0
        for k in range(n):
            result += 1.0 / ((k + q) ** s)
        return result

    def hurwitz_zeta_derivative(s, q, n=100):
        result = 0.0
        for k in range(n):
            result -= np.log(k + q) / ((k + q) ** s)
        return result

    def compute_alpha_mle(data, xmin):
        n = len(data)
        log_sum = np.sum(np.log(data))
        target = -log_sum / n

        def objective(alpha):
            if alpha <= 1:
                return 1e10
            zeta = hurwitz_zeta(alpha, xmin)
            zeta_prime = hurwitz_zeta_derivative(alpha, xmin)
            if zeta <= 0 or np.isnan(zeta) or np.isinf(zeta):
                return 1e10
            return (zeta_prime / zeta) - target

        from scipy.optimize import brentq

        try:
            return brentq(objective, 1.01, 20)
        except:
            return 1 + n / np.sum(np.log(data[data >= xmin] / xmin))

    results = []

    for vel0 in valores_vel0:
        df_filt = aplicar_filtros(df, ff, vel0)
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

        ek = 0.5 * (
            cm_data["avg_vx"] ** 2 + cm_data["avg_vy"] ** 2 + cm_data["avg_vz"] ** 2
        )
        km = ek.mean()

        xmin_options = [1, int(raw_sizes.min())]
        best_alpha = None
        best_xmin = None
        best_D = float("inf")

        for xm in xmin_options:
            alpha = compute_alpha_mle(raw_sizes, xm)
            try:
                fit_test = powerlaw.Fit(
                    raw_sizes,
                    discrete=True,
                    verbose=False,
                    xmin=xm,
                    parameter_ranges={
                        "power_law": {"alpha": (1, 10)},
                        "truncated_power_law": {
                            "alpha": (1, 10),
                            "Lambda": (1e-10, 10),
                        },
                        "exponential": {"Lambda": (1e-10, 10)},
                        "lognormal": {"mu": (-1, 10), "sigma": (1e-10, 10)},
                    },
                )
                D = fit_test.power_law.D
                if D < best_D:
                    best_D = D
                    best_alpha = alpha
                    best_xmin = xm
            except Exception:
                continue

        if best_alpha is None or best_xmin is None:
            continue

        alpha_mle = best_alpha
        xmin_calc = best_xmin

        try:
            fit = powerlaw.Fit(
                raw_sizes,
                discrete=True,
                verbose=False,
                xmin=xmin_calc,
                parameter_ranges={
                    "power_law": {"alpha": (1, 10)},
                    "truncated_power_law": {"alpha": (1, 10), "Lambda": (1e-10, 10)},
                    "exponential": {"Lambda": (1e-10, 10)},
                    "lognormal": {"mu": (-1, 10), "sigma": (1e-10, 10)},
                },
            )

            alpha_tpl = fit.truncated_power_law.alpha
            lambda_tpl = fit.truncated_power_law.Lambda
            lambda_exp = fit.exponential.Lambda
            mu_ln = fit.lognormal.mu
            sigma_ln = fit.lognormal.sigma

            results.append(
                {
                    "vel0": vel0,
                    "km": km,
                    "alpha_pl": alpha_mle,
                    "xmin_pl": xmin_calc,
                    "alpha_tpl": alpha_tpl,
                    "lambda_tpl": lambda_tpl,
                    "lambda_exp": lambda_exp,
                    "mu_ln": mu_ln,
                    "sigma_ln": sigma_ln,
                }
            )
        except Exception:
            continue

    if not results:
        return plt.figure()

    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values("km")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].plot(
        res_df["km"], res_df["alpha_pl"], "o-", color="#e74c3c", markersize=8
    )
    axes[0, 0].set_xlabel("K/m", fontsize=fs)
    axes[0, 0].set_ylabel(r"$\alpha$", fontsize=fs)
    axes[0, 0].set_title("Power Law: α", fontsize=fs * 0.9)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(
        res_df["km"], res_df["xmin_pl"], "o-", color="#e74c3c", markersize=8
    )
    axes[0, 1].set_xlabel("K/m", fontsize=fs)
    axes[0, 1].set_ylabel(r"$x_{min}$", fontsize=fs)
    axes[0, 1].set_title("Power Law: xmin", fontsize=fs * 0.9)
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(
        res_df["km"], res_df["alpha_tpl"], "o-", color="#2ecc71", markersize=8
    )
    axes[0, 2].set_xlabel("K/m", fontsize=fs)
    axes[0, 2].set_ylabel(r"$\alpha$", fontsize=fs)
    axes[0, 2].set_title("Truncated PL: α", fontsize=fs * 0.9)
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(
        res_df["km"], res_df["lambda_tpl"], "o-", color="#2ecc71", markersize=8
    )
    axes[1, 0].set_xlabel("K/m", fontsize=fs)
    axes[1, 0].set_ylabel(r"$\lambda$", fontsize=fs)
    axes[1, 0].set_title("Truncated PL: λ", fontsize=fs * 0.9)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(
        res_df["km"], res_df["lambda_exp"], "o-", color="#f39c12", markersize=8
    )
    axes[1, 1].set_xlabel("K/m", fontsize=fs)
    axes[1, 1].set_ylabel(r"$\lambda$", fontsize=fs)
    axes[1, 1].set_title("Exponential: λ", fontsize=fs * 0.9)
    axes[1, 1].grid(True, alpha=0.3)

    valid_ln = res_df[res_df["mu_ln"] >= -1]
    axes[1, 2].plot(
        valid_ln["km"],
        valid_ln["mu_ln"],
        "o-",
        color="#9b59b6",
        markersize=8,
        label="μ",
    )
    axes[1, 2].plot(
        valid_ln["km"],
        valid_ln["sigma_ln"],
        "s-",
        color="#3498db",
        markersize=8,
        label="σ",
    )
    axes[1, 2].set_xlabel("K/m", fontsize=fs)
    axes[1, 2].set_ylabel("Valor", fontsize=fs)
    axes[1, 2].set_title("Lognormal (μ ≥ -1)", fontsize=fs * 0.9)
    axes[1, 2].legend(fontsize=fs * 0.6)
    axes[1, 2].grid(True, alpha=0.3)

    fig.suptitle(f"Parámetros vs K/m (ff={ff}, i={I_VALUE})", fontsize=fs)
    plt.tight_layout()

    return fig


def guardar_pdf(figuras: list[Figure], salida: Path) -> None:
    """Guarda todas las figuras en un PDF multipágina.

    Args:
        figuras: Lista de objetos Figure de matplotlib.
        salida: Ruta del archivo PDF de salida.
    """
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(salida) as pdf:
        for fig in figuras:
            if fig and fig.get_axes():
                pdf.savefig(fig)
                plt.close(fig)


def main() -> None:
    """Punto de entrada del script de análisis."""
    print("Cargando datos...")
    df = cargar_datos()
    print(f"Datos cargados: {len(df)} filas")

    SALIDA_DIR.mkdir(parents=True, exist_ok=True)

    valores_ff = [0.15, 0.35]
    valores_vel0 = sorted(df["vel0"].unique())

    figuras: list[Figure] = []

    for ff in valores_ff:
        dir_ff = SALIDA_DIR / f"ff_{ff}"
        dir_ff.mkdir(parents=True, exist_ok=True)

        print(f"Procesando ff={ff}...")
        fig_ids_box = grafico_ids_boxplot(df, ff)
        if fig_ids_box and fig_ids_box.get_axes():
            figuras.append(fig_ids_box)
            plt.savefig(dir_ff / "ids_boxplot.png", bbox_inches="tight")
            plt.close(fig_ids_box)

        fig_ids = grafico_ids_por_vel0(df, ff)
        if fig_ids and fig_ids.get_axes():
            figuras.append(fig_ids)
            plt.savefig(dir_ff / "ids_por_vel0.png", bbox_inches="tight")
            plt.close(fig_ids)

        for vel0 in valores_vel0:
            print(f"  vel0={vel0}...")
            figs_ccdf = grafico_cluster_ccdf(df, ff, vel0)
            for i, fig_cc in enumerate(figs_ccdf):
                if fig_cc and fig_cc.get_axes():
                    figuras.append(fig_cc)
                    if i == 0:
                        plt.savefig(dir_ff / f"cluster_ccdf_{vel0}.png", bbox_inches="tight")
                    plt.close(fig_cc)

            fig_angulo = grafico_angulo_vel_cm(df, ff, vel0)
            if fig_angulo and fig_angulo.get_axes():
                figuras.append(fig_angulo)
                plt.savefig(dir_ff / f"angulo_vel_cm_{vel0}.png", bbox_inches="tight")
                plt.close(fig_angulo)

            fig_heatmap = grafico_heatmap_erosion(df, ff, vel0)
            if fig_heatmap and fig_heatmap.get_axes():
                figuras.append(fig_heatmap)
                plt.savefig(dir_ff / f"heatmap_erosion_{vel0}.png", bbox_inches="tight")
                plt.close(fig_heatmap)

        fig_energia = grafico_energia_km(df, ff)
        if fig_energia and fig_energia.get_axes():
            figuras.append(fig_energia)
            plt.savefig(dir_ff / "energia_km.png", bbox_inches="tight")
            plt.close(fig_energia)

        fig_penetracion = grafico_penetracion(df, ff)
        if fig_penetracion and fig_penetracion.get_axes():
            figuras.append(fig_penetracion)
            plt.savefig(dir_ff / "penetracion.png", bbox_inches="tight")
            plt.close(fig_penetracion)

        fig_params = grafico_parametros_vs_energia(df, ff)
        if fig_params and fig_params.get_axes():
            figuras.append(fig_params)
            plt.savefig(dir_ff / "parametros_vs_energia.png", bbox_inches="tight")
            plt.close(fig_params)

    pdf_final = SALIDA_DIR / "resultado_final.pdf"
    print(f"Generando PDF final: {pdf_final}")
    guardar_pdf(figuras, pdf_final)
    print(f"¡Listo! Figuras en {SALIDA_DIR}")


if __name__ == "__main__":
    main()
