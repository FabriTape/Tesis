"""Módulo de intervalos de confianza para análisis de erosión.

Calcula ICs del centro de la distribución de erosión (Y) usando solo MAD
como filtro de outliers, con opción de mostrar IC Student o Error Estándar.
"""

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cluster_calc import compute_unique_ids_raw_with_positions
from constants import FF_VALUES
from data_loader import load_data
from matplotlib.figure import Figure
from scipy.stats import t

# Configuración: True = IC Student, False = Error Estándar (default)
USE_STUDENT_IC = False


def filter_mad(data: np.ndarray) -> np.ndarray:
    """Filtra outliers mediante Median Absolute Deviation.

    Elimina datos donde |Xi - Mediana| > 3 * MAD.
    Añade ceros para mantener 10 datos totales.
    Si después de filtrar todo es 0, usa los datos originales.

    Args:
        data: Array de muestras de erosión.

    Returns:
        Array con outliers eliminados y ceros añadidos para mantener n=10.
    """
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    filtered = data[np.abs(data - median) <= 3 * mad]

    if np.sum(filtered) == 0:
        filtered = data

    n_zeros = 10 - len(filtered)
    zeros = np.zeros(n_zeros)

    return np.concatenate([filtered, zeros])


def filter_tukey(data: np.ndarray) -> np.ndarray:
    """Filtra outliers mediante el rango intercuartílico (IQR).

    Elimina datos fuera de [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
    Añade ceros para mantener 10 datos totales.
    Si después de filtrar todo es 0, usa los datos originales.

    Args:
        data: Array de muestras de erosión.

    Returns:
        Array con outliers eliminados y ceros añadidos para mantener n=10.
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    filtered = data[(data >= lower) & (data <= upper)]

    if np.sum(filtered) == 0:
        filtered = data

    n_zeros = 10 - len(filtered)
    zeros = np.zeros(n_zeros)

    return np.concatenate([filtered, zeros])


def compute_student_ci(
    data: np.ndarray, alpha: float = 0.05
) -> Tuple[float, float, float]:
    """Calcula IC con distribución t-Student.

    Si mu > 0: err% = sigma/mu * 100
    Si mu = 0: err% = sigma/rango * 100

    Args:
        data: Array de muestras (ya incluye ceros).
        alpha: Nivel de significancia (default 0.05 para IC 95%).

    Returns:
        Tupla (lower, upper, err_porcentual).
    """
    from scipy.stats import t

    n = len(data)
    mu = np.mean(data)
    std = np.std(data, ddof=1)

    sigma = t.ppf(0.975, n - 1) * std / np.sqrt(n)

    if mu == 0:
        rango = np.max(data) - np.min(data)
        err_pct = (sigma / rango) * 100 if rango > 0 else 0
    else:
        err_pct = (sigma / mu) * 100

    return mu - sigma, mu + sigma, err_pct


def compute_binomial_ci(data: np.ndarray) -> Tuple[float, float, float]:
    """Calcula IC binomial usando estadísticos de orden.

    Si mu > 0: err% = sigma/mu * 100
    Si mu = 0: err% = sigma/rango * 100

    Args:
        data: Array de muestras (ya incluye ceros).

    Returns:
        Tupla (lower, upper, err_porcentual).
    """
    sorted_data = np.sort(data)
    n = len(sorted_data)

    if n < 3:
        return (0.0, 0.0, 0.0)

    X_lo = sorted_data[1]
    X_hi = sorted_data[-2]
    mu = np.median(sorted_data)
    sigma = (X_hi - X_lo) / 2

    if mu == 0:
        rango = np.max(data) - np.min(data)
        err_pct = (sigma / rango) * 100 if rango > 0 else 0
    else:
        err_pct = (sigma / mu) * 100

    return X_lo, X_hi, err_pct


def compute_bootstrap_ci(
    data: np.ndarray, B: int = 200_000
) -> Tuple[float, float, float]:
    """Calcula IC por remuestreo empírico (bootstrap).

    Si mu > 0: err% = sigma/mu * 100
    Si mu = 0: err% = sigma/rango * 100

    Args:
        data: Array de muestras.
        B: Número de remuestras bootstrap.

    Returns:
        Tupla (lower, upper, err_porcentual).
    """
    n = len(data)
    mu = np.mean(data)

    if n < 2:
        return (0.0, 0.0, 0.0)

    indices = np.random.randint(0, n, size=(B, n))
    bootstrap_means = data[indices].mean(axis=1)

    lo = float(np.percentile(bootstrap_means, 2.5))
    hi = float(np.percentile(bootstrap_means, 97.5))
    sigma = (hi - lo) / 2

    if mu == 0:
        rango = np.max(data) - np.min(data)
        err_pct = (sigma / rango) * 100 if rango > 0 else 0
    else:
        err_pct = ((hi - lo) / (2 * mu)) * 100

    return lo, hi, err_pct


def compute_standard_error(data: np.ndarray) -> Tuple[float, float, float]:
    """Calcula error estándar (SE = s/√n) y err%.

    Args:
        data: Array de muestras.

    Returns:
        Tupla (mean - SE, mean + SE, err_porcentual).
    """
    n = len(data)
    mu = np.mean(data)
    std = np.std(data, ddof=1)
    se = std / np.sqrt(n)

    if mu == 0:
        rango = np.max(data) - np.min(data)
        err_pct = (se / rango) * 100 if rango > 0 else 0
    else:
        err_pct = (se / mu) * 100

    return mu - se, mu + se, err_pct


def compute_error_pct(lower: float, upper: float, mean: float) -> float:
    """Calcula el error porcentual de la semiamplitud.

    err % = ((LimSup - LimInf) / (2 * Media)) * 100

    Args:
        lower: Límite inferior del IC.
        upper: Límite superior del IC.
        mean: Media de la muestra.

    Returns:
        Error porcentual.
    """
    if mean == 0:
        return 0.0
    return ((upper - lower) / (2 * mean)) * 100


def build_table_row(
    n: int,
    mean: float,
    std: float,
    t_crit: float,
    main_ci: Tuple[float, float, float],
) -> dict:
    """Construye una fila de la tabla con IC Student o Error Estándar.

    Args:
        n: Tamaño de la muestra.
        mean: Media muestral.
        std: Desviación estándar muestral.
        t_crit: Valor crítico t-Student.
        main_ci: Tupla (lower, upper, err_pct) del IC seleccionado (Student o SE).

    Returns:
        Diccionario con las columnas de la tabla.
    """
    se = std / np.sqrt(n)
    label = "IC Student" if USE_STUDENT_IC else "Media ± SE"

    if mean == 0:
        return {
            "Tamaño n": n,
            label: "0.0000 ± 0.0000",
            "err %": "—",
        }

    if USE_STUDENT_IC:
        margin = t_crit * se
    else:
        margin = se

    return {
        "Tamaño n": n,
        label: f"{mean:.4f} ± {margin:.4f}",
        "err %": f"{main_ci[2]:.2f}",
    }


def compute_ics_for_sample(data: np.ndarray) -> dict:
    """Calcula ICs y métricas para un conjunto de muestras.

    Args:
        data: Array de muestras de erosión.

    Returns:
        Diccionario con fila de tabla lista para DataFrame.
    """
    n = len(data)
    mean = float(np.mean(data))
    std = float(np.std(data, ddof=1))
    t_crit = float(t.ppf(0.975, df=n - 1))

    if USE_STUDENT_IC:
        main_ci = compute_student_ci(data)
    else:
        main_ci = compute_standard_error(data)

    return build_table_row(n, mean, std, t_crit, main_ci)


def compute_ics_numeric(data: np.ndarray) -> dict:
    """Calcula métricas numéricas del IC seleccionado (Student o SE).

    Args:
        data: Array de muestras de erosión.

    Returns:
        Diccionario con n, mean, std y límites del IC.
    """
    n = len(data)
    mean = float(np.mean(data))
    std = float(np.std(data, ddof=1))

    unique_vals = len(set(data))
    if mean == 0 or unique_vals < 2:
        return {
            "n": n,
            "mean": mean,
            "std": std,
            "err": 0.0,
            "ci_lo": 0.0,
            "ci_hi": 0.0,
        }

    if USE_STUDENT_IC:
        ci = compute_student_ci(data)
    else:
        ci = compute_standard_error(data)

    return {
        "n": n,
        "mean": mean,
        "std": std,
        "err": ci[2],
        "ci_lo": ci[0],
        "ci_hi": ci[1],
    }


def _round_1sf(val: float) -> str:
    """Redondea un valor a 1 cifra significativa.

    Args:
        val: Número a redondear.

    Returns:
        String con 1 cifra significativa.
    """
    if val == 0 or np.isnan(val):
        return "—" if np.isnan(val) else "0"
    exp = int(np.floor(np.log10(abs(val))))
    rounded = round(val, -exp)
    if exp >= 0:
        return str(int(rounded))
    decimals = max(0, -exp)
    return f"{rounded:.{decimals}f}"


def create_per_ff_consolidated_figure(ff: float, results: list[dict]) -> Figure:
    fs = 9
    sorted_results = sorted(results, key=lambda r: r["vel0"], reverse=True)
    vel_values = [str(int(r["vel0"])) for r in sorted_results]

    blocks = [("raw", "Crudos"), ("mad", "MAD")]

    ci_label = "IC Student" if USE_STUDENT_IC else "Media ± SE"

    def _get_intervals(res, block_key):
        m = res.get(block_key)
        if not m:
            return "—"
        mean = m.get("mean", 0)
        margin = (m['ci_hi'] - m['ci_lo']) / 2
        return f"{mean:.2f} ± {margin:.2f}"

    def _get_n(r, block_key):
        if block_key == "raw":
            return r.get("n_original", 10)
        elif block_key == "mad":
            n_original = r.get("n_original", 10)
            n_outliers = r.get("n_outliers_mad", 0)
            return n_original - n_outliers
        return "—"

    cell_text = []
    for block_key, b_label in blocks:
        for r in sorted_results:
            m = r.get(block_key)
            n = _get_n(r, block_key)
            interval = _get_intervals(r, block_key)
            cell_text.append(
                [
                    ff,
                    int(r["vel0"]),
                    n,
                    b_label,
                    interval,
                ]
            )

    col_labels = [
        "ff",
        "vel",
        "N",
        "método",
        ci_label,
    ]

    fig, ax = plt.subplots(
        figsize=(3 + 2.0 * len(vel_values) * len(blocks), 0.8 + 0.45 * len(cell_text))
    )
    ax.set_title(
        f"Intervalos de Confianza — ff = {ff}",
        fontsize=fs + 3,
        fontweight="bold",
        pad=12,
    )
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fs - 1)
    table.scale(1.2, 1.5)

    for (r, c), cell in table.get_celld().items():
        if c < 4:
            cell.set_width(0.08)
        if r == 0:
            cell.set_facecolor("#1e40af")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            if r % 2 == 1:
                cell.set_facecolor("#ffffff")
            else:
                cell.set_facecolor("#f3f4f6")
        cell.set_edgecolor("#cccccc")

    plt.tight_layout()
    return fig


def build_master_table_figure(all_results: list[dict]) -> Figure:
    col_labels = [
        "ff",
        "vel",
        "N",
        "método",
        "err %",
    ]

    def _get_errors(m):
        if not m or m.get("err", 0) == 0:
            return "—"
        return round(m.get("err", 0), 2)

    def _get_n(r, block_key):
        if block_key == "raw":
            return r.get("n_original", 10)
        elif block_key == "mad":
            n_original = r.get("n_original", 10)
            n_outliers = r.get("n_outliers_mad", 0)
            return n_original - n_outliers
        return "—"

    rows = []
    for r in all_results:
        ff = r["ff"]
        vel0 = int(r["vel0"])
        for block_key, b_label in [
            ("raw", "Crudos"),
            ("mad", "MAD"),
        ]:
            m = r.get(block_key)
            if m:
                rows.append(
                    [
                        ff,
                        vel0,
                        _get_n(r, block_key),
                        b_label,
                        _get_errors(m),
                    ]
                )
            else:
                rows.append([ff, vel0, "—", b_label, "—"])

    df = pd.DataFrame(rows, columns=col_labels)
    df = df.sort_values(["ff", "vel"], ascending=[True, False])

    fs = 10
    fig, ax = plt.subplots(figsize=(4 + 1.2 * len(df.columns), 0.8 + 0.45 * len(df)))
    ax.set_title("Errores Porcentuales", fontsize=fs + 3, fontweight="bold", pad=12)
    ax.axis("off")

    cell_text = df.values

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )

    for (r, c), cell in table.get_celld().items():
        if c < 4:
            cell.set_width(0.08)
        if r == 0:
            cell.set_facecolor("#1e40af")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            if r % 2 == 1:
                cell.set_facecolor("#ffffff")
            else:
                cell.set_facecolor("#f3f4f6")
        cell.set_edgecolor("#cccccc")

    table.scale(1.2, 1.5)
    return fig


def create_sensitivity_figure(all_results: list[dict]) -> Figure:
    """Genera gráfico de barras agrupadas de sensibilidad de error %.

    Eje X: vel0 (-50, -25, -10, -5).
    Hue: 2 métodos (Crudos, MAD).
    Paneles: ff=0.15 y ff=0.35.

    Args:
        all_results: Lista de dicts con métricas numéricas por ff/vel0.

    Returns:
        Figura matplotlib con el catplot de sensibilidad.
    """
    records = []
    for r in all_results:
        ff = r["ff"]
        vel0 = int(r["vel0"])

        records.append(
            {
                "ff": ff,
                "vel0": vel0,
                "Método": "Crudos",
                "err%": r["raw"]["err"],
            }
        )

        if r["mad"]:
            records.append(
                {
                    "ff": ff,
                    "vel0": vel0,
                    "Método": "MAD",
                    "err%": r["mad"]["err"],
                }
            )

    df = pd.DataFrame(records)
    df["vel0"] = df["vel0"].astype(int)

    vel_order = sorted(df["vel0"].unique(), reverse=True)
    method_order = ["Crudos", "MAD"]

    palette = {
        "Crudos": "#e74c3c",
        "MAD": "#2ecc71",
    }

    g = sns.catplot(
        data=df,
        x="vel0",
        y="err%",
        hue="Método",
        col="ff",
        kind="bar",
        height=5,
        aspect=1.3,
        palette=palette,
        order=vel_order,
        hue_order=method_order,
        facet_kws={"despine": True},
    )
    if g.legend is not None:
        g.legend.remove()

    for ax in g.axes.flat:
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    fs = 12

    for ax in g.axes.flat:
        ax.set_xlabel("vel0", fontsize=fs)
        ax.set_ylabel("err %", fontsize=fs)
        ax.tick_params(labelsize=fs * 0.8)
        ax.legend(fontsize=fs * 0.7, title_fontsize=fs * 0.8)
        ax.grid(True, alpha=0.3, axis="y")

    g.fig.suptitle(
        "Sensibilidad del Error % por Método de IC y Filtro",
        fontsize=fs + 2,
        fontweight="bold",
        y=1.005,
    )
    g.fig.tight_layout()
    return g.fig


def run_confidence_intervals(i_value: int) -> list[Figure]:
    """Ejecuta el análisis completo de intervalos de confianza.

    Args:
        i_value: Paso temporal a usar para filtrar los datos.

    Returns:
        Lista de figuras matplotlib con las tablas de ICs.
    """
    print("Cargando datos...")
    df = load_data()
    print(f"Datos cargados: {len(df)} filas\n")

    I_VALUE = i_value

    print(f"I_VALUE: {I_VALUE}\n")

    figures: list[Figure] = []
    all_results: list[dict] = []

    for ff in FF_VALUES:
        erosion_data = compute_unique_ids_raw_with_positions(df, ff, I_VALUE)
        print(f"{'=' * 100}")
        print(f"Factor de fricción: ff = {ff}")
        print(f"{'=' * 100}\n")

        ff_results: list[dict] = []

        for vel0 in sorted(erosion_data.keys()):
            sim_df = erosion_data[vel0]
            raw = sim_df["unique_ids"].values.astype(float)
            n_original = len(raw)

            # Añadir ceros justo antes de calcular ICs
            n_zeros = 10 - n_original
            if n_zeros > 0:
                raw_with_zeros = np.concatenate([raw, np.zeros(n_zeros)])
            else:
                raw_with_zeros = raw

            km = (vel0**2) / 2

            print(f"{'─' * 100}")
            print(f"K/m = {km:.1f} J/kg | vel0 = {vel0} | n_original = {n_original}")
            print(f"{'─' * 100}\n")

            # Calcular outliers filtrados para MAD (sobre datos ORIGINALES)
            median_raw = np.median(raw)
            mad_raw = np.median(np.abs(raw - median_raw))
            filtered_raw_mad = raw[np.abs(raw - median_raw) <= 3 * mad_raw]
            n_outliers_mad = n_original - len(filtered_raw_mad)

            row_raw = compute_ics_for_sample(raw_with_zeros)
            row_raw["Tamaño n"] = n_original
            df_crudos = pd.DataFrame([row_raw])
            print("TABLA 1: DATOS CRUDOS")
            print(df_crudos.to_string(index=False))
            print()

            mad_filtered = filter_mad(raw_with_zeros)
            df_mad = None
            if len(mad_filtered) >= 2:
                row_mad = compute_ics_for_sample(mad_filtered)
                row_mad["Tamaño n"] = n_original - n_outliers_mad
                df_mad = pd.DataFrame([row_mad])
                print(f"TABLA 2: FILTRO MAD (n={n_original - n_outliers_mad})")
                print(df_mad.to_string(index=False))
            else:
                print(
                    f"TABLA 2: FILTRO MAD — datos insuficientes tras filtrar (n={len(mad_filtered)})"
                )
            print()

            # (n_outliers_mad ya calculado arriba)

            raw_metrics = compute_ics_numeric(raw_with_zeros)
            mad_metrics = (
                compute_ics_numeric(mad_filtered) if len(mad_filtered) >= 2 else None
            )

            all_results.append(
                {
                    "ff": ff,
                    "vel0": vel0,
                    "n_original": n_original,
                    "n_outliers_mad": n_outliers_mad,
                    "raw": raw_metrics,
                    "mad": mad_metrics,
                }
            )

            ff_results.append(
                {
                    "vel0": vel0,
                    "n_original": n_original,
                    "n_outliers_mad": n_outliers_mad,
                    "raw": raw_metrics,
                    "mad": mad_metrics,
                }
            )

        fig_ff = create_per_ff_consolidated_figure(ff, ff_results)
        figures.append(fig_ff)

    fig_master = build_master_table_figure(all_results)
    figures.insert(0, fig_master)

    fig_sensitivity = create_sensitivity_figure(all_results)
    figures.insert(0, fig_sensitivity)

    return figures


if __name__ == "__main__":
    run_confidence_intervals()
