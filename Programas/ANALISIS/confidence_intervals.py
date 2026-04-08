"""Módulo de intervalos de confianza para análisis de erosión.

Calcula ICs del centro de la distribución de erosión (Y) mediante tres
métodos (t-Student, Binomial, Bootstrap) sobre datos crudos y tras aplicar
filtros de outliers (MAD y Tukey), generando tablas comparativas como
figuras matplotlib y por consola.
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


def filter_mad(data: np.ndarray) -> np.ndarray:
    """Filtra outliers mediante Median Absolute Deviation.

    Elimina datos donde |Xi - Mediana| > 3 * MAD.

    Args:
        data: Array de muestras de erosión.

    Returns:
        Array sin outliers detectados por MAD.
    """
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    return data[np.abs(data - median) <= 3 * mad]


def filter_tukey(data: np.ndarray) -> np.ndarray:
    """Filtra outliers mediante el rango intercuartílico (IQR).

    Elimina datos fuera de [Q1 - 1.5*IQR, Q3 + 1.5*IQR].

    Args:
        data: Array de muestras de erosión.

    Returns:
        Array sin outliers detectados por Tukey.
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return data[(data >= lower) & (data <= upper)]


def compute_student_ci(data: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    """Calcula IC paramétrico con distribución t-Student.

    Fórmula: X̄ ± t_{n-1, α/2} * σ̂ / √n

    Args:
        data: Array de muestras.
        alpha: Nivel de significancia (default 0.05 para IC 95%).

    Returns:
        Tupla (limite_inferior, limite_superior).
    """
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    t_crit = t.ppf(1 - alpha / 2, df=n - 1)
    margin = t_crit * std / np.sqrt(n)
    return mean - margin, mean + margin


def compute_binomial_ci(sorted_data: np.ndarray) -> Tuple[float, float]:
    """Calcula IC no paramétrico con estadísticos de orden.

    Usa [X_(2), X_(n-1)] del arreglo ordenado.

    Args:
        sorted_data: Array ordenado de menor a mayor.

    Returns:
        Tupla (limite_inferior, limite_superior).
    """
    return sorted_data[1], sorted_data[len(sorted_data) - 2]


def compute_bootstrap_ci(data: np.ndarray, B: int = 100_000) -> Tuple[float, float]:
    """Calcula IC por remuestreo empírico (bootstrap) vectorizado.

    Genera B medias con reemplazo y toma percentiles 2.5 y 97.5.

    Args:
        data: Array de muestras.
        B: Número de remuestras bootstrap.

    Returns:
        Tupla (limite_inferior, limite_superior).
    """
    n = len(data)
    indices = np.random.randint(0, n, size=(B, n))
    bootstrap_means = data[indices].mean(axis=1)
    return (
        float(np.percentile(bootstrap_means, 2.5)),
        float(np.percentile(bootstrap_means, 97.5)),
    )


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
    return ((upper - lower) / (2 * mean)) * 100


def build_table_row(
    n: int,
    mean: float,
    std: float,
    t_crit: float,
    student_ci: Tuple[float, float],
    binomial_ci: Tuple[float, float],
    bootstrap_ci: Tuple[float, float],
) -> dict:
    """Construye una fila de la tabla comparativa de ICs.

    Args:
        n: Tamaño de la muestra.
        mean: Media muestral.
        std: Desviación estándar muestral.
        t_crit: Valor crítico t-Student.
        student_ci: Tupla (lower, upper) del IC t-Student.
        binomial_ci: Tupla (lower, upper) del IC Binomial.
        bootstrap_ci: Tupla (lower, upper) del IC Bootstrap.

    Returns:
        Diccionario con las columnas de la tabla.
    """
    student_err = compute_error_pct(student_ci[0], student_ci[1], mean)
    binomial_err = compute_error_pct(binomial_ci[0], binomial_ci[1], mean)
    bootstrap_err = compute_error_pct(bootstrap_ci[0], bootstrap_ci[1], mean)

    return {
        "Tamaño n": n,
        "Media ± t·σ/√n": f"{mean:.4f} ± {t_crit * std / np.sqrt(n):.4f}",
        "err % Student": f"{student_err:.2f}",
        "IC Binomial": f"[{binomial_ci[0]:.4f}, {binomial_ci[1]:.4f}]",
        "err % Binomial": f"{binomial_err:.2f}",
        "IC Bootstrap": f"[{bootstrap_ci[0]:.4f}, {bootstrap_ci[1]:.4f}]",
        "err % Bootstrap": f"{bootstrap_err:.2f}",
    }


def compute_ics_for_sample(data: np.ndarray) -> dict:
    """Calcula los tres ICs y métricas para un conjunto de muestras.

    Args:
        data: Array de muestras de erosión.

    Returns:
        Diccionario con fila de tabla lista para DataFrame.
    """
    n = len(data)
    mean = float(np.mean(data))
    std = float(np.std(data, ddof=1))
    t_crit = float(t.ppf(0.975, df=n - 1))

    sorted_data = np.sort(data)
    student_ci = compute_student_ci(data)
    binomial_ci = compute_binomial_ci(sorted_data)
    bootstrap_ci = compute_bootstrap_ci(data)

    return build_table_row(n, mean, std, t_crit, student_ci, binomial_ci, bootstrap_ci)


def compute_ics_numeric(data: np.ndarray) -> dict:
    """Calcula métricas numéricas de ICs (sin formateo de string).

    Args:
        data: Array de muestras de erosión.

    Returns:
        Diccionario con n, mean, std y límites de cada IC.
    """
    n = len(data)
    mean = float(np.mean(data))
    std = float(np.std(data, ddof=1))

    sorted_data = np.sort(data)
    student_ci = compute_student_ci(data)
    binomial_ci = compute_binomial_ci(sorted_data)
    bootstrap_ci = compute_bootstrap_ci(data)

    err_student = compute_error_pct(student_ci[0], student_ci[1], mean)
    err_binomial = compute_error_pct(binomial_ci[0], binomial_ci[1], mean)
    err_bootstrap = compute_error_pct(bootstrap_ci[0], bootstrap_ci[1], mean)

    return {
        "n": n,
        "mean": mean,
        "std": std,
        "err_student": err_student,
        "err_binomial": err_binomial,
        "err_bootstrap": err_bootstrap,
        "student_lo": student_ci[0],
        "student_hi": student_ci[1],
        "binomial_lo": binomial_ci[0],
        "binomial_hi": binomial_ci[1],
        "bootstrap_lo": bootstrap_ci[0],
        "bootstrap_hi": bootstrap_ci[1],
    }


def _round_1sf(val: float) -> str:
    """Redondea un valor a 1 cifra significativa.

    Args:
        val: Número a redondear.

    Returns:
        String con 1 cifra significativa.
    """
    if val == 0:
        return "0"
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

    # Mapeo de claves a etiquetas y bloques
    metrics_map = {"S": "Student", "B": "Bootstrap", "Bin": "Binomial"}
    blocks = [("raw", "Crudos"), ("mad", "MAD"), ("tukey", "Tukey")]

    def _get_val(res, block_key, metric_key):
        m = res.get(block_key)
        if not m:
            return "—"

        # Helper para formatear intervalos
        f = lambda lo, hi: f"{_round_1sf((lo + hi) / 2)} ± {_round_1sf((hi - lo) / 2)}"

        if metric_key == "S":
            return f(m["student_lo"], m["student_hi"])
        if metric_key == "B":
            return f(m["bootstrap_lo"], m["bootstrap_hi"])
        return f"[{_round_1sf(m['binomial_lo'])}, {_round_1sf(m['binomial_hi'])}]"

    # Construcción dinámica de etiquetas y texto
    row_labels = [
        f"{b_label} {metrics_map[m]}"
        for m in ["S", "B", "Bin"]
        for _, b_label in blocks
    ]
    cell_text = [
        [_get_val(r, b, m) for r in sorted_results]
        for m in ["S", "B", "Bin"]
        for b, _ in blocks
    ]

    fig, ax = plt.subplots(
        figsize=(4 + 2.5 * len(vel_values), 1.2 + 0.45 * len(row_labels))
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
        rowLabels=row_labels,
        colLabels=vel_values,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fs - 1)
    table.scale(1.2, 1.5)

    for (r, c), cell in table.get_celld().items():
        # 1. Estilo para el Encabezado de Columnas (vel0)
        if r == 0:
            cell.set_facecolor("#1e40af")  # Azul oscuro para jerarquía superior
            cell.set_text_props(color="white", fontweight="bold")

        # 2. Estilo para las etiquetas de fila (Crudos/MAD/Tukey + Método)
        elif c == -1:
            cell.set_facecolor("#3b82f6")  # Azul medio
            cell.set_text_props(
                color="white", fontweight="bold", ha="right", fontsize=fs - 2
            )

        # 3. Estilo para el cuerpo de la tabla (Bloques de datos)
        else:
            # Agrupación por Filtro (Cada 3 filas cambia el bloque)
            # Ajustamos los índices: r=1,2,3 -> Crudos; r=4,5,6 -> MAD; r=7,8,9 -> Tukey
            if 1 <= r <= 3:
                cell.set_facecolor("#ffffff")  # Blanco para Crudos
            elif 4 <= r <= 6:
                cell.set_facecolor("#f9fafb")  # Gris muy tenue para MAD
            elif 7 <= r <= 9:
                cell.set_facecolor("#f3f4f6")  # Gris tenue para Tukey

        # 4. Líneas de división de bloques y bordes generales
        cell.set_edgecolor("#cccccc")
        if r in (4, 7):
            cell.set_linewidth(2.0)
            cell.set_edgecolor("#4b5563")  # Separador de bloque más fuerte

    plt.tight_layout()
    return fig


def build_master_table_figure(all_results: list[dict]) -> Figure:
    # 1. Crear el DataFrame con MultiIndex en las columnas
    columns = pd.MultiIndex.from_tuples(
        [
            ("Parámetros", "ff"),
            ("Parámetros", "vel0"),
            ("Crudos", "n"),
            ("Crudos", "err% S"),
            ("Crudos", "err% B"),
            ("MAD", "n"),
            ("MAD", "err% S"),
            ("MAD", "err% B"),
            ("Tukey", "n"),
            ("Tukey", "err% S"),
            ("Tukey", "err% B"),
        ]
    )

    rows = []
    for r in all_results:
        # Extraemos los datos puros (sin formatear a string todavía) [cite: 4, 6]
        res = [r["ff"], int(r["vel0"])]
        for b in ["raw", "mad", "tukey"]:
            m = r.get(b)
            if m:
                res.extend(
                    [
                        int(m["n"]),
                        round(m["err_student"], 2),
                        round(m["err_bootstrap"], 2),
                    ]
                )
            else:
                res.extend([None, None, None])
        rows.append(res)

    df = pd.DataFrame(rows, columns=columns).sort_values(
        [("Parámetros", "ff"), ("Parámetros", "vel0")], ascending=[True, False]
    )

    # 2. El truco de la doble fila de encabezado
    # ax.table no soporta MultiIndex nativo, así que inyectamos los nombres de los filtros
    # manualmente sobre la primera fila de la tabla de Matplotlib
    fig, ax = plt.subplots(figsize=(20, 1 + 0.4 * len(df)))
    ax.axis("off")

    # Combinamos los niveles del MultiIndex para las etiquetas de columna
    col_labels = [f"{c[0]}\n{c[1]}" for c in df.columns]

    # Formateamos los datos para la tabla (reemplazando None por '—') [cite: 4, 12]
    cell_text = df.fillna("—").values

    table = ax.table(
        cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center"
    )

    # Estilo: Resaltamos las columnas de los parámetros físicos [cite: 2, 10]
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1e40af")  # Azul más oscuro para encabezados
            cell.set_text_props(color="white", fontweight="bold")
        elif c in [0, 1]:  # ff y vel0
            cell.set_facecolor("#f8fafc")
            cell.set_text_props(fontweight="bold")

    table.scale(1, 2)
    return fig


def create_sensitivity_figure(all_results: list[dict]) -> Figure:
    """Genera gráfico de barras agrupadas de sensibilidad de error %.

    Eje X: vel0 (-50, -25, -10, -5).
    Hue: 6 métodos (Crudo/MAD/Tukey × Student/Bootstrap).
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
                "Método": "Crudo Student",
                "err%": r["raw"]["err_student"],
            }
        )
        records.append(
            {
                "ff": ff,
                "vel0": vel0,
                "Método": "Crudo Bootstrap",
                "err%": r["raw"]["err_bootstrap"],
            }
        )

        if r["mad"]:
            records.append(
                {
                    "ff": ff,
                    "vel0": vel0,
                    "Método": "MAD Student",
                    "err%": r["mad"]["err_student"],
                }
            )
            records.append(
                {
                    "ff": ff,
                    "vel0": vel0,
                    "Método": "MAD Bootstrap",
                    "err%": r["mad"]["err_bootstrap"],
                }
            )

        if r["tukey"]:
            records.append(
                {
                    "ff": ff,
                    "vel0": vel0,
                    "Método": "Tukey Student",
                    "err%": r["tukey"]["err_student"],
                }
            )
            records.append(
                {
                    "ff": ff,
                    "vel0": vel0,
                    "Método": "Tukey Bootstrap",
                    "err%": r["tukey"]["err_bootstrap"],
                }
            )

    df = pd.DataFrame(records)
    df["vel0"] = df["vel0"].astype(int)

    vel_order = sorted(df["vel0"].unique(), reverse=True)
    method_order = [
        "Crudo Student",
        "Crudo Bootstrap",
        "MAD Student",
        "MAD Bootstrap",
        "Tukey Student",
        "Tukey Bootstrap",
    ]

    palette = {
        "Crudo Student": "#e74c3c",
        "Crudo Bootstrap": "#c0392b",
        "MAD Student": "#2ecc71",
        "MAD Bootstrap": "#27ae60",
        "Tukey Student": "#3498db",
        "Tukey Bootstrap": "#2980b9",
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
    # 1. LA OPCIÓN NUCLEAR: Destruir cualquier leyenda residual
    # Primero, eliminamos la leyenda global si Seaborn la forzó de todos modos
    if g.legend is not None:
        g.legend.remove()

    # Segundo, iteramos por cada panel y matamos las leyendas locales
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
    # NUEVO: Modificar la leyenda global de la figura (fuera del bucle for)

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
            km = (vel0**2) / 2

            print(f"{'─' * 100}")
            print(f"K/m = {km:.1f} J/kg | vel0 = {vel0} | n_original = {len(raw)}")
            print(f"{'─' * 100}\n")

            row_raw = compute_ics_for_sample(raw)
            df_crudos = pd.DataFrame([row_raw])
            print("TABLA 1: DATOS CRUDOS")
            print(df_crudos.to_string(index=False))
            print()

            mad_filtered = filter_mad(raw)
            df_mad = None
            if len(mad_filtered) >= 3:
                row_mad = compute_ics_for_sample(mad_filtered)
                df_mad = pd.DataFrame([row_mad])
                print(f"TABLA 2: FILTRO MAD (n={len(mad_filtered)})")
                print(df_mad.to_string(index=False))
            else:
                print(
                    f"TABLA 2: FILTRO MAD — datos insuficientes tras filtrar (n={len(mad_filtered)})"
                )
            print()

            tukey_filtered = filter_tukey(raw)
            df_tukey = None
            if len(tukey_filtered) >= 3:
                row_tukey = compute_ics_for_sample(tukey_filtered)
                df_tukey = pd.DataFrame([row_tukey])
                print(f"TABLA 3: FILTRO TUKEY (n={len(tukey_filtered)})")
                print(df_tukey.to_string(index=False))
            else:
                print(
                    f"TABLA 3: FILTRO TUKEY — datos insuficientes tras filtrar (n={len(tukey_filtered)})"
                )
            print()

            raw_metrics = compute_ics_numeric(raw)
            mad_metrics = (
                compute_ics_numeric(mad_filtered) if len(mad_filtered) >= 3 else None
            )
            tukey_metrics = (
                compute_ics_numeric(tukey_filtered)
                if len(tukey_filtered) >= 3
                else None
            )

            all_results.append(
                {
                    "ff": ff,
                    "vel0": vel0,
                    "raw": raw_metrics,
                    "mad": mad_metrics,
                    "tukey": tukey_metrics,
                }
            )

            ff_results.append(
                {
                    "vel0": vel0,
                    "raw": raw_metrics,
                    "mad": mad_metrics,
                    "tukey": tukey_metrics,
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
