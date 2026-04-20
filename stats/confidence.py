"""Módulo de intervalos de confianza para análisis de erosión.

Calcula ICs del centro de la distribución de erosión (Y) usando solo MAD
como filtro de outliers, con opción de mostrar IC Student o Error Estándar.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.stats import t

# Configuración por defecto
USE_STUDENT_IC: bool = False
N_MOV: int = 9
P_MAD: float = 2.5


def calc_ic(data: np.ndarray, use_student: bool = USE_STUDENT_IC) -> dict:
    """Calcula las métricas unificadas, evitando crasheos matemáticos.

    Args:
        data: Array de datos.
        use_student: Si True, usa IC de Student. Si False, error estándar.

    Returns:
        Dict con n, valor (con incertidumbre), error%.
    """
    n = len(data)
    if n < 2:
        return {
            "n": n,
            "v": np.mean(data) if n else 0.0,
            "err": 0.0,
        }

    mu = np.mean(data)
    se = np.std(data, ddof=1) / np.sqrt(n)

    if use_student:
        margin = t.ppf(0.975, n - 1) * se
    else:
        margin = se

    err = (margin / mu * 100) if mu != 0 else 0.0
    return {"n": n, "v": mu, "err": err}


def _calc_ic_polars(
    data: pl.Series,
    use_student: bool = USE_STUDENT_IC,
) -> dict:
    """Calcula métricas usando Polars Series.

    Args:
        data: Series de Polars.
        use_student: Si True, IC de Student.

    Returns:
        Dict con n, v, err.
    """
    arr = data.to_numpy()
    return calc_ic(arr, use_student)


def _style(ax, table, title: str) -> None:
    """Aplica formato universal a cualquier tabla matplotlib."""
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.axis("off")
    for (r, c), cell in table.get_celld().items():
        if c < 4:
            cell.set_width(0.08)
        cell.set_facecolor("#1e40af" if r == 0 else ("#fff" if r % 2 else "#f3f4f6"))
        if r == 0:
            cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("#ccc")
    table.scale(1.2, 1.5)


def run_confidence_intervals(
    df_raw: pl.DataFrame,
    ff_values: list[float] | None = None,
    n_mov: int = N_MOV,
    p_mad: float = P_MAD,
    use_student: bool = USE_STUDENT_IC,
) -> tuple[list, pl.DataFrame]:
    """Ejecuta el cálculo de intervalos de confianza.

    Args:
        df_raw: DataFrame con columnas: ff, vel0, id.
        ff_values: Lista de valores de ff a analizar.
        n_mov: Número de movimientos por simulación.
        p_mad: Multiplicador MAD para filtro de outliers.
        use_student: Si True, usa IC de Student.

    Returns:
        Tupla (figuras, DataFrame de resultados).
    """
    from matplotlib.figure import Figure

    records = []
    guardado_de_erosion_crudo = pl.DataFrame()

    ff_vals = ff_values or df_raw["ff"].unique().to_list()

    for ff in ff_vals:
        erosions = compute_unique_ids_raw(df_raw, ff)
        for vel0, ids_list in erosions.items():
            raw = np.array(ids_list).astype(float)
            data = np.concatenate([raw, np.zeros(max(0, n_mov - len(raw)))])

            med = np.median(data)
            mad = np.median(np.abs(data - med))
            filt = data if mad == 0 else data[np.abs(data - med) <= p_mad * mad]

            datos_totales = {
                f"Posición {i}": data[i] for i in range(len(data))
            }
            datos_totales.update({"ff": ff, "vel0": vel0})
            guardado_de_erosion_crudo = pl.concat(
                [guardado_de_erosion_crudo, pl.DataFrame([datos_totales])]
            )

            for label, arr in [("Crudos", data), ("MAD", filt)]:
                m = calc_ic(arr, use_student)
                records.append(
                    {
                        "ff": ff,
                        "vel0": vel0,
                        "Método": label,
                        "n": m["n"],
                        "err": round(m["err"], 1),
                        "v": m["v"],
                    }
                )

    guardado_de_erosion_crudo.write_csv("guardado_de_erosion_crudo.csv")
    df = pl.DataFrame(records)
    figs: list[Figure] = []

    return figs, df


def compute_unique_ids_raw(
    df: pl.DataFrame, ff: float
) -> dict[float, list[int]]:
    """Calcula IDs únicos por cada simulación individual.

    Nota: Esta es una copia de stats/clusters.compute_unique_ids_raw
    para evitar importación circular.

    Args:
        df: DataFrame con columnas: vel0, angulo, posx, posy, id.
        ff: Fraction fill.

    Returns:
        Dict con vel0 -> lista de IDs únicos.
    """
    df_filt = df.filter(
        (pl.col("ff") == ff)
        & (pl.col("i") == pl.col("i").max())
        & (pl.col("type") == 2)
        & ((pl.col("z") > 1e-4) | (pl.col("vz") > 0))
    )

    grouped = df_filt.group_by(["vel0", "angulo", "posx", "posy"]).agg(
        pl.col("id").n_unique().alias("unique_ids")
    )

    result: dict[float, list[int]] = {}
    for vel0 in grouped["vel0"].unique().to_list():
        result[vel0] = grouped.filter(pl.col("vel0") == vel0)[
            "unique_ids"
        ].to_list()
    return result