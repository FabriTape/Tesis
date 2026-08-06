"""Módulo de intervalos de confianza para análisis de erosión.

Calcula ICs del centro de la distribución de erosión (Y) usando solo MAD
como filtro de outliers, con opción de mostrar IC Student o Error Estándar.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cluster_calc import compute_unique_ids_raw
from constants import FF_VALUES
from kinematics_calc import get_global_positions, get_max_positions
from scipy.stats import t
from uncertainties import ufloat, ufloat_fromstr

# Configuración
USE_STUDENT_IC, P_MAD = False, 2.5


def calc_ic(data: np.ndarray) -> dict:
    """Calcula las métricas unificadas, evitando crasheos matemáticos."""
    n = len(data)
    if n < 2:
        return {"n": n, "v": ufloat(np.mean(data) if n else 0, 0), "err": 0.0}
    mu, se = np.mean(data), np.std(data, ddof=1) / np.sqrt(n)
    margin = t.ppf(0.975, n - 1) * se if USE_STUDENT_IC else se
    valor = ufloat_fromstr(f"{ufloat(mu, margin):.1u}")
    err = (
        (valor.std_dev / valor.nominal_value * 100) if valor.nominal_value != 0 else 0.0
    )
    return {"n": n, "v": valor, "err": float(err)}


def _style(ax, table, title):
    """Aplica formato universal a cualquier tabla matplotlib del script."""
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


def run_confidence_intervals(df_raw):
    records = []
    guardado_de_erosion_crudo = pd.DataFrame()

    for ff in FF_VALUES:
        global_grid = get_global_positions(df_raw, ff)
        max_data = get_max_positions(df_raw, ff)
        erosions = compute_unique_ids_raw(df_raw, ff, global_grid, max_data)
        for vel0, sim_arr in erosions.items():
            data = sim_arr[~np.isnan(sim_arr)]
            raw = np.array(data).astype(float)

            med = np.median(data)
            mad = np.median(np.abs(data - med))
            filt = data if mad == 0 else data[np.abs(data - med) <= P_MAD * mad]

            print(f"\n--- DEBUG: ff={ff}, vel0={vel0} ---")
            print(f"Valores Crudos (N={len(data)}): {np.sort(data)}")
            print(f"Mediana: {med:.1f}")
            print(f"MAD: {mad:.1f}")
            print(
                f"Ventana de aceptación: [{med - P_MAD * mad:.1f}, {med + P_MAD * mad:.1f}]"
            )
            print(f"Se filtraron {len(data) - len(filt)} datos.")
            datos_totales = {"Posición " + str(i): data[i] for i in range(len(data))}
            datos_totales.update({"ff": ff, "vel0": int(vel0)})
            guardado_de_erosion_crudo = pd.concat(
                [
                    guardado_de_erosion_crudo,
                    pd.DataFrame([datos_totales]),
                ]
            )
            for label, arr in [("Crudos", data), ("MAD", filt)]:
                m = calc_ic(arr)
                records.append(
                    {
                        "ff": ff,
                        "vel0": int(vel0),
                        "Método": label,
                        "n": m["n"],
                        "err": round(m["err"], 1),
                        "v": f"{m['v']:.1u}",
                    }
                )
    guardado_de_erosion_crudo.to_csv("guardado_de_erosion_crudo.csv", index=False)
    df = pd.DataFrame(records)
    figs = []

    # 2. Generación de gráfico de barras
    g = sns.catplot(
        data=df,
        x="vel0",
        y="err",
        hue="Método",
        col="ff",
        kind="bar",
        height=5,
        aspect=1.3,
        palette={"Crudos": "#e74c3c", "MAD": "#2ecc71"},
    )
    g.fig.suptitle("Sensibilidad del Error % por Método", y=1.05, weight="bold")
    figs.append(g.fig)
    karws = {
        "colLabels": ["ff", "vel", "N", "Método", "IC"],
        "loc": "center",
    }
    # 3. Generación de Tabla Maestra
    fig_m, ax_m = plt.subplots(figsize=(6, 1 + 0.45 * len(df)))
    d_err = df.sort_values(["ff", "vel0"], ascending=[True, False])
    _style(
        ax_m,
        ax_m.table(
            cellText=d_err[["ff", "vel0", "n", "Método", "err"]].astype(str).values,
            **karws,
        ),
        "Errores %",
    )
    figs.append(fig_m)

    # 4. Generación de Tablas Consolidadas (una por FF)
    for ff, d in df.groupby("ff"):
        fig, ax = plt.subplots(figsize=(6, 1 + 0.45 * len(d)))
        _style(
            ax,
            ax.table(
                cellText=d[["ff", "vel0", "n", "Método", "v"]].astype(str).values,
                **karws,
            ),
            f"IC — ff = {ff}",
        )
        figs.append(fig)

    return figs, df


if __name__ == "__main__":
    run_confidence_intervals()
