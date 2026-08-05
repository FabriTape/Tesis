"""Verifica cuantitativamente si las distribuciones P(k) sin relleno suman 1.

Replica exactamente la lógica NUEVA de _plot_distribution_panel (L489-501):
  1. N_k = suma de F(n)_j agrupando por (count, φ)
  2. P(k) = N_k / Σ_φ(N_k)

Testea DOS cosas:
  1. ¿Cada impacto individual suma 1? (por construcción de Prob_local)
  2. ¿La distribución del ensamble (suma global) suma 1? (la que se grafica)

Nota: este archivo es autocontenido (versión archivada en legacy/). Las
funciones grilla, extract_cluster_data y _prepare_cluster_data_sin_relleno
fueron copiadas literalmente de erosion.py; la extracción de clusters se
hizo más liviana (sin generar los heatmaps).
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---------- paths (idénticos a erosion.py) ----------
CSV_PATH = Path("datos/resultados_9_mov/merge_resultados.csv")
PHI_VALS = [0.15, 0.35]
CASOS = [(1, 0), (3, 90), (3, 0)]


def grilla(np, df):
    """Construye una grilla (cuadrícula) de celdas [grid_x, grid_y] para el
    caso con N_p proyectiles.

    Copiada de erosion.grilla.

    Args:
        np: número de proyectiles del caso a considerar.
        df: DataFrame completo de la simulación.

    Returns:
        base_grid: DataFrame indexado por (grid_x, grid_y) con la columna
            booleana "existe".
        data: df filtrado al caso "np", con las columnas grid_x/grid_y.
    """
    data = df[df["np"] == np].copy()
    step = data[data["posx"] > 0]["posx"].min()
    data["grid_x"] = (data["posx"] / step).round().astype(int)
    data["grid_y"] = (data["posy"] / step).round().astype(int)

    x_min = int(data["grid_x"].min())
    y_min = int(data["grid_y"].min())
    x_max = int(data["grid_x"].max())
    y_max = int(data["grid_y"].max())

    reales = (
        data[["grid_x", "grid_y"]]
        .drop_duplicates()
        .assign(existe=True)
        .set_index(["grid_x", "grid_y"])
    )
    x_r, y_r = range(x_min, x_max + 1), range(y_min, y_max + 1)
    base_grid = (
        pd.MultiIndex.from_product([x_r, y_r], names=["grid_x", "grid_y"])
        .to_frame(index=False)
        .set_index(["grid_x", "grid_y"])
    )
    base_grid = base_grid.join(reales).fillna(False).infer_objects(copy=False)
    return base_grid, data


def extract_cluster_data(df, np_val, angle, ff, v):
    """Agrega los clusters eyectados de una simulación en un DataFrame
    resumen (un cluster por fila).

    Copiada de erosion.extract_cluster_data.

    Args:
        df: datos de una simulación (ff, v, np, angulo fijos).
        np_val, angle, ff, v: etiquetas del caso, agregadas al resultado.

    Returns:
        DataFrame con columnas posx, posy, cluster, count, x_cm, y_cm, z_cm,
        v_cm_*, km, rg, angulo_vel, np, angulo, ff, v.
    """
    i_max = df["i"].max()
    mask = (df["i"] == i_max) & ((df["type"] == 1) | df["proyectil_erosion"])
    df_filt = df[mask]
    if df_filt.empty:
        return pd.DataFrame()

    df_cluster = (
        df_filt.groupby(["posx", "posy", "cluster"])
        .agg(
            count=("cluster", "size"),
            x_cm=("x", "mean"),
            y_cm=("y", "mean"),
            z_cm=("z", "mean"),
            v_cm_x=("vx", "mean"),
            v_cm_y=("vy", "mean"),
            v_cm_z=("vz", "mean"),
            var_x=("x", lambda x: x.var(ddof=0)),
            var_y=("y", lambda x: x.var(ddof=0)),
            var_z=("z", lambda x: x.var(ddof=0)),
        )
        .reset_index()
    )

    # Momento cinético del centro de masa y radio de giro del cluster
    v_cm = df_cluster[["v_cm_x", "v_cm_y", "v_cm_z"]].values
    km = 0.5 * np.sum(v_cm**2, axis=1)
    rg = np.sqrt(df_cluster["var_x"] + df_cluster["var_y"] + df_cluster["var_z"])
    v_mag = np.sqrt(2 * km)
    angulo_vel = np.where(
        v_mag > 0,
        np.degrees(np.arccos(np.clip(df_cluster["v_cm_z"] / v_mag, -1.0, 1.0))),
        0.0,
    )

    return df_cluster.assign(
        km=km, rg=rg, angulo_vel=angulo_vel, np=np_val, angulo=angle, ff=ff, v=v
    ).drop(columns=["var_x", "var_y", "var_z"])


def extract_cluster_data_de_casos(df, casos):
    """Extrae los clusters de todos los casos sin generar heatmaps.

    Versión liviana de erosion.extract_main_data: recorre cada (np, ángulo)
    y cada (ff, v) presente y agrega los clusters eyectados.

    Args:
        df: DataFrame completo de la simulación.
        casos: lista de tuplas (n_proyectiles, ángulo).

    Returns:
        df_cluster: DataFrame con los clusters de todos los casos.
    """
    datos_cluster = []
    for n, alfa in casos:
        _, df_grid = grilla(n, df)
        for ff in PHI_VALS:
            vels = sorted(df_grid.loc[df_grid["ff"] == ff, "vel0"].abs().unique())
            for v in vels:
                sim = df_grid[
                    (df_grid["ff"] == ff)
                    & (df_grid["vel0"] == -1 * v)
                    & (df_grid["np"] == n)
                    & (df_grid["angulo"] == alfa)
                ]
                if not sim.empty:
                    datos_cluster.append(extract_cluster_data(sim, n, alfa, ff, v))
    return pd.concat(datos_cluster)


def _prepare_cluster_data_sin_relleno(casos, df_dist):
    """Agrupa los clusters por impacto sin rellenar tamaños no observados.

    Copiada de erosion._prepare_cluster_data_sin_relleno.

    F(n)_j: número de clusters de tamaño n eyectados en el impacto j.
    M_t: total de clusters eyectados en ese impacto.

    Args:
        casos: lista de tuplas (n_proyectiles, ángulo).
        df_dist: DataFrame con un cluster por fila.

    Returns:
        dfs: lista (una por caso) con columnas posx, posy, np, angulo, ff,
            count, v, F(n)_j y M_t.
    """
    dfs = []
    for n, alfa in casos:
        subset = df_dist[(df_dist["angulo"] == alfa) & (df_dist["np"] == n)].copy()
        df_plot = subset.rename(columns={"ff": r"$\phi$"})

        df_por_impacto = (
            df_plot.groupby(["posx", "posy", "np", "angulo", r"$\phi$", "count", "v"])
            .size()
            .reset_index(name="F(n)_j")
        )
        df_por_impacto["M_t"] = df_por_impacto.groupby(
            ["posx", "posy", r"$\phi$", "v"]
        )["F(n)_j"].transform("sum")
        dfs.append(df_por_impacto)
    return dfs


def main() -> None:
    df = pd.read_csv(CSV_PATH)

    # Sólo necesitamos los clusters (no los heatmaps ni los proyectiles)
    df_cluster = extract_cluster_data_de_casos(df, CASOS)

    dfs = _prepare_cluster_data_sin_relleno(CASOS, df_cluster)

    velocities = sorted(set(v for d in dfs for v in d["v"].unique()))

    tolerancia = 1e-10
    todos_ok = True

    for caso_idx, (n, alfa) in enumerate(CASOS):
        print(f"\n{'='*70}")
        print(f"Caso Np={n}, α={alfa}")
        print(f"{'='*70}")

        for v in velocities:
            df_v = dfs[caso_idx][dfs[caso_idx]["v"] == v].copy()
            if df_v.empty:
                continue

            # Probabilidad local de cada tamaño en cada impacto
            df_v["Prob_local"] = np.where(
                df_v["M_t"] > 0, df_v["F(n)_j"] / df_v["M_t"], 0.0
            )

            # ---- Test 1: ¿Cada impacto individual suma 1? ----
            # Por construcción, Prob_local = F(n)_j / M_t suma 1 sobre los
            # tamaños n de un mismo impacto.
            imp_sums = (
                df_v.groupby(["posx", "posy", r"$\phi$"])["Prob_local"]
                .sum()
                .reset_index()
            )
            max_err_imp = (imp_sums["Prob_local"] - 1.0).abs().max()
            ok_imp = max_err_imp < tolerancia

            if not ok_imp:
                todos_ok = False

            tag = "OK" if ok_imp else "FALLA"
            print(f"  v={v:.0f} m/s | Test 1 (impacto individual): "
                  f"max|suma-1| = {max_err_imp:.2e}  [{tag}]")

            # ---- Test 2: ¿La distribución del ensamble suma 1? ----
            # P(k) global del ensamble: N_k = Σ_j F(n)_j por tamaño,
            # normalizado por el total Σ_k N_k. Este es el que se grafica.
            phi_vals = sorted(df_v[r"$\phi$"].unique())

            for phi_val in phi_vals:
                sub = df_v[df_v[r"$\phi$"] == phi_val]

                stats = (
                    sub.groupby("count")["F(n)_j"]
                    .sum()
                    .reset_index(name="N_k")
                )
                phi_sum = stats["N_k"].sum()
                stats["P_k"] = np.where(phi_sum > 0, stats["N_k"] / phi_sum, 0.0)
                sum_pk = stats["P_k"].sum()
                err = abs(sum_pk - 1.0)
                ok_sum = err < tolerancia

                if not ok_sum:
                    todos_ok = False

                tag2 = "OK" if ok_sum else "FALLA"
                n_impactos = sub["posx"].nunique()
                counts_presentes = stats["count"].tolist()

                print(
                    f"    φ={phi_val} | Test 2 (ensamble): "
                    f"Σ P(k) = {sum_pk:.6f}, "
                    f"n_impactos={n_impactos}, "
                    f"counts={counts_presentes}  [{tag2}]"
                )

    print(f"\n{'='*70}")
    if todos_ok:
        print("RESULTADO: TODAS las distribuciones suman 1 ✓")
    else:
        print("RESULTADO: HAY distribuciones que NO suman 1 ✗")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
