"""verificar_polars.py -- Compara el pipeline Polars (generador_figuras.py)
contra el pipeline Pandas original (erosion.py / erosion_49.py).

Uso::

    python3 verificar_polars.py          # ambos sets
    python3 verificar_polars.py 9_mov
    python3 verificar_polars.py 49_mov

Verifica que el procesamiento con Polars reproduzca exactamente (con
tolerancia numérica) los mismos resultados que la implementación Pandas:

  - ``df_dist``, ``df_cluster`` y ``df_proyectil`` (capa de datos con caché),
  - las tablas de rebote, de porcentaje de proyectil y de erosión cruda,
  - la tabla resumen en LaTeX del set 9 (contra ``erosion.tabla_resumen``).

NOTA: los ajustes de ley de potencia NO se comparan. A propósito el nuevo
script ajusta ``y = a * (x / 0.17) ** c`` sin el parámetro de traslación ``b``
(el set 49 ya lo forzaba a cero con ``z = x - b * 0``), por lo que los
exponentes difieren del ``erosion.py`` original y esa diferencia es intencional.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import polars as pl

import erosion
import generador_figuras as gf
from generador_figuras import DATASETS

RTOL = 1e-9
ATOL = 1e-9


def _col_numerica(series):
    """Convierte una columna (pandas o polars) a un array float comparable.

    Booleans -> 1.0/0.0, nulos -> NaN. Las columnas enteras y flotantes se
    convierten a float64 (todos los ids/valores de estos datos caben exactos).
    """
    if isinstance(series, pl.Series):
        series = series.to_pandas()
    s = series.reset_index(drop=True)
    if s.dtype.kind in "biuf":
        return np.asarray(s.astype(float), dtype=float)
    arr = s.astype(object).to_numpy()
    out = np.empty(len(arr), dtype=float)
    for i, v in enumerate(arr):
        if pd.isna(v):
            out[i] = np.nan
        elif isinstance(v, (bool, np.bool_)):
            out[i] = 1.0 if v else 0.0
        else:
            out[i] = float(v)
    return out


def _frames_iguales(pd_a, pd_b, key_cols, nombre, tol=(RTOL, ATOL)):
    """Ordena ambas tablas por ``key_cols`` y compara columna a columna."""
    if list(pd_a.columns) != list(pd_b.columns):
        raise AssertionError(
            f"{nombre}: columnas distintas\n  pandas={list(pd_a.columns)}\n  polars={list(pd_b.columns)}"
        )
    if len(pd_a) != len(pd_b):
        raise AssertionError(f"{nombre}: {len(pd_a)} filas (pandas) vs {len(pd_b)} (polars)")
    a = pd_a.sort_values(key_cols).reset_index(drop=True)
    b = pd_b.sort_values(key_cols).reset_index(drop=True)
    for c in a.columns:
        x = _col_numerica(a[c])
        y = _col_numerica(b[c])
        if not np.allclose(x, y, rtol=tol[0], atol=tol[1], equal_nan=True):
            idx = np.where(~np.isclose(x, y, rtol=tol[0], atol=tol[1], equal_nan=True))[0]
            i = idx[0]
            raise AssertionError(
                f"{nombre} | columna '{c}': discrepan {len(idx)} filas "
                f"(ej. pandas={x[i]!r} vs polars={y[i]!r})"
            )
    return True


# ---------------------------------------------------------------------------
# Referencia Pandas (réplica de procesar_datos, sin plotting)
# ---------------------------------------------------------------------------

def _homologar(df):
    df = df.copy()
    df["np"] = df["np"].replace(3, 1)
    df["angulo"] = df["angulo"].replace(90, 0)
    return df


def referencia_pandas(cfg):
    df = _homologar(pd.read_csv(cfg["csv"])) if cfg["homologar"] else pd.read_csv(cfg["csv"])
    datos_dist, datos_cluster, datos_proy = [], [], []
    for n, alfa in cfg["casos"]:
        base_grid, df_grid = erosion.grilla(n, df)
        for ff in cfg["ff_vals"]:
            vels = sorted(df_grid.loc[df_grid["ff"] == ff, "vel0"].abs().unique())
            for v in vels:
                mbase = (
                    (df_grid["ff"] == ff)
                    & (df_grid["vel0"] == -1 * v)
                    & (df_grid["np"] == n)
                    & (df_grid["angulo"] == alfa)
                )
                rots = [None]
                if cfg["rot_cols"]:
                    rr = df_grid.loc[mbase, cfg["rot_cols"]].drop_duplicates()
                    rots = [
                        dict(zip(cfg["rot_cols"], t)) for t in rr.itertuples(index=False, name=None)
                    ]
                for rot in rots:
                    mask = mbase
                    if rot is not None:
                        m = mbase.copy()
                        for k in cfg["rot_cols"]:
                            m &= df_grid[k] == rot[k]
                        mask = m
                    sim = df_grid[mask]
                    if sim.empty:
                        continue
                    i_max = sim["i"].max()
                    emask = (sim["i"] == i_max) & (sim["proyectil_erosion"] != False)
                    ec = sim[emask].groupby(["grid_x", "grid_y"])["id"].nunique().rename("yield")
                    res = base_grid.join(ec)
                    res.loc[res["existe"] & res["yield"].isna(), "yield"] = 0
                    valores = res.loc[res["existe"], "yield"].reset_index(drop=True)
                    dd = pd.DataFrame(
                        {"vel0": abs(v), "yield": valores, "ff": ff, "np": n, "angulo": alfa}
                    )
                    if rot is not None:
                        for k in cfg["rot_cols"]:
                            dd[k] = rot[k]
                    datos_dist.append(dd)
                    dc = erosion.extract_cluster_data(sim, n, alfa, ff, v)
                    if not dc.empty:
                        if rot is not None:
                            for k in cfg["rot_cols"]:
                                dc[k] = rot[k]
                        datos_cluster.append(dc)
                    datos_proy.append(sim)
    return pd.concat(datos_dist), pd.concat(datos_cluster), pd.concat(datos_proy)


def _combo_key(cfg, extra=()):
    return ["np", "angulo", "ff", "vel0"] + cfg["rot_cols"] + list(extra)


# ---------------------------------------------------------------------------
# Tablas (réplicas Pandas con soporte de rotación)
# ---------------------------------------------------------------------------

def tabla_rebote_pd(cfg, df):
    i_max = df["i"].max()
    proj = df[(df["type"] == 2) & (df["i"] == i_max)].copy()
    proj["vel0"] = proj["vel0"].abs()
    gcols = ["np", "angulo", "ff", "vel0", "posx", "posy"] + cfg["rot_cols"]
    prob = proj.groupby(gcols)["proyectil_erosion"].mean().reset_index(name="p_rebote")
    resumen = (
        prob.groupby(["np", "angulo", "ff", "vel0"], as_index=False)["p_rebote"]
        .agg(probabilidad="mean", se="sem")
        .sort_values(["np", "angulo", "ff", "vel0"])
        .reset_index(drop=True)
    )
    return resumen


def tabla_porcentaje_pd(cfg, df):
    i_max = df["i"].max()
    proj = df[
        (df["type"] == 2) & (df["i"] == i_max) & df["proyectil_erosion"].fillna(False)
    ].copy()
    eros = df[(df["type"] == 1) & (df["i"] == i_max)].copy()
    proj["vel0"] = proj["vel0"].abs()
    eros["vel0"] = eros["vel0"].abs()
    gcols = ["np", "angulo", "ff", "vel0", "posx", "posy"] + cfg["rot_cols"]
    tp = proj.groupby(gcols).size().rename("count_proj").reset_index()
    te = eros.groupby(gcols).size().rename("count_eros").reset_index()
    m = tp.merge(te, on=gcols, how="outer").fillna(0)
    m["total_eventos"] = m["count_proj"] + m["count_eros"]
    m["porcentaje_erosion"] = np.where(
        m["total_eventos"] > 0, 100 * m["count_proj"] / m["total_eventos"], 0.0
    )
    resumen = (
        m.groupby(["np", "angulo", "ff", "vel0"], as_index=False)["porcentaje_erosion"]
        .agg(mean="mean", se="sem")
        .sort_values(["np", "angulo", "ff", "vel0"])
        .reset_index(drop=True)
    )
    return resumen


def tabla_erosion_cruda_pd(cfg, df):
    i_max = df["i"].max()
    proj = df[
        (df["type"] == 2) & (df["i"] == i_max) & df["proyectil_erosion"].fillna(False)
    ].copy()
    eros = df[(df["type"] == 1) & (df["i"] == i_max)].copy()
    proj["vel0"] = proj["vel0"].abs()
    eros["vel0"] = eros["vel0"].abs()
    gcols = ["np", "angulo", "ff", "vel0", "posx", "posy"] + cfg["rot_cols"]
    tp = proj.groupby(gcols).size().rename("count_proj").reset_index()
    te = eros.groupby(gcols).size().rename("count_eros").reset_index()
    m = tp.merge(te, on=gcols, how="outer").fillna(0)
    m["total_eventos"] = m["count_proj"] + m["count_eros"]
    m["porcentaje_erosion"] = np.where(
        m["total_eventos"] > 0, 100 * m["count_proj"] / m["total_eventos"], 0.0
    )
    return m.sort_values(gcols).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Verificación por set
# ---------------------------------------------------------------------------

def verificar_set(setname):
    cfg = DATASETS[setname]
    print(f"=== {setname} ===")

    df_pd = _homologar(pd.read_csv(cfg["csv"])) if cfg["homologar"] else pd.read_csv(cfg["csv"])

    print("  [datos] pipeline pandas (referencia)...")
    dist_pd, cluster_pd, proy_pd = referencia_pandas(cfg)

    print("  [datos] pipeline polars (procesar_datos, forzar)...")
    dist_pl, cluster_pl, proy_pl = gf.procesar_datos(cfg, forzar=True)
    dist_pl = dist_pl.to_pandas()
    cluster_pl = cluster_pl.to_pandas()
    proy_pl = proy_pl.to_pandas()

    # --- df_dist: por combinación, los yields ordenados deben coincidir ---
    print("  [df_dist] comparando...")
    key = _combo_key(cfg)
    for cols, gr_pd in dist_pd.groupby(key, sort=False):
        gr_pl = dist_pl.set_index(key).sort_index().loc[cols].reset_index()
        a = np.sort(np.asarray(gr_pd["yield"], dtype=float))
        b = np.sort(np.asarray(gr_pl["yield"], dtype=float))
        assert len(a) == len(b), f"yields: {len(a)} vs {len(b)} en {cols}"
        assert np.allclose(a, b, atol=0), f"yields difieren en {cols}"

    # --- df_cluster ---
    print("  [df_cluster] comparando...")
    ckey = ["posx", "posy", "cluster", "np", "angulo", "ff", "v"] + cfg["rot_cols"]
    _frames_iguales(cluster_pd, cluster_pl, ckey, "df_cluster")

    # --- df_proyectil: mismo multiset de filas por combinación ---
    print("  [df_proyectil] comparando...")
    # (i, id) puede repetirse entre distintas realizaciones de rotación, por lo
    # que la clave de ordenación incluye las columnas de rotación.
    pkey = ["i", "id", "grid_x", "grid_y", "posx", "posy", "np", "angulo", "ff", "vel0"] + cfg["rot_cols"]
    proy_pd_sorted = proy_pd.sort_values(pkey).reset_index(drop=True)
    proy_pl_sorted = proy_pl.sort_values(pkey).reset_index(drop=True)
    _frames_iguales(proy_pd_sorted, proy_pl_sorted, pkey, "df_proyectil")

    # --- Tablas (réplica pandas con rotación vs polars) ---
    print("  [tablas] rebote / %proyectil / erosión cruda...")
    gkey = ["np", "angulo", "ff", "vel0"]

    reb_pd = tabla_rebote_pd(cfg, df_pd)
    reb_pl = gf.tabla_rebote_proyectil(cfg, gf.leer_crudo(cfg))[1].to_pandas()
    _frames_iguales(reb_pd.sort_values(gkey).reset_index(drop=True),
                    reb_pl.sort_values(gkey).reset_index(drop=True), gkey, "rebote")

    pct_pd = tabla_porcentaje_pd(cfg, df_pd)
    pct_pl = gf.tabla_porcentaje_proyectil(cfg, gf.leer_crudo(cfg))[1].to_pandas()
    _frames_iguales(pct_pd.sort_values(gkey).reset_index(drop=True),
                    pct_pl.sort_values(gkey).reset_index(drop=True), gkey, "%proyectil")

    cruda_pd = tabla_erosion_cruda_pd(cfg, df_pd)
    cruda_pl = gf.tabla_erosion_cruda(cfg, gf.leer_crudo(cfg)).to_pandas()
    ckey_cruda = ["np", "angulo", "ff", "vel0", "posx", "posy"] + cfg["rot_cols"]
    _frames_iguales(cruda_pd, cruda_pl, ckey_cruda, "erosión cruda")

    # --- Set 9: la tabla resumen LaTeX debe coincidir con la original ---
    if setname == "9_mov":
        print("  [tabla_resumen] comparando LaTeX con erosion.py...")
        dist_pl2, _, proy_pl2 = gf.procesar_datos(cfg)
        latex_orig = erosion.tabla_resumen(dist_pd, proy_pd, df_pd, cfg["casos"])
        latex_nuevo = gf.tabla_resumen(cfg, dist_pl2, proy_pl2, gf.leer_crudo(cfg))
        if latex_orig != latex_nuevo:
            for l1, l2 in zip(latex_orig.splitlines(), latex_nuevo.splitlines()):
                if l1 != l2:
                    print(f"    orig: {l1!r}\n    nuevo:{l2!r}")
            raise AssertionError("tabla_resumen LaTeX difiere del original")
        print("    LaTeX idéntico (%d líneas)" % len(latex_nuevo.splitlines()))

    print(f"=== {setname}: OK ===")


def main():
    sets = sys.argv[1:] if len(sys.argv) > 1 else list(DATASETS)
    for s in sets:
        if s not in DATASETS:
            sys.exit(f"set desconocido: {s} (opciones: {list(DATASETS)})")
    for s in sets:
        verificar_set(s)
    print("\nTODAS LAS VERIFICACIONES PASARON")


if __name__ == "__main__":
    main()
