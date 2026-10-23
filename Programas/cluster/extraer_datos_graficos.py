"""extraer_datos_graficos.py -- Exporta los datos (media + SEM) de los graficos.

Lee los CSV ya procesados en ``datos/procesados/<set>/`` y escribe, para cada
set, un CSV en formato largo con una fila por punto dibujado:

    nombre_grafico, valor, dispersion, velocidad, np, angulo, ff, repeticiones, k

``valor`` es la media del punto y ``dispersion`` su error estandar (SEM,
``std(ddof=1)/sqrt(n)``). Para los graficos de PDF de clusters, ``k`` es el
tamano de cluster y la dispersion es el SEM de ``P_j(k)`` entre los impactos.

Uso::

    python3 extraer_datos_graficos.py

El script usa solo Polars (no importa ``generador_figuras.py`` para no depender
de seaborn/scipy/pandas, que no estan instalados en este entorno).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

BASE = Path(__file__).resolve().parent

ROT_COLS = ["rot_angulo", "rot_eje_x", "rot_eje_y", "rot_eje_z"]

DATASETS = {
    "9_mov": {
        "cache": BASE / "datos" / "procesados" / "9_mov",
        "out": BASE / "datos" / "9_mov" / "datos_graficos.csv",
        "casos": [(1, 0), (3, 90), (3, 0)],
        "ff_vals": [0.15, 0.35],
        "rot_cols": [],
        "homologar": False,
        "reps": 9,
    },
    "49_mov": {
        "cache": BASE / "datos" / "procesados" / "49_mov",
        "out": BASE / "datos" / "49_mov" / "datos_graficos.csv",
        "casos": [(1, 0)],
        "ff_vals": [0.15],
        "rot_cols": ROT_COLS,
        "homologar": True,
        "reps": 245,
    },
}

COLS = [
    "nombre_grafico",
    "valor",
    "dispersion",
    "velocidad",
    "np",
    "angulo",
    "ff",
    "repeticiones",
    "k",
]

G_Y = "plot_power_law_todos.png"
G_DMAX_P = "plot_max_profundidad_proyectil_vs_VELOCIDAD_vs_NP_vs_ANGULO_vs_FF.png"
G_DMEAN_P = "plot_mean_profundidad_proyectil_vs_VELOCIDAD_vs_NP_vs_ANGULO_vs_FF.png"
G_DMAX_L = "plot_max_profundidad_lecho_vs_VELOCIDAD_vs_NP_vs_ANGULO_vs_FF.png"
G_DMEAN_L = "plot_mean_profundidad_lecho_vs_VELOCIDAD_vs_NP_vs_ANGULO_vs_FF.png"
G_REBOTE = "heatmap_rebote.png"
G_EROSION = "heatmap_erosion.png"
G_PDF = "plot_PDF_por_vel_cluster.png"
G_PDF_COMP = "plot_PDF_comparativo.png"


def _load(cfg):
    """Carga los CSV procesados de un set (y homologa np/angulo si corresponde)."""
    dfs = {}
    for name in ("df_dist", "df_cluster", "df_proyectil"):
        df = pl.read_csv(cfg["cache"] / f"{name}.csv")
        if cfg["homologar"]:
            df = df.with_columns(
                pl.when(pl.col("np") == 3)
                .then(pl.lit(1, dtype=pl.Int64))
                .otherwise(pl.col("np"))
                .alias("np"),
                pl.when(pl.col("angulo") == 90)
                .then(pl.lit(0, dtype=pl.Int64))
                .otherwise(pl.col("angulo"))
                .alias("angulo"),
            )
        dfs[name] = df
    return dfs


def _mean_sem(df, group_cols, value_col):
    """Media y SEM de ``value_col`` por ``group_cols``."""
    return df.group_by(group_cols).agg(
        pl.col(value_col).mean().alias("valor"),
        (pl.col(value_col).std(ddof=1) / pl.col(value_col).count().sqrt()).alias(
            "dispersion"
        ),
    )


def _with_meta(df, cfg, graph_name):
    """Agrega las columnas comunes de metadata y ordena el esquema."""
    return df.with_columns(
        pl.lit(graph_name).alias("nombre_grafico"),
        pl.lit(cfg["reps"], dtype=pl.Int64).alias("repeticiones"),
        pl.lit(None, dtype=pl.Int64).alias("k"),
    ).select(COLS)


def _erosion_y(cfg, df_dist, graph_name):
    """Erosion media (Y) por velocidad, caso y fraccion de llenado."""
    df = df_dist.with_columns(pl.col("vel0").abs().alias("velocidad"))
    out = _mean_sem(df, ["np", "angulo", "ff", "velocidad"], "yield")
    return _with_meta(out, cfg, graph_name)


def _depth_metric(cfg, df_proyectil, n, alfa, tipo, agg_name, graph_name):
    """Profundidad (max/media) del proyectil o del lecho por velocidad y ff."""
    df = df_proyectil.filter((pl.col("np") == n) & (pl.col("angulo") == alfa))
    if tipo == "proyectil":
        df = df.filter(
            pl.col("proyectil_erosion").eq(False)
            & (pl.col("i") == 80)
            & (pl.col("z") < 100e-6)
        )
    else:
        df = df.filter(
            (pl.col("type") == 1) & (pl.col("i") == 0) & (pl.col("z") < 100e-6)
        )
    if df.is_empty():
        return None

    df = df.with_columns((100 - pl.col("z") * 1e6).alias("depth"))
    expr = pl.col("depth").max() if agg_name == "max" else pl.col("depth").mean()
    df = df.group_by(["vel0", "posx", "posy", "ff"] + cfg["rot_cols"]).agg(
        expr.alias("depth")
    )
    df = df.with_columns(pl.col("vel0").abs().alias("velocidad"))
    out = _mean_sem(df, ["velocidad", "ff"], "depth")
    out = out.with_columns(
        pl.lit(n, dtype=pl.Int64).alias("np"),
        pl.lit(alfa, dtype=pl.Int64).alias("angulo"),
    )
    return _with_meta(out, cfg, graph_name)


def _rebote(cfg, df_proyectil, graph_name):
    """Proporcion de masa (rebote) por velocidad, caso y ff."""
    i_max = df_proyectil["i"].max()
    proj = df_proyectil.filter(
        (pl.col("type") == 2) & (pl.col("i") == i_max)
    ).with_columns(pl.col("vel0").abs().alias("velocidad"))
    if proj.is_empty():
        return None

    gcols = ["np", "angulo", "ff", "velocidad", "posx", "posy"] + cfg["rot_cols"]
    prob = proj.group_by(gcols).agg(
        pl.col("proyectil_erosion").cast(pl.Float64).mean().alias("p_rebote")
    )
    out = _mean_sem(prob, ["np", "angulo", "ff", "velocidad"], "p_rebote")
    out = out.with_columns(
        (pl.col("valor") * 100).alias("valor"),
        (pl.col("dispersion") * 100).alias("dispersion"),
    )
    return _with_meta(out, cfg, graph_name)


def _erosion_proyectil(cfg, df_proyectil, graph_name):
    """Proporcion de erosion del proyectil (%) por velocidad, caso y ff."""
    i_max = df_proyectil["i"].max()
    proj = df_proyectil.filter(
        (pl.col("type") == 2)
        & (pl.col("i") == i_max)
        & pl.col("proyectil_erosion")
    ).with_columns(pl.col("vel0").abs().alias("velocidad"))
    eros = df_proyectil.filter(
        (pl.col("type") == 1) & (pl.col("i") == i_max)
    ).with_columns(pl.col("vel0").abs().alias("velocidad"))

    gcols = ["np", "angulo", "ff", "velocidad", "posx", "posy"] + cfg["rot_cols"]
    total_proj = proj.group_by(gcols).agg(pl.len().alias("count_proj"))
    total_eros = eros.group_by(gcols).agg(pl.len().alias("count_eros"))
    df = total_proj.join(total_eros, on=gcols, how="full", coalesce=True).with_columns(
        pl.col("count_proj").fill_null(0),
        pl.col("count_eros").fill_null(0),
    )
    df = df.with_columns(
        (pl.col("count_proj") + pl.col("count_eros")).alias("total_eventos")
    )
    df = df.with_columns(
        pl.when(pl.col("total_eventos") > 0)
        .then(100 * pl.col("count_proj") / pl.col("total_eventos"))
        .otherwise(0.0)
        .alias("porcentaje_erosion")
    )
    out = _mean_sem(df, ["np", "angulo", "ff", "velocidad"], "porcentaje_erosion")
    return _with_meta(out, cfg, graph_name)


def _pdf(cfg, df_cluster, n, alfa, v, ff, graph_name):
    """P(k) y su SEM para un panel (caso, velocidad, ff) del grafico de PDF."""
    impact_cols = ["posx", "posy"] + cfg["rot_cols"]
    sub = df_cluster.filter(
        (pl.col("np") == n)
        & (pl.col("angulo") == alfa)
        & (pl.col("v") == v)
        & (pl.col("ff") == ff)
    )
    if sub.is_empty():
        return None

    f_counts = sub.group_by(impact_cols + ["count"]).agg(pl.len().alias("F"))
    totales = f_counts.group_by(impact_cols).agg(pl.col("F").sum().alias("M_t"))
    f_counts = f_counts.join(totales, on=impact_cols, how="left").filter(
        pl.col("M_t") > 0
    )
    if f_counts.is_empty():
        return None

    impacts = f_counts.select(impact_cols).unique()
    counts = f_counts.select("count").unique()
    grid = impacts.join(counts, how="cross")
    grid = grid.join(
        f_counts.select(impact_cols + ["count", "F"]),
        on=impact_cols + ["count"],
        how="left",
    ).with_columns(pl.col("F").fill_null(0))
    grid = grid.join(totales, on=impact_cols, how="left")
    grid = grid.with_columns((pl.col("F") / pl.col("M_t")).alias("P_j"))

    ensemble = grid.group_by("count").agg(pl.col("F").sum().alias("N_k"))
    total = ensemble["N_k"].sum()
    if not total:
        return None
    ensemble = ensemble.with_columns(
        (pl.col("N_k") / pl.lit(float(total))).alias("valor")
    )
    sem = grid.group_by("count").agg(
        (pl.col("P_j").std(ddof=1) / pl.col("P_j").count().sqrt()).alias("dispersion")
    )
    out = ensemble.join(sem, on="count", how="left").filter(pl.col("valor") > 0)
    out = out.with_columns(
        pl.lit(graph_name).alias("nombre_grafico"),
        pl.lit(v, dtype=pl.Int64).alias("velocidad"),
        pl.lit(n, dtype=pl.Int64).alias("np"),
        pl.lit(alfa, dtype=pl.Int64).alias("angulo"),
        pl.lit(ff, dtype=pl.Float64).alias("ff"),
        pl.lit(cfg["reps"], dtype=pl.Int64).alias("repeticiones"),
        pl.col("count").alias("k"),
    ).select(COLS)
    return out


def _pdf_set(cfg, df_cluster, graph_name):
    """Todos los paneles del grafico de PDF de un set."""
    frames = []
    for n, alfa in cfg["casos"]:
        for ff in cfg["ff_vals"]:
            vels = sorted(
                df_cluster.filter(
                    (pl.col("np") == n)
                    & (pl.col("angulo") == alfa)
                    & (pl.col("ff") == ff)
                )["v"]
                .unique()
                .to_list()
            )
            for v in vels:
                out = _pdf(cfg, df_cluster, n, alfa, v, ff, graph_name)
                if out is not None and not out.is_empty():
                    frames.append(out)
    return _concat(frames)


def _concat(frames):
    """Concatena verticalmente los DataFrames no vacios."""
    frames = [f for f in frames if f is not None and not f.is_empty()]
    if not frames:
        return None
    return pl.concat(frames, how="vertical")


def _scalar_frames(cfg, dfs):
    """Construye las filas de todos los graficos escalares (no PDF)."""
    frames = [_erosion_y(cfg, dfs["df_dist"], G_Y)]
    for n, alfa in cfg["casos"]:
        for tipo, agg_name, graph_name in [
            ("proyectil", "max", G_DMAX_P),
            ("proyectil", "mean", G_DMEAN_P),
            ("lecho", "max", G_DMAX_L),
            ("lecho", "mean", G_DMEAN_L),
        ]:
            out = _depth_metric(
                cfg, dfs["df_proyectil"], n, alfa, tipo, agg_name, graph_name
            )
            frames.append(out)
    frames.append(_rebote(cfg, dfs["df_proyectil"], G_REBOTE))
    frames.append(_erosion_proyectil(cfg, dfs["df_proyectil"], G_EROSION))
    return frames


def _write(cfg, out):
    """Valida el DataFrame final y lo escribe a disco."""
    out = out.sort(
        ["nombre_grafico", "velocidad", "np", "angulo", "ff", "k"],
        nulls_last=True,
    )
    cfg["out"].parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(cfg["out"])
    print(f"{cfg['out']}: {out.height} filas")
    print(out.group_by("nombre_grafico").len().sort("nombre_grafico"))
    return out


def main():
    """Genera los dos CSV (uno por set)."""
    data = {name: _load(cfg) for name, cfg in DATASETS.items()}

    cfg_9 = DATASETS["9_mov"]
    frames_9 = _scalar_frames(cfg_9, data["9_mov"])
    frames_9.append(_pdf_set(cfg_9, data["9_mov"]["df_cluster"], G_PDF))
    out_9 = _concat(frames_9)
    if out_9 is None:
        raise RuntimeError("No se generaron datos para 9_mov")
    _write(cfg_9, out_9)

    cfg_49 = DATASETS["49_mov"]
    frames_49 = _scalar_frames(cfg_49, data["49_mov"])
    frames_49.append(_pdf_set(cfg_49, data["49_mov"]["df_cluster"], G_PDF))

    comp = []
    for n, alfa in cfg_9["casos"]:
        comp.append(
            _pdf(cfg_9, data["9_mov"]["df_cluster"], n, alfa, 50, 0.15, G_PDF_COMP)
        )
    for n, alfa in cfg_49["casos"]:
        comp.append(
            _pdf(cfg_49, data["49_mov"]["df_cluster"], n, alfa, 50, 0.15, G_PDF_COMP)
        )
    frames_49.append(_concat(comp))

    out_49 = _concat(frames_49)
    if out_49 is None:
        raise RuntimeError("No se generaron datos para 49_mov")
    _write(cfg_49, out_49)

    if out_9["valor"].null_count() or out_49["valor"].null_count():
        raise RuntimeError("Hay valores nulos en la columna valor")
    print("ok")


if __name__ == "__main__":
    main()
