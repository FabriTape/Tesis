"""generador_figuras.py -- Análisis de erosión unificado (9 y 49 movimientos).

Reemplaza a ``erosion.py`` (set de 9 repeticiones) y ``erosion_49.py`` (set de
49 repeticiones) en un único script. Cambios respecto de los originales:

- Todo el procesamiento de datos usa **Polars** (antes Pandas). Pandas queda
  solo en la capa de presentación (pivots para LaTeX/heatmaps y ``to_latex``),
  para preservar el formato exacto de las tablas.
- Los resultados procesados (``df_dist``, ``df_cluster``, ``df_proyectil``) se
  guardan como CSV en ``datos/procesados/<set>/``. Si ya existen, se
  reutilizan en lugar de volver a filtrar los CSV crudos.
- El modelo de ley de potencia se ajusta **sin** el parámetro de traslación b:
  ``y = a * (x / 0.17) ** c``.
- Las figuras se guardan en carpetas separadas por set
  (``figuras/9_mov/`` y ``figuras/49_mov/``), por lo que los nombres de
  archivo no se pisan entre sets.
- Los gráficos del set de 9 repeticiones se dibujan con **seaborn** (igual que
  ``erosion.py``); los del set de 49 repeticiones se dibujan con matplotlib puro
  (igual que ``erosion_49.py``). El flag ``sns`` por set controla el estilo.

Uso::

    python generador_figuras.py           # procesa ambos sets y genera todo
    python generador_figuras.py 9_mov     # solo el set de 9
    python generador_figuras.py 49_mov    # solo el set de 49
    python generador_figuras.py 9_mov 49_mov --forzar   # ignora el caché

La equivalencia con la versión original (pandas) se verifica con
``verificar_polars.py``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
from scipy.stats import t
from sklearn.metrics import mean_squared_error
from uncertainties import unumpy as unp

warnings.filterwarnings("ignore", category=FutureWarning, module="matplotlib")
warnings.filterwarnings(
    "ignore", message="Covariance of the parameters could not be estimated"
)

PHI = r"$\phi$"
ROT_COLS = ["rot_angulo", "rot_eje_x", "rot_eje_y", "rot_eje_z"]

BASE = Path(__file__).resolve().parent

# Configuración por set de datos. Cada entrada define los casos (np, ángulo),
# las fracciones de llenado, las columnas de rotación, los estilos de fuente,
# el comportamiento de cada set (relleno de ceros, supervivencia, comparativo,
# ajustes de profundidad, etc.) y las carpetas de salida.
DATASETS = {
    "9_mov": {
        "csv": BASE / "datos/resultados_9_mov/merge_resultados.csv",
        "casos": [(1, 0), (3, 90), (3, 0)],
        "ff_vals": [0.15, 0.35],
        "rot_cols": [],
        "marker_styles": {(1, 0): "o", (3, 90): "^", (3, 0): "s"},
        "mk_size": 15,
        "handle_mk": 20,
        "hollow_lw": 3,
        "col_labels": ["(a)", "(b)", "(c)"],
        "grid_alpha": 0.5,
        "annot_position": "bottom-right",
        "fit_lw": 3.8,
        "fit_alpha": 0.9,
        "cluster_xlim": (0.95, 12),
        "cluster_ylim": (1e-3, 1.1),
        "cluster_trend_max": 10 ** 1.1,
        "homologar": False,
        "sns": True,
        "relleno": False,
        "fit_astrom": False,
        "fit_depth": True,
        "survival": False,
        "comparativo": False,
        "tabla_resumen_tipo": "multi",
        "depth_labels": [
            ("d_max", "proyectil", "max"),
            ("d_mean", "proyectil", "mean"),
            ("D_max", "lecho", "max"),
            ("D_mean", "lecho", "mean"),
        ],
        "depth_rows": [
            (r"$d_{\max}$ (proyectil)", "d_max", "proyectil", "max"),
            (r"$d_{mean}$ (proyectil)", "d_mean", "proyectil", "mean"),
            (r"$d_{\max}$ (lecho)", "D_max", "lecho", "max"),
            (r"$d_{mean}$ (lecho)", "D_mean", "lecho", "mean"),
        ],
        "fonts": {
            "axes_label": 20,
            "tick_label": 13,
            "legend": 23,
            "title": 21,
            "subplot_label": 20,
            "annotation": 20,
            "heatmap_cell": 10,
            "heatmap_annot": 13,
            "comp_legend": 12,
        },
        "fig_dir": BASE / "figuras" / "9_mov",
        "cache_dir": BASE / "datos" / "procesados" / "9_mov",
        "out_dir": BASE / "datos" / "9_mov",
    },
    "49_mov": {
        "csv": BASE / "datos/resultados_49_mov/merge_resultados.csv",
        "casos": [(1, 0)],
        "ff_vals": [0.15],
        "rot_cols": ROT_COLS,
        "marker_styles": {(1, 0): "o"},
        "mk_size": 13,
        "handle_mk": 13,
        "hollow_lw": 1.5,
        "col_labels": None,
        "grid_alpha": 0.7,
        "annot_position": "top-left",
        "fit_lw": 1.8,
        "fit_alpha": 0.7,
        "cluster_xlim": (0.95, 41),
        "cluster_ylim": (2.1e-4, 1.01),
        "cluster_trend_max": 41,
        "homologar": True,
        "sns": False,
        "relleno": True,
        "fit_astrom": True,
        "fit_depth": False,
        "survival": True,
        "comparativo": True,
        "tabla_resumen_tipo": "single",
        "depth_labels": [
            ("d_max", "proyectil", "max"),
            ("d_mean", "proyectil", "mean"),
            ("D_max", "lecho", "max"),
            ("D_mean", "lecho", "mean"),
        ],
        "depth_rows": [
            (r"$d_{\max}$", "d_max", "proyectil", "max"),
            (r"$d_{mean}$", "d_mean", "proyectil", "mean"),
            (r"$D_{\max}$", "D_max", "lecho", "max"),
            (r"$D_{mean}$", "D_mean", "lecho", "mean"),
        ],
        "fonts": {
            "axes_label": 15,
            "tick_label": 13,
            "legend": 15,
            "title": 16,
            "subplot_label": 14,
            "annotation": 20,
            "heatmap_cell": 8,
            "heatmap_annot": 13,
            "comp_legend": 12,
        },
        "fig_dir": BASE / "figuras" / "49_mov",
        "cache_dir": BASE / "datos" / "procesados" / "49_mov",
        "out_dir": BASE / "datos" / "49_mov",
    },
}

FF = 0.15
V = 50.0


def _font(cfg, key):
    """Devuelve un tamaño de fuente de la configuración del set.

    Args:
        cfg (dict): Configuración del set (clave ``fonts``).
        key (str): Nombre del tamaño de fuente.

    Returns:
        int: Tamaño de fuente en puntos.
    """
    return cfg["fonts"][key]


def _sem(expr):
    """Error estándar de la media (equivalente a pandas ``.sem(ddof=1)``).

    Args:
        expr: Expresión de Polars (columna).

    Returns:
        Expresión de Polars con el error estándar de la media.
    """
    return expr.std(ddof=1) / expr.count().sqrt()


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------

def power_law(x, a, c):
    """Ley de potencia sin traslación: ``y = a * (x / 0.17) ** c``.

    Args:
        x (float | np.ndarray): Variable independiente.
        a (float): Amplitud.
        c (float): Exponente.

    Returns:
        float | np.ndarray: Valor de la ley de potencia.
    """
    scalar_input = np.ndim(x) == 0
    x = np.atleast_1d(np.asarray(x, dtype=float))
    result = a * (x / 0.17) ** c
    return result.item() if scalar_input else result


def astrom_model(v, A, beta_r, alpha, lambda_param, v_uncorr, D):
    """Modelo de Aström para la distribución de tamaños de fragmentos eyectados.

    Args:
        v (float | np.ndarray): Tamaño del fragmento.
        A (float): Amplitud global.
        beta_r (float): Fracción de fragmentos secundarios.
        alpha (float): Exponente del término de potencia.
        lambda_param (float): Escala del término de corte exponencial.
        v_uncorr (float): Tamaño de normalización del término poissoniano.
        D (float): Dimensión fractal.

    Returns:
        float | np.ndarray: Densidad de tamaños modelada.
    """
    v = np.maximum(v, 1e-12)
    cutoff_term = ((2.0 / lambda_param) ** D) * v
    term1 = (1.0 - beta_r) * (v**-alpha) * np.exp(-cutoff_term)
    poisson_term = (1.0 / v_uncorr) * ((v ** (1.0 / D)) + lambda_param) ** D
    term2 = beta_r * np.exp(-poisson_term)
    return A * (term1 + term2)


def power_law_cluster(k, a):
    """Ley de potencia para el tamaño de cluster: ``a * k ** (-2.8)``.

    Args:
        k (float | np.ndarray): Tamaño de cluster.
        a (float): Amplitud.

    Returns:
        float | np.ndarray: PDF del modelo.
    """
    return a * k ** (-2.8)


def survival_cumsum(modelo, k, *params, k_max, n_grid=1000):
    """Supervivencia P(K > k) integrando numéricamente la PDF del modelo.

    Args:
        modelo (Callable): Función de densidad del modelo.
        k (float | np.ndarray): Tamaños de cluster de consulta.
        *params: Parámetros del modelo.
        k_max (float): Tamaño máximo de cluster (corte de integración).
        n_grid (int): Número de puntos de la grilla de integración.

    Returns:
        float | np.ndarray: Supervivencia P(K > k).
    """
    k = np.asarray(k, dtype=float)
    scalar = np.ndim(k) == 0
    k_arr = np.atleast_1d(k)
    k_grid = np.linspace(1, k_max, n_grid)
    dk = k_grid[1] - k_grid[0]
    pdf_vals = modelo(k_grid, *params)
    pdf_vals = np.maximum(pdf_vals, 0)
    cdf = np.cumsum(pdf_vals) * dk
    cdf_norm = cdf / cdf[-1]
    surv = np.where(k_arr <= k_max, 1.0 - np.interp(k_arr, k_grid, cdf_norm), 0.0)
    return surv.item() if scalar else surv


def survival_power_law(k, K_max):
    """Supervivencia analítica de la ley de potencia con corte en ``K_max``.

    Args:
        k (float | np.ndarray): Tamaño de cluster.
        K_max (float): Tamaño máximo de cluster.

    Returns:
        float | np.ndarray: Supervivencia P(K > k).
    """
    k = np.asarray(k, dtype=float)
    num = K_max ** (1 - 2.8) - k ** (1 - 2.8)
    den = K_max ** (1 - 2.8) - 1
    return np.where(k <= K_max, np.maximum(num / den, 0.0), 0.0)


# ---------------------------------------------------------------------------
# Lectura y capa de datos (Polars)
# ---------------------------------------------------------------------------

def leer_crudo(cfg):
    """Lee el CSV crudo y aplica la homologación del set 49 (np 3->1, ang 90->0).

    Args:
        cfg (dict): Configuración del set (clave ``csv``).

    Returns:
        pl.DataFrame: Datos crudos con columnas en Float64.
    """
    df = pl.read_csv(cfg["csv"])
    # Polars puede leer los flotantes como Float32; se fuerza Float64 para
    # reproducir exactamente la precisión de pandas (evita redondeos en la
    # asignación de celdas de la grilla).
    for c in df.columns:
        if df[c].dtype == pl.Float32:
            df = df.with_columns(pl.col(c).cast(pl.Float64))
    if cfg["homologar"]:
        df = df.with_columns(
            pl.col("np").replace(3, 1),
            pl.col("angulo").replace(90, 0),
        )
    return df


def grilla_pl(cfg, df, np_val):
    """Equivalente en Polars a ``grilla()`` del script original.

    Args:
        cfg (dict): Configuración del set.
        df (pl.DataFrame): Datos crudos.
        np_val (int): Número de proyectiles a filtrar.

    Returns:
        tuple[pl.DataFrame, pl.DataFrame]: ``(base_grid, data)``, la grilla
        completa con la columna ``existe`` y los datos con ``grid_x``/``grid_y``.
    """
    data = df.filter(pl.col("np") == np_val)
    step = data.filter(pl.col("posx") > 0)["posx"].min()
    data = data.with_columns(
        (pl.col("posx") / step).round().cast(pl.Int64).alias("grid_x"),
        (pl.col("posy") / step).round().cast(pl.Int64).alias("grid_y"),
    )
    x_min, y_min = data["grid_x"].min(), data["grid_y"].min()
    x_max, y_max = data["grid_x"].max(), data["grid_y"].max()
    reales = (
        data.select(["grid_x", "grid_y"])
        .unique()
        .with_columns(pl.lit(True).alias("existe"))
    )
    base_grid = (
        pl.DataFrame({"grid_x": range(x_min, x_max + 1)})
        .join(pl.DataFrame({"grid_y": range(y_min, y_max + 1)}), how="cross")
        .join(reales, on=["grid_x", "grid_y"], how="left")
        .with_columns(pl.col("existe").fill_null(False))
    )
    return base_grid, data


def _yields_por_celda(sim, base_grid):
    """Yield por celda al timestep final.

    Nota: ``proyectil_erosion != False`` en pandas considera los NaN (partículas
    de lecho) como True; en Polars se reproduce con ``fill_null(True)``.

    Args:
        sim (pl.DataFrame): Datos de una simulación.
        base_grid (pl.DataFrame): Grilla base con la columna ``existe``.

    Returns:
        tuple[pl.DataFrame, pl.Series]: ``(res, valores_reales)``, el yield por
        celda sobre la grilla completa y los valores de las celdas existentes.
    """
    i_max = sim["i"].max()
    erosion_count = (
        sim.filter((pl.col("i") == i_max) & (pl.col("proyectil_erosion").fill_null(True)))
        .group_by(["grid_x", "grid_y"])
        .agg(pl.col("id").n_unique().alias("yield"))
    )
    res = base_grid.join(erosion_count, on=["grid_x", "grid_y"], how="left")
    res = res.with_columns(
        pl.when(pl.col("existe") & pl.col("yield").is_null())
        .then(pl.lit(0))
        .otherwise(pl.col("yield"))
        .alias("yield")
    )
    valores_reales = res.filter(pl.col("existe"))["yield"]
    return res, valores_reales


def extract_cluster_data(cfg, sim, np_val, angle, ff, v, rot=None):
    """Agrupa por cluster al timestep final (equivalente a la versión pandas).

    Args:
        cfg (dict): Configuración del set.
        sim (pl.DataFrame): Datos de la simulación.
        np_val (int): Número de proyectiles.
        angle (int): Ángulo de impacto.
        ff (float): Fracción de llenado.
        v (float): Velocidad de impacto.
        rot (dict | None): Parámetros de rotación (set 49).

    Returns:
        pl.DataFrame: Clusters con ``count``, centro de masa, radios,
        ``km``, ``rg`` y ``angulo_vel``.
    """
    i_max = sim["i"].max()
    mask = (pl.col("i") == i_max) & (
        (pl.col("type") == 1) | pl.col("proyectil_erosion").fill_null(False)
    )
    df_filt = sim.filter(mask)
    if df_filt.is_empty():
        return pl.DataFrame()
    df_cluster = df_filt.group_by(["posx", "posy", "cluster"]).agg(
        pl.len().alias("count"),
        pl.col("x").mean().alias("x_cm"),
        pl.col("y").mean().alias("y_cm"),
        pl.col("z").mean().alias("z_cm"),
        pl.col("vx").mean().alias("v_cm_x"),
        pl.col("vy").mean().alias("v_cm_y"),
        pl.col("vz").mean().alias("v_cm_z"),
        pl.col("x").var(ddof=0).alias("var_x"),
        pl.col("y").var(ddof=0).alias("var_y"),
        pl.col("z").var(ddof=0).alias("var_z"),
    )
    v_cm = df_cluster.select(["v_cm_x", "v_cm_y", "v_cm_z"]).to_numpy()
    km = 0.5 * np.sum(v_cm**2, axis=1)
    rg = np.sqrt(
        df_cluster["var_x"].to_numpy()
        + df_cluster["var_y"].to_numpy()
        + df_cluster["var_z"].to_numpy()
    )
    v_mag = np.sqrt(2 * km)
    with np.errstate(divide="ignore", invalid="ignore"):
        angulo_vel = np.where(
            v_mag > 0,
            np.degrees(
                np.arccos(np.clip(df_cluster["v_cm_z"].to_numpy() / v_mag, -1.0, 1.0))
            ),
            0.0,
        )
    out = df_cluster.with_columns(
        pl.Series("km", km),
        pl.Series("rg", rg),
        pl.Series("angulo_vel", angulo_vel),
        pl.lit(np_val).alias("np"),
        pl.lit(angle).alias("angulo"),
        pl.lit(ff).alias("ff"),
        pl.lit(v).alias("v"),
    ).drop(["var_x", "var_y", "var_z"])
    if rot is not None:
        out = out.with_columns([pl.lit(rot[k]).alias(k) for k in cfg["rot_cols"]])
    return out


def procesar_datos(cfg, forzar=False):
    """Extrae ``df_dist``, ``df_cluster`` y ``df_proyectil`` usando caché CSV.

    Si los tres CSV procesados ya existen se reutilizan; en caso contrario se
    filtran los CSV crudos y se guardan. ``df_proyectil`` incluye ``grid_x`` y
    ``grid_y`` para poder reconstruir los heatmaps sin volver a filtrar.

    Args:
        cfg (dict): Configuración del set (clave ``cache_dir``).
        forzar (bool): Si ``True`` ignora la caché y reprocesa todo.

    Returns:
        tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]: ``(df_dist,
        df_cluster, df_proyectil)``.
    """
    cache = cfg["cache_dir"]
    cache.mkdir(parents=True, exist_ok=True)
    paths = {
        "df_dist": cache / "df_dist.csv",
        "df_cluster": cache / "df_cluster.csv",
        "df_proyectil": cache / "df_proyectil.csv",
    }
    if not forzar and all(p.exists() for p in paths.values()):
        return (
            pl.read_csv(paths["df_dist"]),
            pl.read_csv(paths["df_cluster"]),
            pl.read_csv(paths["df_proyectil"]),
        )

    df = leer_crudo(cfg)
    datos_distribucion, datos_cluster, datos_proyectil = [], [], []
    for n, alfa in cfg["casos"]:
        base_grid, df_grid = grilla_pl(cfg, df, n)
        for ff in cfg["ff_vals"]:
            vels = sorted(
                df_grid.filter(pl.col("ff") == ff)["vel0"].abs().unique().to_list()
            )
            for v in vels:
                mask_base = (
                    (pl.col("ff") == ff)
                    & (pl.col("vel0") == -1 * v)
                    & (pl.col("np") == n)
                    & (pl.col("angulo") == alfa)
                )
                if not cfg["rot_cols"]:
                    rots = [None]
                else:
                    rotacion = df_grid.filter(mask_base)[cfg["rot_cols"]].unique()
                    rots = list(rotacion.iter_rows(named=True))
                for rot in rots:
                    if rot is None:
                        mask = mask_base
                    else:
                        mask = mask_base & (
                            (pl.col("rot_angulo") == rot["rot_angulo"])
                            & (pl.col("rot_eje_x") == rot["rot_eje_x"])
                            & (pl.col("rot_eje_y") == rot["rot_eje_y"])
                            & (pl.col("rot_eje_z") == rot["rot_eje_z"])
                        )
                    sim = df_grid.filter(mask)
                    if sim.is_empty():
                        continue
                    res, valores = _yields_por_celda(sim, base_grid)
                    df_out = pl.DataFrame(
                        {
                            "vel0": abs(v),
                            "yield": valores,
                            "ff": ff,
                            "np": n,
                            "angulo": alfa,
                        }
                    )
                    if rot is not None:
                        df_out = df_out.with_columns(
                            [pl.lit(rot[k]).alias(k) for k in cfg["rot_cols"]]
                        )
                    datos_distribucion.append(df_out)
                    df_cluster_i = extract_cluster_data(cfg, sim, n, alfa, ff, v, rot)
                    if not df_cluster_i.is_empty():
                        datos_cluster.append(df_cluster_i)
                    datos_proyectil.append(sim)

    df_dist = pl.concat(datos_distribucion, how="vertical")
    df_cluster = pl.concat(datos_cluster, how="vertical")
    df_proyectil = pl.concat(datos_proyectil, how="vertical")

    df_dist.write_csv(paths["df_dist"])
    df_cluster.write_csv(paths["df_cluster"])
    df_proyectil.write_csv(paths["df_proyectil"])
    return df_dist, df_cluster, df_proyectil


def _prepare_cluster_data(cfg, df_cluster, df_raw, relleno):
    """Conteos ``F(n)_j`` por impacto y masa total ``M_t`` (Polars).

    Con ``relleno=True`` (set 49) completa las combinaciones vacías con ceros,
    de modo que la suma del ensamble incluye los clústeres no observados.

    Args:
        cfg (dict): Configuración del set (clave ``rot_cols``).
        df_cluster (pl.DataFrame): Datos de clusters.
        df_raw (pl.DataFrame): Datos crudos (para la grilla).
        relleno (bool): Si ``True`` rellena las combinaciones vacías con ceros.

    Returns:
        list[pl.DataFrame]: Un dataframe por caso con ``F(n)_j`` y ``M_t``.
    """
    rot_cols = cfg["rot_cols"]
    dfs = []
    for n, alfa in cfg["casos"]:
        df_plot = df_cluster.filter(
            (pl.col("angulo") == alfa) & (pl.col("np") == n)
        ).rename({"ff": PHI})

        group_cols = ["posx", "posy", "np", "angulo", PHI, "count", "v"] + rot_cols
        df_por_impacto = df_plot.group_by(group_cols).agg(pl.len().alias("F(n)_j"))
        mt_group_cols = ["posx", "posy", PHI, "v"] + rot_cols
        df_por_impacto = df_por_impacto.with_columns(
            pl.col("F(n)_j").sum().over(mt_group_cols).alias("M_t")
        )

        if relleno:
            df_tamanos = sorted(df_plot["count"].unique().to_list())
            _, df_grid = grilla_pl(cfg, df_raw, n)
            grid_existe = df_grid.select(["posx", "posy"]).unique()
            pos_phi = grid_existe.join(df_plot.select(PHI).unique(), how="cross")
            all_combos = pos_phi.join(pl.DataFrame({"count": df_tamanos}), how="cross")
            all_combos = all_combos.join(df_plot.select("v").unique(), how="cross")
            if rot_cols:
                all_combos = all_combos.join(df_plot.select(rot_cols).unique(), how="cross")
            merge_cols = ["posx", "posy", PHI, "count", "v"] + rot_cols
            df_por_impacto = all_combos.join(df_por_impacto, on=merge_cols, how="left")
            all_cols = ["posx", "posy", "v", PHI, "count"] + rot_cols
            df_por_impacto = df_por_impacto.sort(all_cols).with_columns(
                pl.col("F(n)_j").fill_null(0)
            )
            df_por_impacto = df_por_impacto.with_columns(
                pl.col("F(n)_j").sum().over(mt_group_cols).alias("M_t")
            )
        dfs.append(df_por_impacto)
    return dfs


# ---------------------------------------------------------------------------
# Tablas resumen (Polars) + pivots de presentación (Pandas)
# ---------------------------------------------------------------------------

def _pivot_para_heatmap(resumen, index_cols, column_cols, values, cfg):
    """Convierte un resumen (Polars) en un pivot (Pandas) para LaTeX/heatmap.

    Es la única parte que conserva Pandas: el pivot MultiIndex es necesario
    para ``to_latex`` y para extraer los niveles de columna (v, phi).

    Args:
        resumen (pl.DataFrame): Datos resumidos a pivotear.
        index_cols (list[str]): Columnas del índice del pivot.
        column_cols (list[str]): Columnas de las columnas del pivot.
        values (str): Columna de valores.
        cfg (dict): Configuración del set (no usado directamente).

    Returns:
        pd.DataFrame: Pivot MultiIndex con columnas (v, phi) e índice (Np, orientación).
    """
    import pandas as pd

    pivot = resumen.to_pandas().pivot_table(
        index=index_cols,
        columns=column_cols,
        values=values,
        aggfunc="first",
    )
    pivot.columns.names = [r"$v$", r"$\phi$"]
    pivot.index.names = [r"$N_p$", "Orientación"]
    idx_frame = pivot.index.to_frame()
    idx_frame["Orientación"] = idx_frame["Orientación"].astype(object)
    idx_frame.loc[idx_frame[r"$N_p$"] == 1, "Orientación"] = ""
    idx_frame["Orientación"] = (
        idx_frame["Orientación"].map({0: "V", 90: "H"}).fillna(idx_frame["Orientación"])
    )
    pivot.index = pd.MultiIndex.from_frame(idx_frame)
    return pivot


def tabla_rebote_proyectil(cfg, df):
    """Probabilidad de rebote (masa) por (np, angulo, ff, vel0).

    Args:
        cfg (dict): Configuración del set (clave ``rot_cols``).
        df (pl.DataFrame): Datos crudos de una simulación.

    Returns:
        tuple[pd.DataFrame, pl.DataFrame]: ``(pivot, resumen)``, el pivot para
        heatmap (índice np/angulo, columnas vel0/ff) y el resumen con ``se``.
    """
    i_max = df["i"].max()
    proj = df.filter((pl.col("type") == 2) & (pl.col("i") == i_max)).with_columns(
        pl.col("vel0").abs().alias("vel0")
    )
    gcols = ["np", "angulo", "ff", "vel0", "posx", "posy"] + cfg["rot_cols"]
    prob_impacto = proj.group_by(gcols).agg(
        pl.col("proyectil_erosion").mean().alias("p_rebote")
    )
    resumen = (
        prob_impacto.group_by(["np", "angulo", "ff", "vel0"])
        .agg(
            pl.col("p_rebote").mean().alias("probabilidad"),
            _sem(pl.col("p_rebote")).alias("se"),
        )
        .sort(["np", "angulo", "ff", "vel0"])
    )
    pivot = _pivot_para_heatmap(
        resumen, ["np", "angulo"], ["vel0", "ff"], "probabilidad", cfg
    )
    return pivot, resumen


def tabla_porcentaje_proyectil(cfg, df):
    """Proporción de erosión del proyectil por (np, angulo, ff, vel0).

    Args:
        cfg (dict): Configuración del set (clave ``rot_cols``).
        df (pl.DataFrame): Datos crudos de una simulación.

    Returns:
        tuple[pd.DataFrame, pl.DataFrame]: ``(pivot, resumen)`` con el
        porcentaje de erosión y su error estándar.
    """
    i_max = df["i"].max()
    proj = df.filter(
        (pl.col("type") == 2)
        & (pl.col("i") == i_max)
        & pl.col("proyectil_erosion")
    ).with_columns(pl.col("vel0").abs().alias("vel0"))
    eros = df.filter((pl.col("type") == 1) & (pl.col("i") == i_max)).with_columns(
        pl.col("vel0").abs().alias("vel0")
    )
    gcols = ["np", "angulo", "ff", "vel0", "posx", "posy"] + cfg["rot_cols"]
    total_proyectiles = proj.group_by(gcols).agg(pl.len().alias("count_proj"))
    total_erosion = eros.group_by(gcols).agg(pl.len().alias("count_eros"))
    df_merged = total_proyectiles.join(
        total_erosion, on=gcols, how="full", coalesce=True
    ).with_columns(
        pl.col("count_proj").fill_null(0),
        pl.col("count_eros").fill_null(0),
    )
    df_merged = df_merged.with_columns(
        (pl.col("count_proj") + pl.col("count_eros")).alias("total_eventos")
    )
    df_merged = df_merged.with_columns(
        pl.when(pl.col("total_eventos") > 0)
        .then(100 * pl.col("count_proj") / pl.col("total_eventos"))
        .otherwise(0)
        .alias("porcentaje_erosion")
    )
    resumen = (
        df_merged.group_by(["np", "angulo", "ff", "vel0"])
        .agg(
            pl.col("porcentaje_erosion").mean().alias("mean"),
            _sem(pl.col("porcentaje_erosion")).alias("se"),
        )
        .sort(["np", "angulo", "ff", "vel0"])
    )
    pivot = _pivot_para_heatmap(
        resumen, ["np", "angulo"], ["vel0", "ff"], "mean", cfg
    )
    return pivot, resumen


def tabla_erosion_cruda(cfg, df):
    """Datos crudos de erosión por posición (posx, posy).

    Args:
        cfg (dict): Configuración del set (clave ``rot_cols``).
        df (pl.DataFrame): Datos crudos de una simulación.

    Returns:
        pl.DataFrame: Eventos por celda con ``count_proj``, ``count_eros``,
        ``total_eventos`` y ``porcentaje_erosion``.
    """
    i_max = df["i"].max()
    proj = df.filter(
        (pl.col("type") == 2)
        & (pl.col("i") == i_max)
        & pl.col("proyectil_erosion")
    ).with_columns(pl.col("vel0").abs().alias("vel0"))
    eros = df.filter((pl.col("type") == 1) & (pl.col("i") == i_max)).with_columns(
        pl.col("vel0").abs().alias("vel0")
    )
    gcols = ["np", "angulo", "ff", "vel0", "posx", "posy"] + cfg["rot_cols"]
    total_proyectiles = proj.group_by(gcols).agg(pl.len().alias("count_proj"))
    total_erosion = eros.group_by(gcols).agg(pl.len().alias("count_eros"))
    df_merged = total_proyectiles.join(
        total_erosion, on=gcols, how="full", coalesce=True
    ).with_columns(
        pl.col("count_proj").fill_null(0),
        pl.col("count_eros").fill_null(0),
    )
    df_merged = df_merged.with_columns(
        (pl.col("count_proj") + pl.col("count_eros")).alias("total_eventos")
    )
    df_merged = df_merged.with_columns(
        pl.when(pl.col("total_eventos") > 0)
        .then(100 * pl.col("count_proj") / pl.col("total_eventos"))
        .otherwise(0)
        .alias("porcentaje_erosion")
    )
    return df_merged.sort(gcols)


def _depth_stats(cfg, sub, agg_name):
    """Estadísticos de profundidad por celda: media y error estándar.

    Args:
        cfg (dict): Configuración del set (clave ``rot_cols``).
        sub (pl.DataFrame): Datos de profundidad de un caso.
        agg_name (str): ``"max"`` para profundidad máxima o ``"mean"`` para media.

    Returns:
        tuple[float, float]: ``(mean, sem)`` de la profundidad por celda.
    """
    expr = pl.col("depth").max() if agg_name == "max" else pl.col("depth").mean()
    stats = sub.group_by(["posx", "posy"] + cfg["rot_cols"]).agg(expr.alias("depth"))
    n = stats.height
    mean = stats["depth"].mean()
    sem = stats["depth"].std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    return mean, sem


def _preparar_profundidad(cfg, df_proyectil, n, alfa, tipo, agg_name):
    """Prepara el dataframe de profundidades (proyectil o lecho) por caso.

    Args:
        cfg (dict): Configuración del set (clave ``rot_cols``).
        df_proyectil (pl.DataFrame): Datos crudos por partícula.
        n (int): Número de proyectiles.
        alfa (int): Ángulo de impacto.
        tipo (str): ``"proyectil"`` o ``"lecho"``.
        agg_name (str): ``"max"`` o ``"mean"`` para la profundidad por celda.

    Returns:
        pl.DataFrame: Profundidad por (vel0, posx, posy, ff) y rotaciones.
    """
    df = df_proyectil.filter(
        (pl.col("np") == n) & (pl.col("angulo") == alfa)
    )
    if tipo == "proyectil":
        df = df.filter(
            pl.col("proyectil_erosion").eq(False)
            & (pl.col("i") == 80)
            & (pl.col("z") < 100e-6)
        )
    else:
        df = df.filter((pl.col("type") == 1) & (pl.col("i") == 0) & (pl.col("z") < 100e-6))
    df = df.with_columns(
        pl.col("vel0").abs().alias("vel0"),
        (100 - pl.col("z") * 1e6).alias("depth"),
    )
    expr = pl.col("depth").max() if agg_name == "max" else pl.col("depth").mean()
    df = df.group_by(["vel0", "posx", "posy", "ff"] + cfg["rot_cols"]).agg(
        expr.alias("depth")
    )
    return df


def _jacobiano_numerico(modelo, x, params, delta=1e-7):
    """Jacobiano numérico por diferencias centrales (para intervalos de confianza).

    Args:
        modelo (Callable): Función del modelo.
        x (np.ndarray): Variable independiente.
        params (np.ndarray): Vector de parámetros en el punto de evaluación.
        delta (float): Paso de las diferencias centrales.

    Returns:
        np.ndarray: Matriz Jacobiana de forma ``(len(x), len(params))``.
    """
    n = len(x)
    p = len(params)
    J = np.empty((n, p))
    for j in range(p):
        params_up = params.copy()
        params_dn = params.copy()
        params_up[j] += delta
        params_dn[j] -= delta
        J[:, j] = (modelo(x, *params_up) - modelo(x, *params_dn)) / (2 * delta)
    return J


def _ajustar_con_ci(modelo, x, y, p0, bounds):
    """Ajuste con intervalos de confianza sobre la nube de puntos (t de Student 95%).

    Args:
        modelo (Callable): Función del modelo.
        x (np.ndarray): Variable independiente.
        y (np.ndarray): Variable dependiente.
        p0 (list[float]): Estimación inicial de parámetros.
        bounds (tuple[list, list]): Cota inferior y superior de cada parámetro.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(popt, ci)``, los parámetros ajustados
        y el semiancho de la incertidumbre (95 %).
    """
    popt, _ = curve_fit(modelo, x, y, p0=p0, maxfev=5000, bounds=bounds)
    J = _jacobiano_numerico(modelo, x, popt)
    residuos = y - modelo(x, *popt)
    n_pts = len(x)
    n_params = len(popt)
    dof = max(n_pts - n_params, 1)
    sigma2 = np.dot(residuos, residuos) / dof
    cov_theta = sigma2 * np.linalg.pinv(J.T @ J)
    t_val = t.ppf(0.975, dof) if dof > 0 else 1.0
    ci = np.sqrt(np.diag(cov_theta)) * t_val
    return popt, ci


def tabla_ajustes_profundidad(cfg, df_proyectil):
    """Ajustes de ley de potencia (sin b) de las profundidades (set 9).

    Args:
        cfg (dict): Configuración del set (clave ``casos`` y ``ff_vals``).
        df_proyectil (pl.DataFrame): Datos crudos por partícula.

    Returns:
        pl.DataFrame: Filas con ``profundidad``, ``n``, ``angulo``, ``ff``,
        ``a`` y ``c`` (con incertidumbre de uncertainties).
    """
    variantes = [
        {
            "nombre": "d_max (proyectil)",
            "tipo": "proyectil",
            "agg": "max",
            "vel0_min": 0,
            "bounds": ([0, 0], [np.inf, 4]),
        },
        {
            "nombre": "d_mean (proyectil)",
            "tipo": "proyectil",
            "agg": "mean",
            "vel0_min": 0,
            "bounds": ([0, 0], [np.inf, 4]),
        },
        {
            "nombre": "D_max (lecho)",
            "tipo": "lecho",
            "agg": "max",
            "vel0_min": 5,
            "bounds": ([-np.inf, 0], [np.inf, np.inf]),
        },
        {
            "nombre": "D_mean (lecho)",
            "tipo": "lecho",
            "agg": "mean",
            "vel0_min": 5,
            "bounds": ([0, 0], [np.inf, np.inf]),
        },
    ]

    rows = []
    for var in variantes:
        for n, alfa in cfg["casos"]:
            sub = _preparar_profundidad(cfg, df_proyectil, n, alfa, var["tipo"], var["agg"])
            for ff in cfg["ff_vals"]:
                ff_data = sub.filter(pl.col("ff") == ff)
                if var["vel0_min"] > 0:
                    ff_data = ff_data.filter(
                        (pl.col("vel0") > var["vel0_min"]) & (pl.col("depth") > 0)
                    )
                else:
                    ff_data = ff_data.filter(
                        (pl.col("depth") > 0) & (pl.col("vel0") > 0)
                    )
                x_fit = ff_data["vel0"].to_numpy()
                y_fit = ff_data["depth"].to_numpy()
                if len(x_fit) < 2:
                    continue
                try:
                    popt, se = _ajustar_con_ci(
                        power_law, x_fit, y_fit, p0=[1, 1], bounds=var["bounds"]
                    )
                    vals = unp.uarray(popt, se)
                    rows.append(
                        {
                            "profundidad": var["nombre"],
                            "n": n,
                            "angulo": alfa,
                            "ff": ff,
                            "a": f"{vals[0]}",
                            "c": f"{vals[1]}",
                        }
                    )
                except (RuntimeError, ValueError, np.linalg.LinAlgError):
                    pass
    return pl.DataFrame(rows)


def ajuste_modelos(cfg, subset, n, x_col="vel0", y_col="yield"):
    """Ajusta ley de potencia (y Aström en el set 49) sobre las medias por v.

    Args:
        cfg (dict): Configuración del set (clave ``fit_astrom``).
        subset (pl.DataFrame): Datos de un caso.
        n (int): Número de proyectiles (usado por el set 49).
        x_col (str): Columna independiente (por defecto ``vel0``).
        y_col (str): Columna dependiente (por defecto ``yield``).

    Returns:
        tuple[list[np.ndarray], list[list[str]]]: ``(raw_results,
        formatted_results)`` con los parámetros crudos y formateados con
        incertidumbre.
    """
    raw_results = []
    formatted_results = []

    stats = (
        subset.group_by(x_col)
        .agg(pl.col(y_col).mean().alias("mean"), _sem(pl.col(y_col)).alias("sem"))
        .sort(x_col)
        .filter(pl.col("sem").is_not_null() & (pl.col("sem") >= 0))
    )
    x_fit = stats[x_col].to_numpy()
    y_fit = stats["mean"].to_numpy()

    n_params = 2
    dof = len(x_fit) - n_params
    t_val = t.ppf(0.975, max(dof, 1)) if dof > 0 else 1.0
    popt, pcov = curve_fit(
        power_law,
        x_fit,
        y_fit,
        p0=[1, 1],
        maxfev=5000,
        bounds=([0, 0], [np.inf, 4]),
    )
    err = np.sqrt(np.diag(pcov)) * t_val
    raw_results.append(popt)
    formatted_results.append([f"{v}" for v in unp.uarray(popt, err)])

    if cfg["fit_astrom"]:
        n_params = 6
        dof = len(x_fit) - n_params
        t_val = t.ppf(0.975, max(dof, 1)) if dof > 0 else 1.0
        try:
            popt, pcov = curve_fit(
                astrom_model,
                x_fit,
                y_fit,
                p0=[1, 0.5, 1, 10, 10, 2],
                maxfev=10000,
                bounds=([0, 0, 0, 1e-6, 1e-6, 1], [np.inf, 1, 5, 100, 100, 10]),
            )
            err = np.sqrt(np.diag(pcov)) * t_val
        except (RuntimeError, ValueError):
            popt = np.full(6, np.nan)
            err = np.full(6, np.nan)
        raw_results.append(popt)
        formatted_results.append([f"{v}" for v in unp.uarray(popt, err)])

    return raw_results, formatted_results


def calcular_metricas(cfg, df, modelo, params, x_col="vel0", y_col="yield"):
    """RMSE sobre datos crudos y chi-cuadrado reducido sobre las medias.

    Args:
        cfg (dict): Configuración del set.
        df (pl.DataFrame): Datos de un caso.
        modelo (Callable): Función del modelo.
        params (np.ndarray): Parámetros ajustados.
        x_col (str): Columna independiente.
        y_col (str): Columna dependiente.

    Returns:
        dict: ``{"rmse": float, "chi2_red": float}``.
    """
    y_obs = df[y_col].to_numpy()
    y_pred = modelo(df[x_col].to_numpy(), *params)
    n_params = len(params)
    rmse = np.sqrt(mean_squared_error(y_obs, y_pred))

    stats = (
        df.group_by(x_col)
        .agg(pl.col(y_col).mean().alias("mean"), _sem(pl.col(y_col)).alias("sem"))
        .sort(x_col)
    )
    xs = stats[x_col].to_numpy()
    means = stats["mean"].to_numpy()
    sems = stats["sem"].to_numpy()

    grupos_validos = [
        i for i in range(len(xs)) if sems[i] > 0 and not np.isnan(sems[i])
    ]
    n_grupos_validos = len(grupos_validos)
    chi2 = sum(((means[i] - modelo(xs[i], *params)) / sems[i]) ** 2 for i in grupos_validos)
    chi2_red = (
        chi2 / (n_grupos_validos - n_params) if n_grupos_validos > n_params else np.nan
    )
    return {"rmse": rmse, "chi2_red": chi2_red}


def _valores_caso(cfg, df_dist, df_proyectil, df, n, alfa):
    """Métricas resumidas (Y, profundidades, proporciones) para un caso.

    Args:
        cfg (dict): Configuración del set (claves ``depth_labels`` y ``depth_rows``).
        df_dist (pl.DataFrame): Distribución de yields por repetición.
        df_proyectil (pl.DataFrame): Datos crudos por partícula.
        df (pl.DataFrame): Datos crudos (para tablas de rebote/erosión).
        n (int): Número de proyectiles.
        alfa (int): Ángulo de impacto.

    Returns:
        dict: Métricas por clave (``Y``, profundidades, ``masa``, ``proyectil``)
        con valores ``(media, error estándar)`` o ``None`` si no hay datos.
    """
    values = {}
    sub = df_dist.filter(
        (pl.col("np") == n)
        & (pl.col("angulo") == alfa)
        & (pl.col("ff") == FF)
        & (pl.col("vel0") == V)
    )
    values["reps"] = sub["yield"].count()
    y = sub["yield"].to_numpy()
    values["Y"] = (float(np.mean(y)), float(np.std(y, ddof=1) / np.sqrt(len(y))))

    proj = df_proyectil.filter(
        (pl.col("np") == n) & (pl.col("angulo") == alfa) & (pl.col("ff") == FF)
    )
    df_v = proj.filter(pl.col("vel0").abs() == V)
    for key, tipo, agg_name in cfg["depth_labels"]:
        if tipo == "proyectil":
            df_filt = df_v.filter(
                pl.col("proyectil_erosion").eq(False)
                & (pl.col("i") == 80)
                & (pl.col("z") < 100e-6)
            )
        else:
            df_filt = df_v.filter(
                (pl.col("type") == 1) & (pl.col("i") == 0) & (pl.col("z") < 100e-6)
            )
        if df_filt.is_empty():
            values[key] = None
        else:
            df_filt = df_filt.with_columns((100 - pl.col("z") * 1e6).alias("depth"))
            values[key] = _depth_stats(cfg, df_filt, agg_name)

    _, df_rebote = tabla_rebote_proyectil(cfg, df)
    sub_r = df_rebote.filter(
        (pl.col("np") == n)
        & (pl.col("angulo") == alfa)
        & (pl.col("ff") == FF)
        & (pl.col("vel0") == V)
    )
    if sub_r.is_empty():
        values["masa"] = None
    else:
        rm, rs = sub_r["probabilidad"][0], sub_r["se"][0]
        values["masa"] = (float(rm) * 100, float(rs) * 100)

    _, df_erosion = tabla_porcentaje_proyectil(cfg, df)
    sub_e = df_erosion.filter(
        (pl.col("np") == n)
        & (pl.col("angulo") == alfa)
        & (pl.col("ff") == FF)
        & (pl.col("vel0") == V)
    )
    if sub_e.is_empty():
        values["proyectil"] = None
    else:
        em, es = sub_e["mean"][0], sub_e["se"][0]
        values["proyectil"] = (float(em), float(es))
    return values


def tabla_resumen(cfg, df_dist, df_proyectil, df):
    """Tabla resumen en LaTeX (multi-columna para 9, una columna para 49).

    Args:
        cfg (dict): Configuración del set (claves ``tabla_resumen_tipo``,
        ``casos``, ``depth_rows``).
        df_dist (pl.DataFrame): Distribución de yields por repetición.
        df_proyectil (pl.DataFrame): Datos crudos por partícula.
        df (pl.DataFrame): Datos crudos.

    Returns:
        str: Código LaTeX de la tabla resumen.
    """
    casos = cfg["casos"]

    if cfg["tabla_resumen_tipo"] == "single":
        n, alfa = casos[0]
        vals = _valores_caso(cfg, df_dist, df_proyectil, df, n, alfa)
        rows = [
            (r"$N_p$", f"{n}"),
            (r"$v$", f"${V:.0f}\\,m/s$"),
            (r"$\phi$", f"{FF}"),
            ("Repeticiones", f'{vals["reps"]}'),
            (r"$Y$", f'${vals["Y"][0]:.3f}\\pm{vals["Y"][1]:.3f}$'),
        ]
        for label, key, _tipo, _agg in cfg["depth_rows"]:
            v = vals[key]
            rows.append((label, "---" if v is None else f"${v[0]:.2f}\\pm{v[1]:.2f}\\,\\mu m$"))
        mv = vals["masa"]
        rows.append(
            (r"Proporci\'on de masa", "---" if mv is None else f"${mv[0]:.1f}\\pm{mv[1]:.1f}\\,\\%$")
        )
        ev = vals["proyectil"]
        rows.append(
            (r"Proporci\'on de proyectil", "---" if ev is None else f"${ev[0]:.1f}\\pm{ev[1]:.1f}\\,\\%$")
        )
        lines = [r"\begin{tabular}{lc}"]
        for i, (c1, c2) in enumerate(rows):
            if i == 4:
                lines.append(r"\hline")
            lines.append(f"{c1} & {c2} \\\\")
        lines.append(r"\end{tabular}")
        return "\n".join(lines)

    # multi (set 9)
    vals_list = [_valores_caso(cfg, df_dist, df_proyectil, df, n, a) for n, a in casos]
    rows = []
    rows.append(("Variable", *[rf"${n},{a}$" for n, a in casos]))
    rows.append((r"$N_p$", *[str(n) for n, _ in casos]))
    rows.append((r"$v$", *[f"${V:.0f}\\,m/s$"] * len(casos)))
    rows.append((r"$\phi$", *[str(FF)] * len(casos)))
    rows.append(("Repeticiones", *[str(v["reps"]) for v in vals_list]))
    rows.append(("",) * (len(casos) + 1))
    rows.append((r"$Y$", *[f'${v["Y"][0]:.3f}\\pm{v["Y"][1]:.3f}$' for v in vals_list]))
    for label, key, _tipo, _agg in cfg["depth_rows"]:
        vals = [v[key] for v in vals_list]
        cells = ["---" if x is None else f"${x[0]:.2f}\\pm{x[1]:.2f}\\,\\mu m$" for x in vals]
        rows.append((label, *cells))
    masa = [v["masa"] for v in vals_list]
    rows.append(
        (r"Proporci\'on de masa", *["---" if x is None else f"${x[0]:.1f}\\pm{x[1]:.1f}\\,\\%$" for x in masa])
    )
    ero = [v["proyectil"] for v in vals_list]
    rows.append(
        (r"Proporci\'on de proyectil", *["---" if x is None else f"${x[0]:.1f}\\pm{x[1]:.1f}\\,\\%$" for x in ero])
    )
    lines = [r"\begin{tabular}{l" + "c" * len(casos) + "}"]
    for i, row in enumerate(rows):
        if i == 5:
            lines.append(r"\hline")
        if row[0] == "":
            lines.append(r"\hline")
        else:
            lines.append(" & ".join(row) + r" \\")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def generar_tabla_latex_profundidad(df, var_nombre, caption, label):
    """Tabla LaTeX del exponente c de las profundidades del lecho (set 9).

    Args:
        df (pl.DataFrame): Ajustes de profundidad (salida de
            :func:`tabla_ajustes_profundidad`).
        var_nombre (str): Variante de profundidad a mostrar.
        caption (str): Caption de la tabla.
        label (str): Etiqueta LaTeX de la tabla.

    Returns:
        str: Código LaTeX de la tabla.
    """
    sub = df.filter(pl.col("profundidad") == var_nombre)
    if sub.is_empty():
        return f"% No hay datos disponibles para {var_nombre}"

    config_names = {
        (1, 0): r"\(N_p:1\)",
        (3, 90): r"\(N_p:3\text{V}\)",
        (3, 0): r"\(N_p:3\text{H}\)",
    }
    lines = [
        r"\begin{table}[H]",
        r"    \centering",
        f"    \\caption{{{caption}}}",
        f"    \\label{{{label}}}",
        r"    \begin{tabular}{lcc}",
        r"        \toprule",
        r"        \textbf{Configuración} & \(\phi = 0.15\) & \(\phi = 0.35\) \\",
        r"        \midrule",
    ]
    for (n, angulo), config_label in config_names.items():
        row_vals = [f"        {config_label}"]
        for ff in [0.15, 0.35]:
            val_series = sub.filter(
                (pl.col("n") == n) & (pl.col("angulo") == angulo) & (pl.col("ff") == ff)
            )
            if not val_series.is_empty():
                val_str = str(val_series["c"][0]).replace("+/-", r" \pm ")
                row_vals.append(f"\\({val_str}\\)")
            else:
                row_vals.append("---")
        lines.append(" & ".join(row_vals) + r" \\")
    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def tabla_comparativa_latex(cfg_9, df_dist_9, df_proyectil_9, df_9, cfg_49,
                            df_dist_49, df_proyectil_49, df_49):
    """Tabla comparativa 9 vs 245 repeticiones (Np:1, v=50, phi=0.15).

    Args:
        cfg_9 (dict): Configuración del set 9.
        df_dist_9 (pl.DataFrame): Distribución de yields del set 9.
        df_proyectil_9 (pl.DataFrame): Datos por partícula del set 9.
        df_9 (pl.DataFrame): Datos crudos del set 9.
        cfg_49 (dict): Configuración del set 49.
        df_dist_49 (pl.DataFrame): Distribución de yields del set 49.
        df_proyectil_49 (pl.DataFrame): Datos por partícula del set 49.
        df_49 (pl.DataFrame): Datos crudos del set 49.

    Returns:
        str: Código LaTeX de la tabla comparativa.
    """
    v_9 = _valores_caso(cfg_9, df_dist_9, df_proyectil_9, df_9, 1, 0)
    v_245 = _valores_caso(cfg_49, df_dist_49, df_proyectil_49, df_49, 1, 0)

    def _fmt(val):
        """Formatea un valor ``(media, error)`` como LaTeX ``$a\\pm b$``.

        Args:
            val (tuple[float, float] | None): Par ``(media, error)``.

        Returns:
            str: Código LaTeX formateado o ``"---"`` si el valor es ``None``.
        """
        if val is None:
            return "---"
        return f"${val[0]:.2f}\\pm{val[1]:.2f}$"

    rows = [
        ("Variable", "9 repeticiones", "245 repeticiones"),
        ("Repeticiones", str(v_9["reps"]), str(v_245["reps"])),
        ("", "", ""),
        (r"$Y$", _fmt(v_9["Y"]), _fmt(v_245["Y"])),
        (r"$d_{mean}$ [\(\mu\text{m}\)]", _fmt(v_9["d_mean"]), _fmt(v_245["d_mean"])),
        (r"$D_{max}$ [\(\mu\text{m}\)]", _fmt(v_9["D_max"]), _fmt(v_245["D_max"])),
        (r"$D_{mean}$ [\(\mu\text{m}\)]", _fmt(v_9["D_mean"]), _fmt(v_245["D_mean"])),
        (r"Proporci\'on de masa [\%]", _fmt(v_9["masa"]), _fmt(v_245["masa"])),
        (r"Proporci\'on de proyectil [\%]", _fmt(v_9["proyectil"]), _fmt(v_245["proyectil"])),
    ]
    lines = [
        r"\begin{table}[H]",
        r"    \centering",
        r"    \caption{Comparación de métricas para 9 y 245 repeticiones en la configuración \(N_p:1\), \(v=50\,\text{m/s}\), \(\phi=0.15\).}",
        r"    \label{tab:estadistica_comparativa}",
        r"    \begin{tabular}{lcc}",
        r"    \hline",
    ]
    for i, row in enumerate(rows):
        if i == 2:
            lines.append(r"    \hline")
        if row[0] == "":
            lines.append(r"    \hline")
        else:
            lines.append(f"    {' & '.join(row)} \\\\")
    lines.append(r"    \hline")
    lines.append(r"    \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Capa de dibujo (matplotlib puro)
# ---------------------------------------------------------------------------

def _marker_kwargs(cfg, n, angle):
    """Devuelve el marcador de matplotlib correspondiente al caso (np, ángulo).

    Args:
        cfg (dict): Configuración del set (clave ``marker_styles``).
        n (int): Número de proyectiles.
        angle (int): Ángulo de impacto.

    Returns:
        dict: Argumentos ``marker`` para ``plt.plot``/``plt.scatter``.
    """
    return {"marker": cfg["marker_styles"][(n, angle)]}


def _case_label(n, alfa):
    """Etiqueta de texto del caso: ``$N_p: n$`` con orientación V u H.

    Args:
        n (int): Número de proyectiles.
        alfa (int): Ángulo de impacto (90 = vertical, 0 = horizontal).

    Returns:
        str: Etiqueta formateada en LaTeX.
    """
    if n == 1:
        return rf"$N_p: {n}$"
    orient = "V" if alfa == 90 else "H"
    return rf"$N_p: {n}$ {orient}"


def _subplot_title(cfg, n, alfa, idx):
    """Título del subplot: etiqueta de columna + caso, p. ej. ``(a) $N_p$: 1``.

    Args:
        cfg (dict): Configuración del set (clave ``col_labels``).
        n (int): Número de proyectiles.
        alfa (int): Ángulo de impacto.
        idx (int): Índice del subplot (columna).

    Returns:
        str: Título formateado en LaTeX.
    """
    if n == 1:
        name = r"$N_p$: 1"
    elif n == 3 and alfa == 0:
        name = r"$N_p$: 3 H"
    else:
        name = r"$N_p$: 3 V"
    if cfg["col_labels"]:
        return f"{cfg['col_labels'][idx]} {name}"
    return name


def _make_hollow_markers(ax, cfg):
    """Convierte los marcadores de un axes en huecos (contorno del color del dato).

    Args:
        ax (matplotlib.axes.Axes): Ejes sobre los que aplicar el estilo.
        cfg (dict): Configuración del set (clave ``hollow_lw``, ``sns``).
    """
    for line in ax.lines:
        line.set_markerfacecolor("none")
        line.set_markeredgecolor(line.get_color())
        line.set_markeredgewidth(cfg["hollow_lw"])
    collection_lw = 1.5 if cfg.get("sns", False) else cfg["hollow_lw"]
    for collection in ax.collections:
        edge = collection.get_edgecolors()
        color = collection.get_facecolors()
        collection.set_facecolor("none")
        collection.set_edgecolor(
            color if len(edge) == 0 or (len(edge) > 0 and edge[0, 3] == 0) else edge
        )
        collection.set_linewidth(collection_lw)


def _phi_palette(cfg, n_colors=None):
    """Paleta de colores asociada a los valores de ``ff``.

    Args:
        cfg (dict): Configuración del set (clave ``sns``).
        n_colors (int | None): Número de colores a devolver.

    Returns:
        list: Paleta de colores (bright para seaborn, tab10 para matplotlib).
    """
    if cfg.get("sns", False):
        palette = sns.color_palette("bright")
    else:
        palette = [plt.cm.tab10(i) for i in range(10)]
    return palette[:n_colors] if n_colors is not None else palette


def _ff_marker_handles(cfg):
    """Handles y etiquetas para las leyendas de erosión (phi y casos).

    Args:
        cfg (dict): Configuración del set (claves ``ff_vals``, ``casos``,
        ``marker_styles``, ``sns``).

    Returns:
        tuple[list, list]: ``(handles, labels)`` para ``ax.legend`` o ``fig.legend``.
    """
    palette = _phi_palette(cfg)
    markers = [cfg["marker_styles"][(n, a)] for n, a in cfg["casos"]]
    if cfg.get("sns", False):
        case_labels = []
        for n, a in cfg["casos"]:
            if n == 1:
                case_labels.append(r"$N_p: 1$")
            else:
                case_labels.append(rf"$N_p: {n} " + ("V" if a == 90 else "H") + "$")
        phi_labels = [rf"$\phi$={val}" for val in cfg["ff_vals"]]
    else:
        case_labels = [_case_label(n, a) for n, a in cfg["casos"]]
        phi_labels = [rf"$\phi={val}$" for val in cfg["ff_vals"]]
    handles, labels = [], []
    for idx, phi_val in enumerate(cfg["ff_vals"]):
        handles.append(Line2D([0], [0], color=palette[idx], linewidth=0))
        labels.append(phi_labels[idx])
    for marker, lbl in zip(markers, case_labels):
        for idx in range(len(cfg["ff_vals"])):
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=marker,
                    color=palette[idx],
                    markerfacecolor="none",
                    markeredgewidth=cfg["hollow_lw"],
                    markersize=cfg["handle_mk"],
                    linestyle="none",
                )
            )
            labels.append(lbl)
    return handles, labels


def _colorear_phi_legend(cfg, fig):
    """Colorea los textos de phi de la leyenda con la paleta del set.

    Args:
        cfg (dict): Configuración del set (clave ``ff_vals``).
        fig (matplotlib.figure.Figure): Figura cuya leyenda se colorea.
    """
    palette = _phi_palette(cfg)
    for i in range(len(cfg["ff_vals"])):
        fig.legends[0].get_texts()[i].set_color(palette[i])


def _annotar_c(cfg, ax, c, idx, n_ff, color):
    """Anota el exponente ``c`` del ajuste dentro del subplot.

    Args:
        cfg (dict): Configuración del set (claves ``annot_position``, ``fonts``).
        ax (matplotlib.axes.Axes): Ejes donde anotar.
        c (float): Exponente ajustado.
        idx (int): Índice de la curva (phi).
        n_ff (int): Cantidad de curvas de phi.
        color: Color del texto.
    """
    if cfg["annot_position"] == "top-left":
        ax.text(
            0.03,
            0.97 - idx * 0.055,
            rf"$c={c:.2f}$",
            transform=ax.transAxes,
            color=color,
            fontsize=_font(cfg, "annotation"),
            fontweight="bold",
            ha="left",
            va="top",
        )
    else:
        ax.text(
            0.97,
            0.03 + (n_ff - 1 - idx) * 0.055,
            rf"$c={c:.2f}$",
            transform=ax.transAxes,
            color=color,
            fontsize=_font(cfg, "annotation"),
            fontweight="bold",
            ha="right",
            va="bottom",
        )


def _style_subplot(cfg, ax, n, alfa, show_ylabel, col_idx):
    """Aplica título, etiquetas y grilla al subplot.

    Args:
        cfg (dict): Configuración del set (claves ``fonts``, ``grid_alpha``).
        ax (matplotlib.axes.Axes): Ejes a estilizar.
        n (int): Número de proyectiles.
        alfa (int): Ángulo de impacto.
        show_ylabel (bool): Si ``True`` muestra la etiqueta del eje y.
        col_idx (int): Índice del subplot.
    """
    ax.set_title(_subplot_title(cfg, n, alfa, col_idx), fontsize=_font(cfg, "title"))
    ax.set_xlabel(r"v [$m/s$]", fontsize=_font(cfg, "subplot_label"))
    if show_ylabel:
        ax.set_ylabel(r"$Y$", fontsize=_font(cfg, "subplot_label"))
    ax.grid(axis="y", linestyle="--", alpha=cfg["grid_alpha"])
    ax.grid(axis="x", linestyle="--", alpha=cfg["grid_alpha"])
    ax.grid(which="minor", linestyle=":", alpha=0.3)
    ax.minorticks_on()


def _mpl_scatter_with_se(cfg, ax, data, x, y, hue, marker, capsize=5):
    """Scatter con barras de error por grupo de hue (matplotlib puro).

    Args:
        cfg (dict): Configuración del set (claves ``mk_size``, ``hollow_lw``).
        ax (matplotlib.axes.Axes): Ejes donde dibujar.
        data (pl.DataFrame): Datos por repetición.
        x (str): Columna del eje x.
        y (str): Columna del eje y.
        hue (str): Columna de agrupación (ff).
        marker (str): Marcador de matplotlib.
        capsize (float): Tamaño de las tapas de las barras de error.
    """
    palette = [plt.cm.tab10(i) for i in range(10)]
    hues = sorted(data[hue].unique().to_list())
    for idx, h in enumerate(hues):
        sub = data.filter(pl.col(hue) == h)
        stats = (
            sub.group_by(x)
            .agg(pl.col(y).mean().alias("mean"), _sem(pl.col(y)).alias("se"))
            .sort(x)
        )
        color = palette[idx]
        ax.errorbar(
            stats[x].to_numpy(),
            stats["mean"].to_numpy(),
            yerr=stats["se"].to_numpy(),
            fmt=marker,
            markersize=cfg["mk_size"],
            markerfacecolor="none",
            markeredgecolor=color,
            markeredgewidth=cfg["hollow_lw"],
            color=color,
            linestyle="",
            capsize=capsize,
            capthick=cfg["hollow_lw"],
            elinewidth=cfg["hollow_lw"],
        )


def plot_erosion_loglog(cfg, df_dist):
    """Grafico log-log de erosion (yield vs velocidad) con ajuste de ley de potencia.

    Args:
        cfg (dict): Configuración del set.
        df_dist (pl.DataFrame): Distribución de yields por repetición.

    Returns:
        matplotlib.figure.Figure: Figura generada.
    """
    fig, axes = plt.subplots(1, len(cfg["casos"]), figsize=(5 * len(cfg["casos"]), 6), sharey=True)
    axes = np.atleast_1d(axes)
    palette = [plt.cm.tab10(i) for i in range(10)]
    for i, (n, alfa) in enumerate(cfg["casos"]):
        ax = axes[i]
        subset = df_dist.filter((pl.col("np") == n) & (pl.col("angulo") == alfa))
        df_plot = subset.rename({"ff": PHI}).with_columns(pl.col("vel0").abs().alias(r"$v$"))
        m_kw = _marker_kwargs(cfg, n, alfa)
        ff_vals = sorted(df_plot[PHI].unique().to_list())
        if cfg.get("sns", False):
            sns.lineplot(
                data=df_plot.to_pandas(),
                x=r"$v$",
                y="yield",
                errorbar="se",
                palette="tab10",
                hue=PHI,
                markersize=cfg["mk_size"],
                linestyle="",
                linewidth=cfg["fit_lw"],
                err_style="bars",
                err_kws={"capsize": 5},
                ax=ax,
                **m_kw,
            )
            _make_hollow_markers(ax, cfg)
            if ax.get_legend() is not None:
                ax.get_legend().remove()
        else:
            _mpl_scatter_with_se(cfg, ax, df_plot, r"$v$", "yield", PHI, m_kw["marker"])
            _make_hollow_markers(ax, cfg)

        for idx, ff in enumerate(ff_vals):
            ff_data = df_plot.filter(pl.col(PHI) == ff)
            x_fit = ff_data[r"$v$"].to_numpy()
            y_fit = ff_data["yield"].to_numpy()
            mask_fit = (y_fit > 0) & (x_fit > 0)
            x_fit, y_fit = x_fit[mask_fit], y_fit[mask_fit]
            if len(x_fit) < 3:
                continue
            try:
                popt, _ = curve_fit(
                    power_law,
                    x_fit,
                    y_fit,
                    p0=[1, 1],
                    maxfev=5000,
                    bounds=([0, 0], [np.inf, 4]),
                )
                _, c = popt
                x_sm = np.logspace(np.log10(x_fit.min()), np.log10(x_fit.max()), 200)
                ax.plot(
                    x_sm,
                    power_law(x_sm, *popt),
                    ":",
                    color=palette[idx],
                    alpha=cfg["fit_alpha"],
                    linewidth=cfg["fit_lw"],
                )
                _annotar_c(cfg, ax, c, idx, len(ff_vals), palette[idx])
            except (RuntimeError, ValueError):
                pass

        _style_subplot(cfg, ax, n, alfa, i == 0, i)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1e-10)
        ax.set_ylim(bottom=1e-10)
        ax.set_xlabel(r"$v$ [$m/s$]", fontsize=_font(cfg, "title"))
        ax.set_ylabel(r"$Y$", fontsize=_font(cfg, "title"))
        ax.tick_params(axis="both", labelsize=16)

    handles, labels = _ff_marker_handles(cfg)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=4,
        fontsize=_font(cfg, "legend"),
    )
    _colorear_phi_legend(cfg, fig)
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.14)
    fig.savefig(cfg["fig_dir"] / "plot_erosion_loglog.png", bbox_inches="tight")
    plt.close(fig)


def datos_por_impacto(cfg, df_proyectil):
    """Erosion y momento de cada impacto (una fila por impacto).

    Como todas las partículas tienen la misma masa, ésta se toma como unidad y
    el momento queda expresado en unidades de ``masa * velocidad``. Por impacto
    se toma ``p_total = N_p * v`` con la velocidad de entrada ``|v| = |vel0|``
    (verificada: ``|v(i=0)| = |vel0|`` para todos los proyectiles). Los
    proyectiles erosionados aportan el momento que conservan **al final**, medido
    con su velocidad real ``|v_fin| = sqrt(vx^2 + vy^2 + vz^2)`` en el último
    timestep, y el resto se considera transferido al lecho:
    ``p_absorbido = p_total - p_reflejado``.

    ``p_desechos`` es el momento que se llevan las partículas erosionadas
    (fragmentos de proyectil y del lecho) en el mismo instante, también con su
    velocidad real. Ojo: en esta simulación los proyectiles no rebotan, penetran
    y se frenan dentro del lecho (``|v_fin| < 0.25 |v0|``), por lo que
    ``p_reflejado`` resulta casi nulo y ``p_absorbido`` es prácticamente
    ``p_total``; ``p_desechos`` es la magnitud que sí discrimina.

    ``Y`` es el yield erosionado del impacto (proyectil y lecho), equivalente al
    que produce ``df_dist``: se cuentan las partículas del último timestep con
    ``proyectil_erosion`` no nulo-falso. La tabla se ancla en el conjunto de
    impactos completo (todas las combinaciones de posición, velocidad y rotación
    presentes en los datos), de modo que se conservan también los impactos sin
    erosión alguna (``n_erod = 0``, ``Y = 0``) y aquéllos cuyo proyectil
    desapareció antes del timestep final.

    Args:
        cfg (dict): Configuración del set (clave ``rot_cols``).
        df_proyectil (pl.DataFrame): Datos crudos por partícula.

    Returns:
        pl.DataFrame: Una fila por impacto con ``n_erod``, ``Y``, ``p_total``,
        ``p_reflejado``, ``p_desechos`` y ``p_absorbido``.
    """
    i_max = df_proyectil["i"].max()
    gcols = ["np", "angulo", "ff", "vel0", "posx", "posy"] + cfg["rot_cols"]
    v_fin = (pl.col("vx") ** 2 + pl.col("vy") ** 2 + pl.col("vz") ** 2).sqrt()
    proyectiles = df_proyectil.filter(
        (pl.col("type") == 2) & (pl.col("i") == i_max)
    ).with_columns(
        pl.col("vel0").abs().alias("vel0"),
        v_fin.alias("v_fin"),
        pl.col("proyectil_erosion").fill_null(False).cast(pl.Int64).alias("erod"),
    )
    erosion = df_proyectil.filter(
        (pl.col("i") == i_max) & pl.col("proyectil_erosion").fill_null(True)
    ).with_columns(pl.col("vel0").abs().alias("vel0"))
    tutti = df_proyectil.filter(pl.col("i") == i_max).with_columns(
        pl.col("vel0").abs().alias("vel0"),
        v_fin.alias("v_fin"),
        (
            pl.col("proyectil_erosion").fill_null(False)
            | (pl.col("lecho_erosion") == "true")
        )
        .cast(pl.Int64)
        .alias("desecho"),
    )
    impactos = (
        df_proyectil.select(gcols).with_columns(pl.col("vel0").abs()).unique()
    )
    momento = proyectiles.group_by(gcols).agg(
        (pl.col("erod") * pl.col("v_fin")).sum().alias("p_reflejado"),
        pl.col("erod").sum().alias("n_erod"),
    )
    momento_desechos = tutti.group_by(gcols).agg(
        (pl.col("desecho") * pl.col("v_fin")).sum().alias("p_desechos")
    )
    y_impacto = erosion.group_by(gcols).agg(pl.col("id").n_unique().alias("Y"))
    return (
        impactos.with_columns((pl.col("np") * pl.col("vel0")).alias("p_total"))
        .join(momento, on=gcols, how="left")
        .join(momento_desechos, on=gcols, how="left")
        .join(y_impacto, on=gcols, how="left")
        .with_columns(
            pl.col("n_erod").fill_null(0),
            pl.col("p_reflejado").fill_null(0.0),
            pl.col("p_desechos").fill_null(0.0),
            pl.col("Y").fill_null(0),
        )
        .with_columns((pl.col("p_total") - pl.col("p_reflejado")).alias("p_absorbido"))
        .sort("p_absorbido")
    )


def _xlim_datos(tabla, col, pad=0.03):
    """Límites del eje x ajustados al rango real de los datos.

    Args:
        tabla (pl.DataFrame): Tabla con una fila por impacto.
        col (str): Columna de la abscisa.
        pad (float): Margen relativo a cada extremo, para que los marcadores de
            los valores mínimo y máximo no queden recortados por el borde.

    Returns:
        tuple[float, float]: ``(xmin, xmax)``.
    """
    lo = float(tabla[col].min())
    hi = float(tabla[col].max())
    return (max(lo * (1 - pad), 0.0), hi * (1 + pad))


def _resumen_por_x(cfg, tabla, x_col, y_col):
    """Resumen de ``y_col`` por caso, phi y valor de ``x_col``.

    Necesario cuando ``y_col`` toma pocos valores distintos: la nube de puntos
    crudos no permite comparar dos ``phi`` si todos caen sobre las mismas
    horizontales, pero la media y el rango por abscisa sí.

    Args:
        cfg (dict): Configuración del set.
        tabla (pl.DataFrame): Tabla con una fila por impacto y columna ``ff``.
        x_col (str): Columna de la abscisa.
        y_col (str): Columna de la ordenada.

    Returns:
        pl.DataFrame: Una fila por ``(caso, phi, x_col)`` con las columnas
        ``mean``, ``lo``, ``hi`` y ``n`` agregadas.
    """
    return (
        tabla.group_by(["np", "angulo", "ff", x_col])
        .agg(
            pl.col(y_col).mean().alias("mean"),
            pl.col(y_col).min().alias("lo"),
            pl.col(y_col).max().alias("hi"),
            pl.len().alias("n"),
        )
        .sort(["np", "angulo", "ff", x_col])
    )


def _scatter_por_caso(
    cfg, tabla, x_col, y_col, xlabel, ylabel, fname,
    hline=None, hlabel="", diagonal=False, dlabel="", xlim=None, ylim=None,
    xscale="symlog", xlinthresh=1, yscale="symlog", ylinthresh=1,
    legend_y=-0.14, bottom=0.14, resumen=None, rlabel="", dodge=0.02,
    xlim_datos=False,
):
    """Scatter de ``y_col`` vs ``x_col`` con un panel por caso y color por phi.

    Se dibuja **un punto por impacto**: el color distingue la fracción de
    llenado y el marcador el caso ``(N_p, angulo)``. Por defecto los ejes van en
    escala log-log (simlog con ``linthresh=1``) para poder representar los ceros
    sin descartar repeticiones.

    Args:
        cfg (dict): Configuración del set.
        tabla (pl.DataFrame): Tabla con una fila por impacto y columna ``ff``.
        x_col (str): Columna de la abscisa.
        y_col (str): Columna de la ordenada.
        xlabel (str): Etiqueta del eje x.
        ylabel (str): Etiqueta del eje y.
        fname (str): Nombre del PNG de salida.
        hline (float | None): Valor de la recta de referencia horizontal.
        hlabel (str): Etiqueta de esa recta en la leyenda.
        diagonal (bool): Si ``True`` dibuja la referencia ``y = x``.
        dlabel (str): Etiqueta de la diagonal en la leyenda.
        xlim (tuple | None): Límites del eje x; si ``None`` se usa autoscale.
        xlim_datos (bool): Si ``True`` y no se pasan ``xlim``, cada panel acota el
            eje x a su propio rango de datos (arrancando en 0 si la escala es
            lineal). Necesario cuando los paneles tienen abscisas distintas, como
            ``v_0`` en un set donde ``p_t = N_p v``.
        ylim (tuple | None): Límites del eje y; si ``None`` se usa autoscale.
        xscale (str): Escala del eje x.
        xlinthresh (float): Ancho de la región lineal de la escala simlog del
            eje x.
        yscale (str): Escala del eje y.
        ylinthresh (float): Ancho de la región lineal de la escala simlog del
            eje y; los valores por debajo se dibujan en escala lineal para que
            el cero sea representable.
        legend_y (float): Posición vertical de la leyenda (coordenadas de figura).
        bottom (float): Margen inferior reservado para la leyenda.
        resumen (pl.DataFrame | None): Salida de :func:`_resumen_por_x`; si se
            pasa, se superpone la media de cada phi como línea continua con
            marcadores.
        rlabel (str): Descripción de esas medias en la leyenda.
        dodge (float): Separación horizontal entre las medias de distintas
            ``phi``, como fracción del rango de abscisas del panel; las medias
            quedan a ``±dodge/2`` del valor central.

    Returns:
        matplotlib.figure.Figure: Figura generada.
    """
    palette = _phi_palette(cfg)
    fig, axes = plt.subplots(1, len(cfg["casos"]), figsize=(5 * len(cfg["casos"]), 6), sharey=True)
    axes = np.atleast_1d(axes)
    ref_lw = cfg["fit_lw"] * 0.5

    for i, (n, alfa) in enumerate(cfg["casos"]):
        ax = axes[i]
        subset = tabla.filter(
            (pl.col("np") == n) & (pl.col("angulo") == alfa)
        ).rename({"ff": PHI})
        m_kw = _marker_kwargs(cfg, n, alfa)
        ff_vals = sorted(subset[PHI].unique().to_list())
        x_lo, x_hi = _xlim_datos(subset, x_col)
        span = x_hi - x_lo
        for idx, ff in enumerate(ff_vals):
            ff_data = subset.filter(pl.col(PHI) == ff)
            if ff_data.is_empty():
                continue
            ax.scatter(
                ff_data[x_col].to_numpy(),
                ff_data[y_col].to_numpy(),
                s=cfg["mk_size"] ** 2,
                facecolors="none",
                linewidths=cfg["hollow_lw"],
                color=palette[idx],
                label=rf"$\phi={ff}$",
                **m_kw,
            )

        if hline is not None:
            ax.axhline(hline, linestyle="--", color="0.4", linewidth=ref_lw, zorder=0)

        if diagonal and not subset.is_empty():
            lim = max(float(subset[x_col].max()), float(subset[y_col].max()))
            ax.plot([0, lim], [0, lim], "--", color="0.4", linewidth=ref_lw, zorder=0)

        if resumen is not None:
            r_case = resumen.filter(
                (pl.col("np") == n) & (pl.col("angulo") == alfa)
            )
            for idx, ff in enumerate(ff_vals):
                r = r_case.filter(pl.col("ff") == ff)
                if r.is_empty():
                    continue
                offset = dodge * span / 2 * (2 * idx - (len(ff_vals) - 1))
                ax.plot(
                    r[x_col].to_numpy() + offset,
                    r["mean"].to_numpy(),
                    "-o",
                    color=palette[idx],
                    markersize=cfg["mk_size"] * 0.6,
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    linewidth=ref_lw * 1.6,
                    zorder=3,
                )

        _style_subplot(cfg, ax, n, alfa, i == 0, i)
        if xscale == "linear":
            ax.set_xscale(xscale)
        else:
            ax.set_xscale(xscale, linthresh=xlinthresh)
        if yscale == "linear":
            ax.set_yscale(yscale)
        else:
            ax.set_yscale(yscale, linthresh=ylinthresh)
        if xlim is not None:
            ax.set_xlim(*xlim)
        elif xlim_datos and not subset.is_empty():
            ax.set_xlim(0.0 if xscale == "linear" else x_lo, x_hi)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_xlabel(xlabel, fontsize=_font(cfg, "title"))
        ax.set_ylabel(ylabel, fontsize=_font(cfg, "title"))
        ax.tick_params(axis="both", labelsize=16)

    handles, labels = _ff_marker_handles(cfg)
    if hline is not None:
        handles.append(Line2D([0], [0], color="0.4", linestyle="--", linewidth=ref_lw))
        labels.append(hlabel)
    if diagonal:
        handles.append(Line2D([0], [0], color="0.4", linestyle="--", linewidth=ref_lw))
        labels.append(dlabel)
    if resumen is not None and rlabel:
        handles.append(
            Line2D(
                [0], [0], color="0.3", linestyle="-", marker="o", markersize=6,
                markeredgecolor="white", linewidth=ref_lw * 1.6,
            )
        )
        labels.append(rlabel)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=4,
        fontsize=_font(cfg, "legend"),
    )
    _colorear_phi_legend(cfg, fig)
    plt.tight_layout()
    fig.subplots_adjust(bottom=bottom)
    fig.savefig(cfg["fig_dir"] / fname, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_erosion_distribution(cfg, df_proyectil):
    """Scatter de la erosion (Y) frente al momento absorbido ``p_absorbido``.

    Se dibuja **un punto por impacto** (repetición), no un promedio por
    velocidad: la abscisa es el momento absorbido de esa repetición y la ordenada
    su yield erosionado. No se incluyen barras de error ni ajustes de ley de
    potencia.

    Args:
        cfg (dict): Configuración del set.
        df_proyectil (pl.DataFrame): Datos crudos por partícula.

    Returns:
        matplotlib.figure.Figure: Figura generada.
    """
    return _scatter_por_caso(
        cfg,
        datos_por_impacto(cfg, df_proyectil),
        "p_absorbido",
        "Y",
        r"$p_{abs}=p_t-p_r$",
        r"$Y$",
        "plot_power_law_todos.png",
    )


def plot_momento_transferencia(cfg, df_proyectil):
    """Fracción del momento que conservan los proyectiles erosionados vs ``v_0``.

    Eje x: velocidad de impacto ``v_0 = |vel0|``. Eje y: fracción de momento
    reflejada ``p_r / p_t``, con ``p_r = Σ |v_fin|`` de los proyectiles erosionados
    en el último timestep (su velocidad real) y ``p_t = N_p v_0``. Se dibuja un
    punto por impacto, con un panel por caso ``(N_p, angulo)`` y color por
    fracción de llenado; sobre la nube se superpone la **media de cada grupo**
    ``(v_0, φ)`` como línea continua, que es lo que permite comparar los rellenos.

    En el set 9 la rejilla de ``v_0`` es la misma para todos los casos
    (5, 10, 25 y 50) y lo que cambia con ``N_p`` es ``p_t = N_p v_0``, así que
    los paneles quedan directamente comparables. Los dos ejes van en escala
    lineal y el rango de x se toma de los datos de cada panel.

    Args:
        cfg (dict): Configuración del set.
        df_proyectil (pl.DataFrame): Datos crudos por partícula.

    Returns:
        matplotlib.figure.Figure: Figura generada.
    """
    tabla = datos_por_impacto(cfg, df_proyectil).with_columns(
        (pl.col("p_reflejado") / pl.col("p_total")).alias("p_r_frac")
    )
    return _scatter_por_caso(
        cfg,
        tabla,
        "vel0",
        "p_r_frac",
        r"$v_0$",
        r"$p_r/p_t$",
        "plot_momento_transferencia.png",
        ylim=(0, float(tabla["p_r_frac"].max()) * 1.05),
        xlim_datos=True,
        xscale="linear",
        yscale="linear",
        legend_y=-0.22,
        bottom=0.22,
        resumen=_resumen_por_x(cfg, tabla, "vel0", "p_r_frac"),
        rlabel=r"media de $p_r/p_t$ por grupo",
    )


def suma_momento_reflejado(cfg, df_proyectil):
    """Balance de momento por caso, factor de llenado y velocidad.

    Agrega sobre **todos los impactos** de cada grupo ``(N_p, angulo, ff, v_0)``. Cada
    punto de la figura es el momento reflejado total de las 9 repeticiones de esa
    condición, y no el valor de un impacto individual. Se conserva ``p_t``, el
    momento inicial de un solo impacto, como referencia.

    Args:
        cfg (dict): Configuración del set.
        df_proyectil (pl.DataFrame): Datos crudos por partícula.

    Returns:
        pl.DataFrame: Una fila por ``(N_p, angulo, ff, v_0)`` con ``suma_p_r``,
        ``p_t`` y la cantidad de impactos en ``n``.
    """
    tabla = datos_por_impacto(cfg, df_proyectil)
    return (
        tabla.group_by(["np", "angulo", "ff", "vel0"])
        .agg(
            pl.col("p_reflejado").sum().alias("suma_p_r"),
            pl.col("p_total").first().alias("p_t"),
            pl.len().alias("n"),
        )
        .sort(["np", "angulo", "ff", "vel0"])
    )


def plot_suma_momento_reflejado(cfg, df_proyectil):
    """Suma total del momento reflejado vs ``v_0``, un panel por caso.

    Figura de **una fila y una columna por caso** (los pares ``(N_p, angulo)`` de
    :attr:`cfg['casos']`), con el color distinguiendo el factor de llenado. Cada
    punto es la suma de ``p_r`` sobre los 9 impactos de esa condición, y la línea
    une los valores a lo largo de ``v_0``.

    Args:
        cfg (dict): Configuración del set.
        df_proyectil (pl.DataFrame): Datos crudos por partícula.

    Returns:
        matplotlib.figure.Figure: Figura generada.
    """
    tabla = suma_momento_reflejado(cfg, df_proyectil)
    y_max = float(tabla["suma_p_r"].max())
    return _scatter_por_caso(
        cfg,
        tabla,
        "vel0",
        "suma_p_r",
        r"$v_0$",
        r"$\sum p_r$",
        "plot_suma_momento_reflejado.png",
        ylim=(0, y_max * 1.05),
        xlim_datos=True,
        xscale="linear",
        yscale="linear",
        legend_y=-0.24,
        bottom=0.24,
        resumen=_resumen_por_x(cfg, tabla, "vel0", "suma_p_r"),
        rlabel=r"suma de $p_r$ por grupo",
    )


def plot_dispersion_momento(cfg, df_proyectil):
    """Desviación estándar de la fracción de momento reflejada vs ``v_0``.

    Complementa :func:`plot_momento_transferencia`: allí cada punto es un impacto y
    la línea es la media del grupo ``(v_0, φ)``; aquí cada punto es directamente la
    **desviación estándar** de ``p_r / p_t`` dentro de ese mismo grupo, es decir la
    dispersión entre las repeticiones. Con ambos gráficos se ve la ubicación y la
    repetibilidad de la fracción reflejada.

    Se dibujan marcadores rellenos unidos por una línea, un panel por caso
    ``(N_p, angulo)`` y color por fracción de llenado, en escala lineal. No se usan
    segmentos verticales de min a max.

    Args:
        cfg (dict): Configuración del set.
        df_proyectil (pl.DataFrame): Datos crudos por partícula.

    Returns:
        matplotlib.figure.Figure: Figura generada.
    """
    tabla = datos_por_impacto(cfg, df_proyectil).with_columns(
        (pl.col("p_reflejado") / pl.col("p_total")).alias("p_r_frac")
    )
    gcols = ["np", "angulo", "ff", "vel0"]
    resumen = (
        tabla.group_by(gcols)
        .agg(
            pl.col("p_r_frac").std().alias("std"),
            pl.col("p_r_frac").mean().alias("mean"),
            pl.len().alias("n"),
        )
        .sort(gcols)
    )
    palette = _phi_palette(cfg)
    fig, axes = plt.subplots(
        1, len(cfg["casos"]), figsize=(5 * len(cfg["casos"]), 6), sharey=True
    )
    axes = np.atleast_1d(axes)
    ref_lw = cfg["fit_lw"] * 0.5
    for i, (n, alfa) in enumerate(cfg["casos"]):
        ax = axes[i]
        sub = resumen.filter((pl.col("np") == n) & (pl.col("angulo") == alfa))
        ff_vals = sorted(cfg["ff_vals"])
        for idx, ff in enumerate(ff_vals):
            g = sub.filter(pl.col("ff") == ff)
            if g.is_empty():
                continue
            ax.plot(
                g["vel0"].to_numpy(),
                g["std"].to_numpy(),
                "-o",
                color=palette[idx],
                markersize=cfg["mk_size"] * 0.7,
                markeredgecolor="white",
                markeredgewidth=0.8,
                linewidth=ref_lw * 1.6,
                zorder=3,
            )
        lo, hi = _xlim_datos(sub, "vel0")
        _style_subplot(cfg, ax, n, alfa, i == 0, i)
        ax.set_xscale("linear")
        ax.set_yscale("linear")
        ax.set_xlim(0.0, hi)
        ax.set_ylim(0.0, float(resumen["std"].max()) * 1.1)
        ax.set_xlabel(r"$v_0$", fontsize=_font(cfg, "title"))
        if i == 0:
            ax.set_ylabel(r"desv. estándar de $p_r/p_t$", fontsize=_font(cfg, "title"))
        ax.tick_params(axis="both", labelsize=16)

    handles, labels = [], []
    for idx, ff in enumerate(cfg["ff_vals"]):
        handles.append(
            Line2D(
                [0],
                [0],
                color=palette[idx],
                marker="o",
                markersize=cfg["handle_mk"],
                linewidth=ref_lw * 1.6,
            )
        )
        labels.append(rf"$\phi={ff}$")
    for marker, lbl in zip(
        [cfg["marker_styles"][(n, a)] for n, a in cfg["casos"]],
        [_case_label(n, a) for n, a in cfg["casos"]],
    ):
        for idx in range(len(cfg["ff_vals"])):
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=marker,
                    color=palette[idx],
                    markerfacecolor="none",
                    markeredgewidth=cfg["hollow_lw"],
                    markersize=cfg["handle_mk"],
                    linestyle="none",
                )
            )
            labels.append(lbl)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=4,
        fontsize=_font(cfg, "legend"),
    )
    _colorear_phi_legend(cfg, fig)
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(cfg["fig_dir"] / "plot_dispersion_momento.png", bbox_inches="tight")
    plt.close(fig)
    return fig


def _gaussiana(x, mu, sigma):
    """Densidad de probabilidad normal.

    Args:
        x (np.ndarray | float): Valores donde evaluar la densidad.
        mu (float): Media.
        sigma (float): Desviación estándar.

    Returns:
        np.ndarray: Densidad evaluada en ``x``.
    """
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def plot_hist_p_r(cfg, df_proyectil, bins=10, caso=None):
    """Histogramas de ``p_r`` por velocidad de impacto y factor de llenado.

    Genera **una figura por caso** de :attr:`cfg['casos']`, es decir por par
    ``(N_p, angulo)`` y no solo por ``N_p``: ``N_p: 3`` tiene dos configuraciones
    distintas (``N_p: 3 V`` con angulo 90 y ``N_p: 3 H`` con angulo 0) que son
    corridas diferentes y no deben mezclarse en un mismo histograma. Los archivos
    se llaman ``plot_hist_p_r_Np{n}_ang{angulo}.png``.

    Cada figura tiene una fila por fracción de llenado y una columna por velocidad
    de impacto. Cada celda muestra el histograma de ``p_r`` de las repeticiones de
    ese grupo con una gaussiana ajustada por máxima verosimilitud (``mu`` y ``sigma``
    son la media y la desviación estándar de la muestra). La gaussiana se dibuja
    sobre su soporte completo, desde 0 hasta donde se anula
    (``max(4 sigma, max p_r)``), sin truncar al rango de los datos.

    Ojo con la interpretación: ``p_r`` tiene una alta proporción de ceros exactos
    (el proyectil se detiene por completo) y cola a la derecha, así que la
    gaussiana es una descripción pobre. Por eso cada celda anota ``mu``, ``sigma``
    y el ``R^2`` del ajuste contra el histograma.

    Args:
        cfg (dict): Configuración del set.
        df_proyectil (pl.DataFrame): Datos crudos por partícula.
        bins (int): Cantidad de bins del histograma, comunes a todas las celdas
            para que las columnas sean comparables.
        caso (tuple | None): Si se pasa ``(n_p, angulo)``, solo se genera esa figura.

    Returns:
        list[matplotlib.figure.Figure]: Figuras generadas.
    """
    tabla = datos_por_impacto(cfg, df_proyectil)
    ff_vals = sorted(cfg["ff_vals"])
    targets = cfg["casos"] if caso is None else [caso]
    palette = _phi_palette(cfg)
    figs = []

    for idx, (n, alfa) in enumerate(targets):
        sub_all = tabla.filter((pl.col("np") == n) & (pl.col("angulo") == alfa))
        if sub_all.is_empty():
            continue
        velos = sorted(sub_all["vel0"].unique().to_list())
        x_max = float(sub_all["p_reflejado"].max())
        edges = np.linspace(0, x_max, bins + 1)
        centers = 0.5 * (edges[1:] + edges[:-1])
        width = (edges[1] - edges[0]) * 0.9
        fig, axes = plt.subplots(
            len(ff_vals),
            len(velos),
            figsize=(4 * len(velos), 3.2 * len(ff_vals)),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        for r, ff in enumerate(ff_vals):
            for c, v in enumerate(velos):
                ax = axes[r][c]
                g = (
                    sub_all.filter((pl.col("ff") == ff) & (pl.col("vel0") == v))[
                        "p_reflejado"
                    ]
                    .to_numpy()
                )
                counts, _ = np.histogram(g, bins=edges)
                ax.bar(
                    centers,
                    counts,
                    width=width,
                    color=palette[ff_vals.index(ff)],
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=2,
                )
                mu0 = float(g.mean())
                sig0 = float(g.std()) or 1.0
                mu, sigma = mu0, sig0
                esperado = _gaussiana(centers, mu, sigma) * len(g) * (edges[1] - edges[0])
                ss_res = float(np.sum((counts - esperado) ** 2))
                ss_tot = float(np.sum((counts - counts.mean()) ** 2))
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
                xg = np.linspace(0, max(4 * sigma, x_max), 400)
                ax.plot(
                    xg,
                    _gaussiana(xg, mu, sigma) * len(g) * (edges[1] - edges[0]),
                    "-",
                    color="0.15",
                    linewidth=cfg["fit_lw"],
                    zorder=3,
                )
                ax.text(
                    0.97,
                    0.95,
                    rf"$\mu$={mu:.2f}, $\sigma$={sigma:.2f}" "\n" rf"$R^2$={r2:.2f}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=_font(cfg, "annotation"),
                )
                if r == 0:
                    ax.set_title(rf"$v_0$={v}", fontsize=_font(cfg, "subplot_label"))
                ax.grid(axis="y", linestyle="--", alpha=cfg["grid_alpha"])
                ax.set_xlim(0, x_max)
                if c == 0:
                    ax.set_ylabel(
                        rf"$\phi$={ff}" "\n" "frecuencia",
                        fontsize=_font(cfg, "subplot_label"),
                    )
                if r == len(ff_vals) - 1:
                    ax.set_xlabel(r"$p_r$", fontsize=_font(cfg, "subplot_label"))
                ax.tick_params(axis="both", labelsize=14)

        handles = [
            Line2D([0], [0], color=palette[i], linewidth=cfg["fit_lw"] * 1.6)
            for i in range(len(ff_vals))
        ] + [Line2D([0], [0], color="0.15", linewidth=cfg["fit_lw"])]
        labels = [rf"$\phi={ff}$" for ff in ff_vals] + ["ajuste gaussiano"]
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.08),
            ncol=len(handles),
            fontsize=_font(cfg, "legend"),
        )
        fig.suptitle(_subplot_title(cfg, n, alfa, idx), fontsize=_font(cfg, "title"))
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.18, top=0.88)
        fig.savefig(
            cfg["fig_dir"] / f"plot_hist_p_r_Np{n}_ang{alfa}.png", bbox_inches="tight"
        )
        plt.close(fig)
        figs.append(fig)
    return figs


def plot_erosion_violin(cfg, df_dist):
    """Violinplot del yield por velocidad y fraccion de llenado, uno por caso.

    Args:
        cfg (dict): Configuración del set.
        df_dist (pl.DataFrame): Distribución de yields por repetición.

    Returns:
        matplotlib.figure.Figure: Figura generada.
    """
    palette = [plt.cm.tab10(i) for i in range(10)]
    fig, axes = plt.subplots(1, len(cfg["casos"]), figsize=(5 * len(cfg["casos"]), 5), sharey=True)
    axes = np.atleast_1d(axes)
    for i, (n, alfa) in enumerate(cfg["casos"]):
        ax = axes[i]
        subset = df_dist.filter((pl.col("np") == n) & (pl.col("angulo") == alfa)).rename(
            {"ff": PHI}
        )
        subset = subset.with_columns(
            pl.col("vel0").cast(pl.Int64).cast(pl.Utf8).alias("vel_str")
        )
        if cfg.get("sns", False):
            sns.violinplot(
                data=subset.to_pandas(),
                x="vel_str",
                y="yield",
                hue=PHI,
                palette="tab10",
                ax=ax,
                split=True,
                inner="quartile",
                density_norm="width",
                bw_method=0.3,
            )
            if ax.get_legend() is not None:
                if i != len(cfg["casos"]) - 1:
                    ax.get_legend().remove()
        else:
            vel_cats = sorted(subset["vel_str"].unique().to_list())
            hue_vals = sorted(subset[PHI].unique().to_list())
            n_vels, n_hues = len(vel_cats), len(hue_vals)
            width = 0.8 / n_hues
            for vi, vel in enumerate(vel_cats):
                for hi, phi in enumerate(hue_vals):
                    data = subset.filter(
                        (pl.col("vel_str") == vel) & (pl.col(PHI) == phi)
                    )["yield"].to_numpy()
                    if len(data) < 2:
                        continue
                    parts = ax.violinplot(
                        [data], positions=[vi], widths=width, showmedians=False, showextrema=False
                    )
                    for pc in parts["bodies"]:
                        pc.set_facecolor(palette[hi])
                        pc.set_alpha(0.6)
                        pc.set_edgecolor("black")
                        pc.set_linewidth(0.8)
                    q25, median, q75 = np.percentile(data, [25, 50, 75])
                    offset = (hi - (n_hues - 1) / 2) * width
                    ax.plot([vi + offset, vi + offset], [q25, q75], color="black", linewidth=1.5)
                    ax.plot(
                        vi + offset,
                        median,
                        color="white",
                        markersize=4,
                        marker="D",
                        markeredgecolor="black",
                        markeredgewidth=0.8,
                    )
            ax.set_xticks(range(n_vels))
            ax.set_xticklabels(vel_cats)
        ax.set_title(_subplot_title(cfg, n, alfa, i), fontsize=_font(cfg, "title"))
        ax.set_xlabel(r"$v$ [m/s]")
        if i == 0:
            ax.set_ylabel(r"$Y$")
    global_min = df_dist["yield"].min() - 1
    global_max = df_dist["yield"].max() + 1
    for ax in np.atleast_1d(axes):
        ax.set_ylim(global_min, global_max)
    plt.tight_layout()
    plt.savefig(cfg["fig_dir"] / "plot_violin_erosion.png", bbox_inches="tight")
    plt.close()


def _stats_pdf(df_v):
    """P(k) = N_k / sum(N_k) del ensamble, con los ceros reemplazados por null.

    Args:
        df_v (pl.DataFrame): Datos por impacto de un caso con ``F(n)_j``.

    Returns:
        pl.DataFrame: Probabilidad ``mean`` por (count, phi) y ``mean_plot``
        con los ceros a ``None``.
    """
    stats = df_v.group_by(["count", PHI]).agg(pl.col("F(n)_j").sum().alias("N_k"))
    stats = stats.with_columns(
        pl.when(pl.col("N_k").sum().over(PHI) > 0)
        .then(pl.col("N_k") / pl.col("N_k").sum().over(PHI))
        .otherwise(0.0)
        .alias("mean")
    )
    stats = stats.with_columns(
        pl.when(pl.col("mean") != 0).then(pl.col("mean")).otherwise(None).alias("mean_plot")
    )
    return stats


def _plot_distribution_panel(cfg, df_por_impacto, y_col, ax, n, alfa):
    """Panel individual de la distribucion de tamanos de cluster.

    Args:
        cfg (dict): Configuración del set.
        df_por_impacto (pl.DataFrame): Datos por impacto de un caso.
        y_col (str): Columna de la variable a graficar (no usado directamente).
        ax (matplotlib.axes.Axes): Ejes del panel.
        n (int): Número de proyectiles.
        alfa (int): Ángulo de impacto.

    Returns:
        tuple[None, None]: Firma compat con la version antigua.
    """
    palette = [plt.cm.tab10(i) for i in range(10)]
    m_kw = _marker_kwargs(cfg, n, alfa)
    m_kw["markerfacecolor"] = "none"
    m_kw["markeredgewidth"] = cfg["hollow_lw"]

    stats = _stats_pdf(df_por_impacto)
    phi_vals = sorted(df_por_impacto[PHI].unique().to_list())

    for idx, phi_val in enumerate(phi_vals):
        subset = stats.filter(pl.col(PHI) == phi_val).sort("count")
        valid = subset.filter(pl.col("mean_plot").is_not_null() & (pl.col("mean_plot") > 0))
        ax.plot(
            valid["count"].to_numpy(),
            valid["mean_plot"].to_numpy(),
            label=phi_val,
            color=palette[idx % len(palette)],
            markersize=cfg["mk_size"],
            linestyle="",
            **m_kw,
        )
        if not valid.is_empty():
            first = valid.row(0, named=True)
            x0, y0 = first["count"], first["mean_plot"]
            x_trend = np.logspace(0, np.log10(cfg["cluster_trend_max"]), 100)
            y_trend = y0 * (x_trend / x0) ** (-2.8)
            ax.plot(
                x_trend,
                y_trend,
                ":",
                color=palette[idx],
                alpha=0.9,
                linewidth=3.5,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*cfg["cluster_xlim"])
    ax.set_ylim(*cfg["cluster_ylim"])
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.grid(axis="x", linestyle="--", alpha=0.7)
    ax.grid(which="minor", linestyle=":", alpha=0.3)
    ax.minorticks_on()
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    return None, None


def plot_cluster_size_distribution(cfg, df_cluster, df_raw, relleno, fname, legend_y, bottom):
    """Grafico de distribucion de tamanos de cluster (por velocidad y caso).

    Args:
        cfg (dict): Configuración del set.
        df_cluster (pl.DataFrame): Datos de clusters.
        df_raw (pl.DataFrame): Datos crudos (para la grilla).
        relleno (bool): Si ``True`` rellena combinaciones vacías con ceros.
        fname (str): Nombre del archivo de salida.
        legend_y (float): Posición vertical de la leyenda.
        bottom (float): Margen inferior de la figura.

    Returns:
        matplotlib.figure.Figure: Figura generada.
    """
    dfs = _prepare_cluster_data(cfg, df_cluster, df_raw, relleno)
    velocities = sorted({v for df in dfs for v in df["v"].unique().to_list()})
    scale = 1.55
    with plt.rc_context(
        {
            "axes.labelsize": round(_font(cfg, "axes_label") * scale),
            "xtick.labelsize": round(_font(cfg, "tick_label") * scale),
            "ytick.labelsize": round(_font(cfg, "tick_label") * scale),
            "legend.fontsize": round(_font(cfg, "legend") * scale),
        }
    ):
        fig, axes = plt.subplots(
            len(velocities),
            len(cfg["casos"]),
            figsize=(5 * len(cfg["casos"]), 5 * len(velocities)),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        for i, v in enumerate(velocities):
            for j, (n, alfa) in enumerate(cfg["casos"]):
                ax = axes[i, j]
                df_v = dfs[j].filter(pl.col("v") == v)
                if df_v.is_empty():
                    ax.set_visible(False)
                    continue
                _plot_distribution_panel(cfg, df_v, "Prob_local", ax, n, alfa)
                if i == 0:
                    ax.set_title(_subplot_title(cfg, n, alfa, j), fontsize=round(_font(cfg, "title") * scale))
                if j == 0:
                    ax.set_ylabel(f"v = {v:.0f} m/s\n$P(k)$")
                else:
                    ax.set_ylabel("")
                if i == len(velocities) - 1:
                    ax.set_xlabel("k")
        handles, labels = _ff_marker_handles(cfg)
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, legend_y), ncol=4)
        _colorear_phi_legend(cfg, fig)
        plt.tight_layout()
        fig.subplots_adjust(bottom=bottom)
        fig.savefig(cfg["fig_dir"] / fname, bbox_inches="tight")
        plt.close(fig)


def plot_cluster_survival(cfg, df_cluster, df_raw):
    """Supervivencia P(K > k) de clusters, un subplot por caso.

    Args:
        cfg (dict): Configuración del set.
        df_cluster (pl.DataFrame): Datos de clusters.
        df_raw (pl.DataFrame): Datos crudos (para la grilla).

    Returns:
        matplotlib.figure.Figure: Figura generada.
    """
    dfs = _prepare_cluster_data(cfg, df_cluster, df_raw, relleno=True)
    rot_cols = cfg["rot_cols"]
    scale = 1.55
    with plt.rc_context(
        {
            "axes.labelsize": round(_font(cfg, "axes_label") * scale),
            "xtick.labelsize": round(_font(cfg, "tick_label") * scale),
            "ytick.labelsize": round(_font(cfg, "tick_label") * scale),
            "legend.fontsize": round(_font(cfg, "legend") * scale),
        }
    ):
        fig, axes = plt.subplots(
            1, len(cfg["casos"]), figsize=(5 * len(cfg["casos"]), 5), sharey=True, squeeze=False
        )
        axes = axes[0]
        for j, (n, alfa) in enumerate(cfg["casos"]):
            ax = axes[j]
            df_v = dfs[j]
            if df_v.is_empty():
                ax.set_visible(False)
                continue
            palette = [plt.cm.tab10(i) for i in range(10)]
            m_kw = _marker_kwargs(cfg, n, alfa)
            m_kw["markerfacecolor"] = "none"
            m_kw["markeredgewidth"] = 3

            df_v = df_v.with_columns((pl.col("F(n)_j") / pl.col("M_t")).alias("P_exact"))
            group_cols = ["posx", "posy", PHI, "v"] + rot_cols
            df_v = df_v.sort(["count"] + group_cols)
            df_v = df_v.with_columns(
                (1 - pl.col("P_exact").cum_sum().shift(1).fill_null(0.0))
                .over(group_cols)
                .alias("P_surv")
            )
            df_v = df_v.filter(pl.col("P_surv").is_not_nan())
            stats = (
                df_v.group_by(["count", PHI])
                .agg(
                    pl.col("P_surv").mean().alias("mean"),
                    _sem(pl.col("P_surv")).alias("se"),
                )
                .sort(["count", PHI])
            )
            phi_vals = sorted(df_v[PHI].unique().to_list())
            for idx, phi_val in enumerate(phi_vals):
                subset = stats.filter(pl.col(PHI) == phi_val).sort("count")
                valid = subset.filter(pl.col("mean").is_not_nan() & (pl.col("mean") > 0))
                if valid.is_empty():
                    continue
                ax.plot(
                    valid["count"].to_numpy(),
                    valid["mean"].to_numpy(),
                    label=phi_val,
                    color=palette[idx % len(palette)],
                    markersize=cfg["mk_size"],
                    linestyle="",
                    **m_kw,
                )
                x_trend = np.logspace(0, np.log10(cfg["cluster_xlim"][1]), 100)
                y_trend = x_trend ** (-1.8)
                ax.plot(x_trend, y_trend, "--", color=palette[idx], alpha=0.5, linewidth=1.5)

            ax.set_title(_subplot_title(cfg, n, alfa, j), fontsize=round(_font(cfg, "title") * scale))
            if j == 0:
                ax.set_ylabel(r"$P(K > k)$")
            ax.set_xlabel("k")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(left=0.95, right=cfg["cluster_xlim"][1])
            ax.set_ylim(bottom=2.1e-4, top=1.01)
            ax.grid(axis="y", linestyle="--", alpha=0.7)
            ax.grid(axis="x", linestyle="--", alpha=0.7)
            ax.grid(which="minor", linestyle=":", alpha=0.3)
            ax.minorticks_on()

        handles, labels = _ff_marker_handles(cfg)
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=4)
        _colorear_phi_legend(cfg, fig)
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.2)
        fig.savefig(cfg["fig_dir"] / "plot_cluster_survival.png", bbox_inches="tight")
        plt.close(fig)


def plot_model_comparison(cfg, df_cluster, df_raw):
    """Compara Power law vs Aström ajustados a la distribucion de clusters.

    Args:
        cfg (dict): Configuración del set.
        df_cluster (pl.DataFrame): Datos de clusters.
        df_raw (pl.DataFrame): Datos crudos (para la grilla).

    Returns:
        matplotlib.figure.Figure: Figura generada.
    """
    dfs = _prepare_cluster_data(cfg, df_cluster, df_raw, relleno=True)
    modelos_plot = {"Power law": power_law_cluster, "Aström": astrom_model}
    colors = {"Power law": "C0", "Aström": "C1"}
    linestyles = {"Power law": "-", "Aström": "--"}
    p0 = {"Power law": [1.8], "Aström": [1, 0.3, 2, 15, 10, 2]}
    bounds = {
        "Power law": ([0.8], [2]),
        "Aström": ([0, 0, 0, 1e-6, 1e-6, 1], [np.inf, 1, 5, 100, 100, 10]),
    }
    palette = [plt.cm.tab10(i) for i in range(10)]
    velocities = sorted({v for df in dfs for v in df["v"].unique().to_list()})
    scale = 1.55
    with plt.rc_context(
        {
            "axes.labelsize": round(_font(cfg, "axes_label") * scale),
            "xtick.labelsize": round(_font(cfg, "tick_label") * scale),
            "ytick.labelsize": round(_font(cfg, "tick_label") * scale),
            "legend.fontsize": round(_font(cfg, "legend") * scale),
        }
    ):
        fig, axes = plt.subplots(
            len(velocities),
            len(cfg["casos"]),
            figsize=(5 * len(cfg["casos"]), 5 * len(velocities)),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        for i, v in enumerate(velocities):
            for j, (n, alfa) in enumerate(cfg["casos"]):
                ax = axes[i, j]
                df_v = dfs[j].filter(pl.col("v") == v)
                if df_v.is_empty():
                    ax.set_visible(False)
                    continue
                m_kw = _marker_kwargs(cfg, n, alfa)
                m_kw["markerfacecolor"] = "none"
                m_kw["markeredgewidth"] = 3

                stats = _stats_pdf(df_v)
                phi_vals = sorted(df_v[PHI].unique().to_list())
                for idx, phi_val in enumerate(phi_vals):
                    subset = stats.filter(pl.col(PHI) == phi_val).sort("count")
                    valid = subset.filter(
                        pl.col("mean_plot").is_not_null() & (pl.col("mean_plot") > 0)
                    )
                    ax.plot(
                        valid["count"].to_numpy(),
                        valid["mean_plot"].to_numpy(),
                        color=palette[idx % len(palette)],
                        markersize=cfg["mk_size"],
                        linestyle="",
                        **m_kw,
                    )
                    x_fit = valid["count"].to_numpy()
                    y_fit = valid["mean_plot"].to_numpy()
                    if len(x_fit) < 3:
                        continue
                    for nombre_modelo, modelo in modelos_plot.items():
                        try:
                            popt, _ = curve_fit(
                                modelo,
                                x_fit,
                                y_fit,
                                p0=p0[nombre_modelo],
                                maxfev=10000,
                                bounds=bounds[nombre_modelo],
                            )
                            x_sm = np.logspace(np.log10(x_fit.min()), np.log10(x_fit.max()), 200)
                            ax.plot(
                                x_sm,
                                modelo(x_sm, *popt),
                                linestyle=linestyles[nombre_modelo],
                                color=colors[nombre_modelo],
                                alpha=0.7,
                                linewidth=1.8,
                            )
                        except (RuntimeError, ValueError):
                            pass

                _style_subplot(cfg, ax, n, alfa, j == 0, j)
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlim(left=0.95, right=41)
                ax.set_ylim(2.1e-4, 1.01)
                ax.grid(axis="y", linestyle="--", alpha=0.7)
                ax.grid(axis="x", linestyle="--", alpha=0.7)
                ax.grid(which="minor", linestyle=":", alpha=0.3)
                ax.minorticks_on()
                if i == 0:
                    ax.set_title(_subplot_title(cfg, n, alfa, j), fontsize=round(_font(cfg, "title") * scale))
                if j == 0:
                    ax.set_ylabel(f"v = {v:.0f} m/s\n$P(k)$")
                else:
                    ax.set_ylabel("")
                if i == len(velocities) - 1:
                    ax.set_xlabel("k")

        handles = [
            Line2D([0], [0], color=colors[name], linestyle=linestyles[name], label=name)
            for name in modelos_plot
        ]
        handles_ff, labels_ff = _ff_marker_handles(cfg)
        handles += handles_ff
        labels = list(modelos_plot.keys()) + labels_ff
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=3)
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.10)
        fig.savefig(cfg["fig_dir"] / "plot_PDF_alternativos.png", bbox_inches="tight")
        plt.close(fig)


def plot_model_survival_comparison(cfg, df_cluster, df_raw):
    """Compara las supervivencias de los modelos Power law y Aström con los datos.

    Args:
        cfg (dict): Configuración del set.
        df_cluster (pl.DataFrame): Datos de clusters.
        df_raw (pl.DataFrame): Datos crudos (para la grilla).

    Returns:
        matplotlib.figure.Figure: Figura generada.
    """
    dfs = _prepare_cluster_data(cfg, df_cluster, df_raw, relleno=True)
    rot_cols = cfg["rot_cols"]

    def _make_survival_model(modelo):
        """Envuelve un modelo para ajustar su supervivencia (ultimo param = k_max).

        Args:
            modelo (Callable): PDF del modelo.

        Returns:
            Callable: Función de supervivencia con parametro adicional ``k_max``.
        """
        return lambda k, *p: survival_cumsum(modelo, k, *p[:-1], k_max=p[-1])

    survival_models = {
        "Power law": survival_power_law,
        "Aström": _make_survival_model(astrom_model),
    }
    colors = {"Power law": "C0", "Aström": "C1"}
    linestyles = {"Power law": "-", "Aström": "--"}
    p0 = {"Power law": [300], "Aström": [1, 0.3, 2, 15, 10, 2, 100]}
    bounds = {
        "Power law": ([300], [310]),
        "Aström": ([0, 0, 0, 1e-6, 1e-6, 1, 41], [np.inf, 1, 5, 100, 100, 10, 600]),
    }
    palette = [plt.cm.tab10(i) for i in range(10)]
    scale = 1.55
    with plt.rc_context(
        {
            "axes.labelsize": round(_font(cfg, "axes_label") * scale),
            "xtick.labelsize": round(_font(cfg, "tick_label") * scale),
            "ytick.labelsize": round(_font(cfg, "tick_label") * scale),
            "legend.fontsize": round(_font(cfg, "legend") * scale),
        }
    ):
        fig, axes = plt.subplots(
            1, len(cfg["casos"]), figsize=(5 * len(cfg["casos"]), 5), sharey=True, squeeze=False
        )
        axes = axes[0]
        for j, (n, alfa) in enumerate(cfg["casos"]):
            ax = axes[j]
            df_v = dfs[j]
            if df_v.is_empty():
                ax.set_visible(False)
                continue
            m_kw = _marker_kwargs(cfg, n, alfa)
            m_kw["markerfacecolor"] = "none"
            m_kw["markeredgewidth"] = 3

            df_v = df_v.with_columns((pl.col("F(n)_j") / pl.col("M_t")).alias("P_exact"))
            group_cols = ["posx", "posy", PHI, "v"] + rot_cols
            df_v = df_v.sort(["count"] + group_cols)
            df_v = df_v.with_columns(
                (1 - pl.col("P_exact").cum_sum().shift(1).fill_null(0.0))
                .over(group_cols)
                .alias("P_surv")
            )
            stats_surv = (
                df_v.filter(pl.col("P_surv").is_not_nan())
                .group_by(["count", PHI])
                .agg(
                    pl.col("P_surv").mean().alias("mean"),
                    _sem(pl.col("P_surv")).alias("se"),
                )
                .sort(["count", PHI])
            )
            phi_vals = sorted(df_v[PHI].unique().to_list())
            for idx, phi_val in enumerate(phi_vals):
                subset = stats_surv.filter(pl.col(PHI) == phi_val).sort("count")
                valid = subset.filter(pl.col("mean").is_not_nan() & (pl.col("mean") > 0))
                ax.plot(
                    valid["count"].to_numpy(),
                    valid["mean"].to_numpy(),
                    color=palette[idx % len(palette)],
                    markersize=cfg["mk_size"],
                    linestyle="",
                    **m_kw,
                )
                x_surv = valid["count"].to_numpy()
                y_surv = valid["mean"].to_numpy()
                if len(x_surv) < 3:
                    continue
                for nombre_modelo, surv_model in survival_models.items():
                    try:
                        popt, _ = curve_fit(
                            surv_model,
                            x_surv,
                            y_surv,
                            p0=p0[nombre_modelo],
                            maxfev=10000,
                            bounds=bounds[nombre_modelo],
                        )
                        k_smooth = np.linspace(1, 41, 200)
                        model_surv = surv_model(k_smooth, *popt)
                        ax.plot(
                            k_smooth,
                            model_surv,
                            linestyle=linestyles[nombre_modelo],
                            color=colors[nombre_modelo],
                            alpha=0.7,
                            linewidth=1.8,
                        )
                    except (RuntimeError, ValueError):
                        pass

            ax.set_title(_subplot_title(cfg, n, alfa, j), fontsize=round(_font(cfg, "title") * scale))
            if j == 0:
                ax.set_ylabel(r"$P(K \geq k)$")
            ax.set_xlabel("k")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(left=0.95, right=41)
            ax.set_ylim(bottom=2.1e-4, top=1.01)
            ax.grid(axis="y", linestyle="--", alpha=0.7)
            ax.grid(axis="x", linestyle="--", alpha=0.7)
            ax.grid(which="minor", linestyle=":", alpha=0.3)
            ax.minorticks_on()

        handles = [
            Line2D([0], [0], color=colors[name], linestyle=linestyles[name], label=name)
            for name in survival_models
        ]
        handles_ff, labels_ff = _ff_marker_handles(cfg)
        handles += handles_ff
        labels = list(survival_models.keys()) + labels_ff
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=3)
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.2)
        fig.savefig(cfg["fig_dir"] / "plot_model_survival.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_profundidad(cfg, df_proyectil, tipo, agg_name, fname, ylabel):
    """Grafico de profundidad (max/mean) del proyectil o del lecho.

    Args:
        cfg (dict): Configuración del set.
        df_proyectil (pl.DataFrame): Datos crudos por partícula.
        tipo (str): ``"proyectil"`` o ``"lecho"``.
        agg_name (str): ``"max"`` o ``"mean"``.
        fname (str): Nombre del archivo de salida.
        ylabel (str): Etiqueta del eje y.

    Returns:
        matplotlib.figure.Figure: Figura generada.
    """
    scale = 1.30
    with plt.rc_context(
        {
            "axes.labelsize": round(_font(cfg, "axes_label") * scale),
            "xtick.labelsize": round(_font(cfg, "tick_label") * scale),
            "ytick.labelsize": round(_font(cfg, "tick_label") * scale),
            "legend.fontsize": round(_font(cfg, "legend") * scale),
        }
    ):
        fig, ax = plt.subplots(1, len(cfg["casos"]), figsize=(5 * len(cfg["casos"]), 6), sharey=True)
        ax = np.atleast_1d(ax)
        palette = [plt.cm.tab10(i) for i in range(10)]
        for i, (n, alfa) in enumerate(cfg["casos"]):
            df = _preparar_profundidad(cfg, df_proyectil, n, alfa, tipo, agg_name).rename(
                {"ff": PHI}
            )
            m_kw = _marker_kwargs(cfg, n, alfa)
            if cfg.get("sns", False):
                sns.lineplot(
                    data=df.to_pandas(),
                    x="vel0",
                    y="depth",
                    errorbar="se",
                    palette="tab10",
                    hue=PHI,
                    markersize=cfg["mk_size"],
                    linestyle="",
                    linewidth=cfg["fit_lw"],
                    err_style="bars",
                    err_kws={"capsize": 5},
                    ax=ax[i],
                    **m_kw,
                )
                _make_hollow_markers(ax[i], cfg)
                if ax[i].get_legend() is not None:
                    ax[i].get_legend().remove()
            else:
                _mpl_scatter_with_se(cfg, ax[i], df, "vel0", "depth", PHI, m_kw["marker"])
                _make_hollow_markers(ax[i], cfg)

            if tipo == "proyectil":
                ax[i].set_ylim(bottom=0.95, top=30)
            ax[i].set_xscale("log")
            ax[i].set_yscale("log")
            ax[i].set_title(_subplot_title(cfg, n, alfa, i), fontsize=round(_font(cfg, "title") * scale))
            ax[i].set_xlabel(r"v [$m/s$]")
            if i == 0:
                ax[i].set_ylabel(ylabel)
            else:
                ax[i].set_ylabel("")
            ax[i].grid(axis="y", linestyle="--", alpha=cfg["grid_alpha"])
            ax[i].grid(axis="x", linestyle="--", alpha=cfg["grid_alpha"])
            ax[i].grid(which="minor", linestyle=":", alpha=0.3)
            ax[i].minorticks_on()

            if cfg["fit_depth"]:
                ff_vals = sorted(df[PHI].unique().to_list())
                for idx, ff_val in enumerate(ff_vals):
                    ff_data = df.filter(pl.col(PHI) == ff_val)
                    if tipo == "lecho":
                        ff_data = ff_data.filter((pl.col("vel0") > 5) & (pl.col("depth") > 0))
                        bounds = ([-np.inf, 0], [np.inf, np.inf])
                        x_sm = np.logspace(np.log10(5), np.log10(ff_data["vel0"].max()), 200)
                    else:
                        ff_data = ff_data.filter((pl.col("depth") > 0) & (pl.col("vel0") > 0))
                        bounds = ([0, 0], [np.inf, 4])
                        x_sm = np.logspace(
                            np.log10(ff_data["vel0"].min()), np.log10(ff_data["vel0"].max()), 200
                        )
                    x_fit = ff_data["vel0"].to_numpy()
                    y_fit = ff_data["depth"].to_numpy()
                    if len(x_fit) < 2:
                        continue
                    try:
                        popt, _ = curve_fit(
                            power_law, x_fit, y_fit, p0=[1, 1], maxfev=5000, bounds=bounds
                        )
                        _, c = popt
                        ax[i].plot(
                            x_sm,
                            power_law(x_sm, *popt),
                            ":",
                            color=palette[idx],
                            alpha=cfg["fit_alpha"],
                            linewidth=cfg["fit_lw"],
                        )
                        _annotar_c(cfg, ax[i], c, idx, len(ff_vals), palette[idx])
                        if tipo == "lecho":
                            ax[i].set_ylim(bottom=1e-1, top=1.1e1)
                    except (RuntimeError, ValueError):
                        pass

        handles, labels = _ff_marker_handles(cfg)
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=4)
        _colorear_phi_legend(cfg, fig)
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    fig.savefig(cfg["fig_dir"] / fname, bbox_inches="tight")
    plt.close()


def plot_max_depth_proyectil_distribution(cfg, df_proyectil):
    """Wrapper de :func:`plot_profundidad` para la profundidad máxima del proyectil.

    Args:
        cfg (dict): Configuración del set.
        df_proyectil (pl.DataFrame): Datos crudos por partícula.
    """
    plot_profundidad(cfg, df_proyectil, "proyectil", "max",
                     "plot_max_profundidad_proyectil_vs_VELOCIDAD_vs_NP_vs_ANGULO_vs_FF.png",
                     r"$d_{max}$ [$\mu m$]" if cfg["col_labels"] else r"d [$\mu m$]")


def plot_mean_depth_proyectil_distribution(cfg, df_proyectil):
    """Wrapper de :func:`plot_profundidad` para la profundidad media del proyectil.

    Args:
        cfg (dict): Configuración del set.
        df_proyectil (pl.DataFrame): Datos crudos por partícula.
    """
    plot_profundidad(cfg, df_proyectil, "proyectil", "mean",
                     "plot_mean_profundidad_proyectil_vs_VELOCIDAD_vs_NP_vs_ANGULO_vs_FF.png",
                     r"$d_{mean}$ [$\mu m$]" if cfg["col_labels"] else r"d [$\mu m$]")


def plot_max_depth_lecho_distribution(cfg, df_proyectil):
    """Wrapper de :func:`plot_profundidad` para la profundidad máxima del lecho.

    Args:
        cfg (dict): Configuración del set.
        df_proyectil (pl.DataFrame): Datos crudos por partícula.
    """
    plot_profundidad(cfg, df_proyectil, "lecho", "max",
                     "plot_max_profundidad_lecho_vs_VELOCIDAD_vs_NP_vs_ANGULO_vs_FF.png",
                     r"$D_{max}$ [$\mu m$]" if cfg["col_labels"] else r"d [$\mu m$]")


def plot_mean_depth_lecho_distribution(cfg, df_proyectil):
    """Wrapper de :func:`plot_profundidad` para la profundidad media del lecho.

    Args:
        cfg (dict): Configuración del set.
        df_proyectil (pl.DataFrame): Datos crudos por partícula.
    """
    plot_profundidad(cfg, df_proyectil, "lecho", "mean",
                     "plot_mean_profundidad_lecho_vs_VELOCIDAD_vs_NP_vs_ANGULO_vs_FF.png",
                     r"$D_{mean}$ [$\mu m$]" if cfg["col_labels"] else r"d [$\mu m$]")


def graficar_heatmap(cfg, pivot, cbar_label="Probabilidad de Rebote", fname=None):
    """Grafica un heatmap del pivot (con seaborn para 9_mov, matplotlib para 49_mov).

    Args:
        cfg (dict): Configuración del set (clave ``sns``).
        pivot (pd.DataFrame): Pivot MultiIndex con columnas (v, phi).
        cbar_label (str): Etiqueta de la barra de color.
        fname (str | None): Ruta de salida; si es ``None`` no guarda.
    """
    pivot_numerico = pivot.astype(float)
    fig, ax = plt.subplots(figsize=(10, 6))
    data = pivot_numerico.values
    vmin = data.min()
    vmax = data.max()
    if cfg.get("sns", False):
        _old_rc = dict(plt.rcParams)
        try:
            sns.heatmap(
                pivot_numerico,
                annot=True,
                fmt=".2f",
                cmap="viridis",
                linewidths=0.5,
                annot_kws={"size": 13},
                cbar_kws={"label": cbar_label},
                vmin=vmin,
                vmax=vmax,
                ax=ax,
            )
            sns.set(font_scale=1.3)
            ff_labels = pivot_numerico.columns.get_level_values(1)
            ax.set_xticks(np.arange(len(ff_labels)) + 0.5)
            ax.set_xticklabels(ff_labels)
            ax.set_xlabel(r"$\phi$")
            ax_top = ax.twiny()
            ax_top.set_xlim(ax.get_xlim())
            v_labels = pivot_numerico.columns.get_level_values(0)
            v_unicos = v_labels.unique()
            posiciones_top, etiquetas_top = [], []
            for v_val in v_unicos:
                indices = np.where(v_labels == v_val)[0]
                posiciones_top.append(indices.mean() + 0.5)
                etiquetas_top.append(v_val)
            ax_top.set_xticks(posiciones_top)
            ax_top.set_xticklabels(etiquetas_top)
            ax_top.set_xlabel(r"$v$ [$m/s$]")
            ax_top.tick_params(axis="x", length=0)
            ax.set_ylabel("$N_p$")
            plt.tight_layout()
            if fname:
                plt.savefig(fname, bbox_inches="tight")
        finally:
            plt.rcParams.clear()
            plt.rcParams.update(_old_rc)
        plt.close(fig)
        return
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="white")
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            text_color = "white" if val > (vmin + vmax) / 2 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=_font(cfg, "heatmap_annot"), color=text_color)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    ff_labels = pivot_numerico.columns.get_level_values(1)
    ax.set_xticks(np.arange(len(ff_labels)) + 0.5)
    ax.set_xticklabels(ff_labels)
    ax.set_xlabel(r"$\phi$")
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    v_labels = pivot_numerico.columns.get_level_values(0)
    v_unicos = v_labels.unique()
    posiciones_top, etiquetas_top = [], []
    for v_val in v_unicos:
        indices = np.where(v_labels == v_val)[0]
        posiciones_top.append(indices.mean() + 0.5)
        etiquetas_top.append(v_val)
    ax_top.set_xticks(posiciones_top)
    ax_top.set_xticklabels(etiquetas_top)
    ax_top.set_xlabel(r"$v$ [$m/s$]")
    ax_top.tick_params(axis="x", length=0)
    ax.set_ylabel("$N_p$")
    plt.tight_layout()
    if fname:
        plt.savefig(fname, bbox_inches="tight")
    plt.close(fig)


def _plot_heatmap_desde_proyectil(cfg, df_proyectil, n, alfa, ff, v, rot=None):
    """Reconstruye el heatmap por celda a partir del ``df_proyectil`` cacheado.

    Args:
        cfg (dict): Configuración del set (claves ``rot_cols``, ``grid_nx``,
        ``grid_ny``, ``sns``).
        df_proyectil (pl.DataFrame): Datos crudos por partícula.
        n (int): Número de proyectiles.
        alfa (int): Ángulo de impacto.
        ff (float): Fracción de llenado.
        v (float): Velocidad de impacto.
        rot (dict | None): Parámetros de rotación (set 49).

    Returns:
        matplotlib.figure.Figure: Figura del heatmap.
    """
    rot_cols = cfg["rot_cols"]
    mask = (
        (pl.col("np") == n)
        & (pl.col("angulo") == alfa)
        & (pl.col("ff") == ff)
        & (pl.col("vel0").abs() == v)
    )
    if rot is not None:
        for k in rot_cols:
            mask &= pl.col(k) == rot[k]
    sim = df_proyectil.filter(mask)
    if sim.is_empty():
        return

    reales = (
        df_proyectil.filter(pl.col("np") == n)
        .select(["grid_x", "grid_y"])
        .unique()
        .with_columns(pl.lit(True).alias("existe"))
    )
    x_min, x_max = reales["grid_x"].min(), reales["grid_x"].max()
    y_min, y_max = reales["grid_y"].min(), reales["grid_y"].max()
    base_grid = (
        pl.DataFrame({"grid_x": range(x_min, x_max + 1)})
        .join(pl.DataFrame({"grid_y": range(y_min, y_max + 1)}), how="cross")
        .join(reales, on=["grid_x", "grid_y"], how="left")
        .with_columns(pl.col("existe").fill_null(False))
    )

    i_max = sim["i"].max()
    erosion_count = (
        sim.filter((pl.col("i") == i_max) & (pl.col("proyectil_erosion").fill_null(True)))
        .group_by(["grid_x", "grid_y"])
        .agg(pl.col("id").n_unique().alias("yield"))
    )
    res = base_grid.join(erosion_count, on=["grid_x", "grid_y"], how="left")
    res = res.with_columns(
        pl.when(pl.col("existe") & pl.col("yield").is_null())
        .then(pl.lit(0))
        .otherwise(pl.col("yield"))
        .alias("yield")
    )

    rows = sorted(res.filter(pl.col("existe"))["grid_y"].unique().to_list(), reverse=True)
    cols = sorted(res.filter(pl.col("existe"))["grid_x"].unique().to_list())
    matrix = np.full((len(rows), len(cols)), np.nan)
    for gx, gy, val in res.select(["grid_x", "grid_y", "yield"]).iter_rows():
        if gy in rows and gx in cols:
            matrix[rows.index(gy), cols.index(gx)] = val

    fig, ax = plt.subplots(figsize=(8, 7))
    if cfg.get("sns", False):
        matrix_df = pd.DataFrame(
            matrix,
            index=[f"{int(r)}" for r in rows],
            columns=[f"{int(c)}" for c in cols],
        )
        sns.heatmap(
            matrix_df,
            annot=True,
            fmt=".0f",
            cmap="YlOrRd",
            square=True,
            cbar_kws={"label": "Yield"},
            ax=ax,
        )
    else:
        cmap = plt.cm.YlOrRd.copy()
        cmap.set_bad(color="white")
        im = ax.imshow(matrix, cmap=cmap, aspect="auto", origin="upper")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([f"{int(c)}" for c in cols])
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([f"{int(r)}" for r in rows])
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{int(val)}", ha="center", va="center",
                            fontsize=_font(cfg, "heatmap_cell"))
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Yield")
    rot_suffix = ""
    if rot is not None:
        rot_suffix = (
            f"_rot{rot['rot_angulo']}_{rot['rot_eje_x']}{rot['rot_eje_y']}{rot['rot_eje_z']}"
        )
    ax.set_title(f"Erosión (ff={ff}, v={v}, np={n}, angle={alfa}{rot_suffix})")
    ax.set_xlabel(r"posx ($\sigma /2$)")
    ax.set_ylabel(r"posy ($\sigma /2$)")
    plt.savefig(
        cfg["fig_dir"] / f"heatmap_ff_{ff}_v{v}_np{n}_angle{alfa}{rot_suffix}.png",
        bbox_inches="tight",
    )
    plt.close(fig)


def _graficar_heatmaps_celdas(cfg, df_proyectil):
    """Genera un heatmap por combinacion (caso, ff, v, rotacion).

    Args:
        cfg (dict): Configuración del set.
        df_proyectil (pl.DataFrame): Datos crudos por partícula.
    """
    for n, alfa in cfg["casos"]:
        for ff in cfg["ff_vals"]:
            vels = sorted(
                df_proyectil.filter(
                    (pl.col("np") == n) & (pl.col("angulo") == alfa) & (pl.col("ff") == ff)
                )["vel0"].abs().unique().to_list()
            )
            for v in vels:
                if cfg["rot_cols"]:
                    rots = df_proyectil.filter(
                        (pl.col("np") == n)
                        & (pl.col("angulo") == alfa)
                        & (pl.col("ff") == ff)
                        & (pl.col("vel0").abs() == v)
                    )[cfg["rot_cols"]].unique().iter_rows(named=True)
                else:
                    rots = [None]
                for rot in rots:
                    _plot_heatmap_desde_proyectil(cfg, df_proyectil, n, alfa, ff, v, rot)


def _compute_pdf_stats(df_por_impacto, phi_target):
    """PDF de tamaños de cluster para un valor de phi dado.

    Args:
        df_por_impacto (pl.DataFrame): Datos por impacto de un caso.
        phi_target (float): Fracción de llenado a seleccionar.

    Returns:
        pl.DataFrame: Puntos (count, mean_plot) válidos para ese phi.
    """
    stats = _stats_pdf(df_por_impacto)
    subset = stats.filter(pl.col(PHI) == phi_target).sort("count")
    return subset.filter(pl.col("mean_plot").is_not_null())


def plot_cluster_pdf_comparativo(dfs_9, dfs_49, v_target=50, ff_target=0.15):
    """Compara la PDF de clusters entre 9 y 245 repeticiones (phi fijo).

    Args:
        dfs_9 (list[pl.DataFrame]): Datos preparados del set 9.
        dfs_49 (list[pl.DataFrame]): Datos preparados del set 49.
        v_target (float): Velocidad a comparar.
        ff_target (float): Fracción de llenado a comparar.

    Returns:
        matplotlib.figure.Figure: Figura generada.
    """
    palette = ["#DC143C", "#00A86B"]
    fig, ax = plt.subplots(figsize=(7, 5))
    datasets = [
        (dfs_9, "9 repeticiones", "o", palette[0]),
        (dfs_49, "245 repeticiones", "s", palette[1]),
    ]
    norm = None
    for dfs, label, marker, color in datasets:
        for df_full in dfs:
            df_v = df_full.filter(pl.col("v") == v_target)
            if df_v.is_empty():
                continue
            valid = _compute_pdf_stats(df_v, ff_target)
            ax.plot(
                valid["count"].to_numpy(),
                valid["mean_plot"].to_numpy(),
                marker=marker,
                markersize=13,
                linestyle="",
                color=color,
                markerfacecolor="none",
                markeredgewidth=3,
                label=label,
            )
            if label == "9 repeticiones":
                p_k1 = valid.filter(pl.col("count") == 1)["mean_plot"]
                if not p_k1.is_empty():
                    norm = float(p_k1[0])
    x_trend = np.logspace(0, np.log10(41), 100)
    y_trend = x_trend ** (-2.8)
    if norm is not None:
        y_trend = y_trend * norm
    ax.plot(x_trend, y_trend, "--", color="black", alpha=1, linewidth=1.5, label=r"$k^{-2.8}$")
    ax.set_xscale("log")
    ax.set_xlim(left=0.95, right=41)
    ax.set_yscale("log")
    ax.set_ylim(2.1e-4, 1.01)
    ax.set_xlabel("k")
    ax.set_ylabel(f"v = {v_target:.0f} m/s\n$P(k)$")
    ax.legend(fontsize=12, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.grid(axis="x", linestyle="--", alpha=0.7)
    ax.grid(which="minor", linestyle=":", alpha=0.3)
    ax.minorticks_on()
    fig.subplots_adjust(bottom=0.14)
    plt.tight_layout()
    fig.savefig(BASE / "figuras" / "49_mov" / "plot_PDF_comparativo.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Análisis por set
# ---------------------------------------------------------------------------

def _metricas_comunes(cfg, df_dist, combinaciones, cols_modelo, ajustes_raw, modelos):
    """Calcula RMSE y chi-cuadrado reducido para todas las combinaciones.

    Args:
        cfg (dict): Configuración del set.
        df_dist (pl.DataFrame): Distribución de yields por repetición.
        combinaciones (list[tuple]): Lista de (n, alfa, ff).
        cols_modelo (dict): Índices de parámetros por modelo.
        ajustes_raw (dict): Parámetros crudos por (n, alfa, ff).
        modelos (dict): Funciones de modelo por nombre.

    Returns:
        pl.DataFrame: Métricas por combinación y modelo.
    """
    metricas = []
    for n, alfa, ff in combinaciones:
        subset = df_dist.filter(
            (pl.col("np") == n) & (pl.col("angulo") == alfa) & (pl.col("ff") == ff)
        )
        if subset.is_empty():
            continue
        for nombre_modelo, modelo in modelos.items():
            params = [ajustes_raw[(n, alfa, ff)][i] for i in cols_modelo[nombre_modelo]]
            m = calcular_metricas(cfg, subset, modelo, params)
            metricas.append({"n": n, "angulo": alfa, "ff": ff, "modelo": nombre_modelo, **m})
    return pl.DataFrame(metricas)


def _tablas_rebote_erosion(cfg, df):
    """Pivots y resúmenes de rebote/erosión + heatmaps + CSV.

    Args:
        cfg (dict): Configuración del set.
        df (pl.DataFrame): Datos crudos de la simulación.

    Returns:
        tuple[str, str, pl.DataFrame]: ``(latex_rebote, latex_erosion,
        df_erosion_cruda)``.
    """
    pivot_rebote, df_rebote = tabla_rebote_proyectil(cfg, df)
    graficar_heatmap(
        cfg, pivot_rebote * 100, "Proporción de masa %", cfg["fig_dir"] / "heatmap_rebote.png"
    )
    df_rebote.write_csv(cfg["out_dir"] / "rebote_proyectil.csv")
    latex_rebote = pivot_rebote.to_latex(multirow=True, float_format="%.2f")

    pivot_erosion, df_erosion = tabla_porcentaje_proyectil(cfg, df)
    graficar_heatmap(
        cfg, pivot_erosion, "Proporción de Erosión %", cfg["fig_dir"] / "heatmap_erosion.png"
    )
    df_erosion.write_csv(cfg["out_dir"] / "erosion_proyectil.csv")
    latex_erosion = pivot_erosion.to_latex(multirow=True, float_format="%.3f")

    df_erosion_cruda = tabla_erosion_cruda(cfg, df)
    df_erosion_cruda.write_csv(cfg["out_dir"] / "erosion_cruda.csv")
    return latex_rebote, latex_erosion, df_erosion_cruda


def _analisis_9(cfg, df_dist, df_cluster, df_proyectil, df_raw):
    """Orquesta el análisis completo del set 9 (figuras, tablas y CSV).

    Args:
        cfg (dict): Configuración del set 9.
        df_dist (pl.DataFrame): Distribución de yields por repetición.
        df_cluster (pl.DataFrame): Datos de clusters.
        df_proyectil (pl.DataFrame): Datos crudos por partícula.
        df_raw (pl.DataFrame): Datos crudos.
    """
    casos = cfg["casos"]
    combinaciones = [(n, a, ff) for n, a in casos for ff in cfg["ff_vals"]]

    cols_ajustes = ["power_a", "power_c"]
    ajustes_formatted = {}
    ajustes_raw = {}
    for n, alfa, ff in combinaciones:
        subset = df_dist.filter(
            (pl.col("np") == n) & (pl.col("angulo") == alfa) & (pl.col("ff") == ff)
        )
        if subset.is_empty():
            continue
        raw, formatted = ajuste_modelos(cfg, subset, n)
        ajustes_raw[(n, alfa, ff)] = [*raw[0]]
        ajustes_formatted[(n, alfa, ff)] = [r for sub in formatted for r in sub]

    plot_erosion_distribution(cfg, df_proyectil)
    plot_momento_transferencia(cfg, df_proyectil)
    plot_suma_momento_reflejado(cfg, df_proyectil)
    plot_dispersion_momento(cfg, df_proyectil)
    plot_hist_p_r(cfg, df_proyectil)
    plot_erosion_loglog(cfg, df_dist)
    plot_erosion_violin(cfg, df_dist)

    plot_cluster_size_distribution(
        cfg, df_cluster, df_raw, relleno=False,
        fname="plot_PDF_por_vel_cluster.png", legend_y=-0.07, bottom=0.08,
    )
    plot_cluster_size_distribution(
        cfg, df_cluster, df_raw, relleno=False,
        fname="plot_PDF_por_vel_cluster_sin_relleno.png", legend_y=0.0, bottom=0.11,
    )

    plot_mean_depth_proyectil_distribution(cfg, df_proyectil)
    plot_max_depth_proyectil_distribution(cfg, df_proyectil)
    plot_mean_depth_lecho_distribution(cfg, df_proyectil)
    plot_max_depth_lecho_distribution(cfg, df_proyectil)

    df_ajustes_prof = tabla_ajustes_profundidad(cfg, df_proyectil)
    df_ajustes_prof.write_csv(cfg["out_dir"] / "ajustes_profundidad.csv")
    latex_dmax = generar_tabla_latex_profundidad(
        df_ajustes_prof,
        var_nombre="D_max (lecho)",
        caption=r"Exponente \(c\) del ajuste de la profundidad máxima de excavación del lecho para cada configuración.",
        label="tab:exponentes_dmax",
    )
    latex_dmean = generar_tabla_latex_profundidad(
        df_ajustes_prof,
        var_nombre="D_mean (lecho)",
        caption=r"Exponente \(c\) del ajuste de la profundidad media de excavación del lecho para cada configuración.",
        label="tab:exponentes_dmean",
    )
    (cfg["out_dir"] / "tabla_dmax_lecho.tex").write_text(latex_dmax)
    (cfg["out_dir"] / "tabla_dmean_lecho.tex").write_text(latex_dmean)

    latex_resumen = tabla_resumen(cfg, df_dist, df_proyectil, df_raw)
    (cfg["out_dir"] / "resumen_graficos.tex").write_text(latex_resumen)

    cols_modelo = {"power_law": [0, 1]}
    metricas = _metricas_comunes(
        cfg, df_dist, combinaciones, cols_modelo, ajustes_raw, {"power_law": power_law}
    )
    metricas.write_csv(cfg["out_dir"] / "metricas.csv")

    latex_rebote, latex_erosion, df_erosion_cruda = _tablas_rebote_erosion(cfg, df_raw)

    records_ajustes = []
    for n, alfa, ff in combinaciones:
        fmt = ajustes_formatted.get((n, alfa, ff))
        if fmt is None:
            continue
        records_ajustes.append({"n": n, "angulo": alfa, "ff": ff, r"$\alpha$": fmt[1]})
    df_ajustes = pl.DataFrame(records_ajustes)
    df_ajustes.write_csv(cfg["out_dir"] / "ajustes.csv")

    df_metricas = metricas.to_pandas()
    df_metricas["rmse"] = df_metricas["rmse"].round(2)
    df_metricas["chi2_red"] = df_metricas["chi2_red"].round(2)
    latex_metricas = df_metricas.set_index(["n", "angulo", "ff", "modelo"]).to_latex(multirow=True)

    df_ajustes_latex = df_ajustes.to_pandas()
    df_ajustes_latex[r"$\alpha$"] = df_ajustes_latex[r"$\alpha$"].apply(
        lambda x: f"${str(x).replace('+/-', r' \pm ')}$" if x is not None else x
    )
    latex_ajustes = df_ajustes_latex.set_index(["n", "angulo", "ff"]).to_latex(
        multirow=True, escape=False
    )

    print("\n% --- TABLA RESUMEN ---")
    print(latex_resumen)
    print("\n% --- TABLA DE REBOTE DE PROYECTIL ---")
    print(latex_rebote)
    print("\n% --- TABLA DE MÉTRICAS ---")
    print(latex_metricas)
    print("\n% --- TABLA DE AJUSTES ---")
    print(latex_ajustes)
    print("\n% --- TABLA DE EROSIÓN DE PROYECTIL ---")
    print(latex_erosion)
    print("\n% --- TABLA D_MAX (LECHO) ---")
    print(latex_dmax)
    print("\n% --- TABLA D_MEAN (LECHO) ---")
    print(latex_dmean)
    print("\n% --- DATOS CRUDOS DE EROSIÓN (por posx, posy) ---")
    print(
        df_erosion_cruda.group_by(["np", "angulo", "ff", "vel0"])
        .agg(
            conteo_posiciones=pl.len(),
            media=pl.col("porcentaje_erosion").mean(),
            std=pl.col("porcentaje_erosion").std(),
            min=pl.col("porcentaje_erosion").min(),
            max=pl.col("porcentaje_erosion").max(),
        )
        .sort(["np", "angulo", "ff", "vel0"])
    )

    _graficar_heatmaps_celdas(cfg, df_proyectil)


def _analisis_49(cfg, df_dist, df_cluster, df_proyectil, df_raw, datos_9):
    """Orquesta el análisis completo del set 49 (figuras, tablas, comparativas).

    Args:
        cfg (dict): Configuración del set 49.
        df_dist (pl.DataFrame): Distribución de yields por repetición.
        df_cluster (pl.DataFrame): Datos de clusters.
        df_proyectil (pl.DataFrame): Datos crudos por partícula.
        df_raw (pl.DataFrame): Datos crudos.
        datos_9 (tuple | None): Resultados del set 9 para la comparativa.
    """
    casos = cfg["casos"]
    combinaciones = [(n, a, ff) for n, a in casos for ff in cfg["ff_vals"]]

    cols_ajustes = [
        "power_a", "power_c",
        "astrom_A", "astrom_beta", "astrom_alpha", "astrom_lambda", "astrom_vuncorr", "astrom_D",
    ]
    ajustes_formatted = {}
    ajustes_raw = {}
    for n, alfa, ff in combinaciones:
        subset = df_dist.filter(
            (pl.col("np") == n) & (pl.col("angulo") == alfa) & (pl.col("ff") == ff)
        )
        if subset.is_empty():
            continue
        raw, formatted = ajuste_modelos(cfg, subset, n)
        ajustes_raw[(n, alfa, ff)] = [*raw[0], *raw[1]]
        ajustes_formatted[(n, alfa, ff)] = [r for sub in formatted for r in sub]

    plot_erosion_distribution(cfg, df_proyectil)
    plot_momento_transferencia(cfg, df_proyectil)
    plot_suma_momento_reflejado(cfg, df_proyectil)
    plot_dispersion_momento(cfg, df_proyectil)
    plot_hist_p_r(cfg, df_proyectil)
    plot_erosion_loglog(cfg, df_dist)
    plot_erosion_violin(cfg, df_dist)

    plot_cluster_size_distribution(
        cfg, df_cluster, df_raw, relleno=True,
        fname="plot_PDF_por_vel_cluster.png", legend_y=-0.2, bottom=0.11,
    )
    plot_cluster_survival(cfg, df_cluster, df_raw)

    plot_mean_depth_proyectil_distribution(cfg, df_proyectil)
    plot_max_depth_proyectil_distribution(cfg, df_proyectil)
    plot_mean_depth_lecho_distribution(cfg, df_proyectil)
    plot_max_depth_lecho_distribution(cfg, df_proyectil)

    latex_resumen = tabla_resumen(cfg, df_dist, df_proyectil, df_raw)
    (cfg["out_dir"] / "resumen_graficos.tex").write_text(latex_resumen)

    cols_modelo = {
        "power_law": [0, 1],
        "astrom": [2, 3, 4, 5, 6, 7],
    }
    metricas = _metricas_comunes(
        cfg, df_dist, combinaciones,
        cols_modelo, ajustes_raw,
        {"power_law": power_law, "astrom": astrom_model},
    )
    metricas.write_csv(cfg["out_dir"] / "metricas.csv")

    plot_model_comparison(cfg, df_cluster, df_raw)
    plot_model_survival_comparison(cfg, df_cluster, df_raw)

    latex_rebote, latex_erosion, df_erosion_cruda = _tablas_rebote_erosion(cfg, df_raw)

    records_ajustes = []
    for n, alfa, ff in combinaciones:
        fmt = ajustes_formatted.get((n, alfa, ff))
        if fmt is None:
            continue
        records_ajustes.append({"n": n, "angulo": alfa, "ff": ff, r"$\alpha$": fmt[1]})
    df_ajustes = pl.DataFrame(records_ajustes)
    df_ajustes.write_csv(cfg["out_dir"] / "ajustes.csv")

    df_metricas = metricas.to_pandas()
    df_metricas["rmse"] = df_metricas["rmse"].round(2)
    df_metricas["chi2_red"] = df_metricas["chi2_red"].round(2)
    latex_metricas = df_metricas.set_index(["n", "angulo", "ff", "modelo"]).to_latex(multirow=True)
    df_ajustes_latex = df_ajustes.to_pandas()
    df_ajustes_latex[r"$\alpha$"] = df_ajustes_latex[r"$\alpha$"].apply(
        lambda x: f"${str(x).replace('+/-', r' \pm ')}$" if x is not None else x
    )
    latex_ajustes = df_ajustes_latex.set_index(["n", "angulo", "ff"]).to_latex(
        multirow=True, escape=False
    )

    print("\n% --- TABLA RESUMEN ---")
    print(latex_resumen)
    print("\n% --- TABLA DE REBOTE DE PROYECTIL ---")
    print(latex_rebote)
    print("\n% --- TABLA DE MÉTRICAS ---")
    print(latex_metricas)
    print("\n% --- TABLA DE AJUSTES ---")
    print(latex_ajustes)
    print("\n% --- TABLA DE EROSIÓN DE PROYECTIL ---")
    print(latex_erosion)
    print("\n% --- DATOS CRUDOS DE EROSIÓN (por posx, posy) ---")
    print(
        df_erosion_cruda.group_by(["np", "angulo", "ff", "vel0"])
        .agg(
            conteo_posiciones=pl.len(),
            media=pl.col("porcentaje_erosion").mean(),
            std=pl.col("porcentaje_erosion").std(),
            min=pl.col("porcentaje_erosion").min(),
            max=pl.col("porcentaje_erosion").max(),
        )
        .sort(["np", "angulo", "ff", "vel0"])
    )

    if datos_9 is not None:
        cfg_9, df_dist_9, df_cluster_9, df_proyectil_9, df_raw_9 = datos_9
        latex_comparativa = tabla_comparativa_latex(
            cfg_9, df_dist_9, df_proyectil_9, df_raw_9,
            cfg, df_dist, df_proyectil, df_raw,
        )
        (cfg["out_dir"] / "tabla_comparativa.tex").write_text(latex_comparativa)
        print("\n% --- TABLA COMPARATIVA (9 vs 245 repeticiones) ---")
        print(latex_comparativa)

        dfs_9 = _prepare_cluster_data(cfg_9, df_cluster_9, df_raw_9, relleno=False)
        dfs_49 = _prepare_cluster_data(cfg, df_cluster, df_raw, relleno=True)
        plot_cluster_pdf_comparativo(dfs_9, dfs_49, v_target=50, ff_target=0.15)

    _graficar_heatmaps_celdas(cfg, df_proyectil)


def _preparar_dirs(cfg):
    """Crea los directorios de salida del set (figuras, caché y tablas).

    Args:
        cfg (dict): Configuración del set.
    """
    for d in (cfg["fig_dir"], cfg["cache_dir"], cfg["out_dir"]):
        d.mkdir(parents=True, exist_ok=True)


def main(names=None, forzar=False):
    """Punto de entrada del script: procesa uno o más sets de datos.

    Args:
        names (list[str] | None): Nombres de los sets (por defecto usa los
            argumentos de ``sys.argv`` o todos los sets).
        forzar (bool): Si ``True`` reprocesa los datos ignorando la caché.
    """
    import sys

    if names is None:
        names = [a for a in sys.argv[1:] if a in DATASETS] or list(DATASETS)
    if "--forzar" in sys.argv[1:]:
        forzar = True

    procesados = {}
    if "9_mov" in names:
        cfg = DATASETS["9_mov"]
        _preparar_dirs(cfg)
        df_dist, df_cluster, df_proyectil = procesar_datos(cfg, forzar)
        df_raw = leer_crudo(cfg)
        _analisis_9(cfg, df_dist, df_cluster, df_proyectil, df_raw)
        procesados["9_mov"] = (cfg, df_dist, df_cluster, df_proyectil, df_raw)

    if "49_mov" in names:
        cfg = DATASETS["49_mov"]
        _preparar_dirs(cfg)
        df_dist, df_cluster, df_proyectil = procesar_datos(cfg, forzar)
        df_raw = leer_crudo(cfg)
        _analisis_49(cfg, df_dist, df_cluster, df_proyectil, df_raw, procesados.get("9_mov"))

    print("exito")


if __name__ == "__main__":
    main()
