"""Funciones cinemáticas para visualización de datos de erosión.

Contiene funciones para obtener la grilla global y calcular datos para heatmaps.
Usa Polars para operaciones vectorizadas.
"""

from __future__ import annotations

import itertools

import polars as pl


def get_grid_pairs(df: pl.DataFrame) -> list[tuple[float, float]]:
    """Obtiene la grilla global (producto cartesiano de todos los posx y posy únicos).

    Args:
        df: DataFrame con columnas: posx, posy.

    Returns:
        Lista de tuplas (posx, posy).
    """
    x_vals = df["posx"].unique().to_list()
    y_vals = df["posy"].unique().to_list()

    return list(itertools.product(x_vals, y_vals))


def compute_heatmap_data(
    df: pl.DataFrame,
    ff: float,
    vel0: float,
) -> pl.DataFrame:
    """Calcula partículas erosionadas por posición usando la grilla global.

    Args:
        df: DataFrame con columnas: posx, posy, type, z, vz.
        ff: Fraction fill.
        vel0: Velocidad inicial.

    Returns:
        DataFrame con datos para heatmap (posy como índice, posx como columnas).
    """
    global_x = sorted(df["posx"].unique().to_list())
    global_y = sorted(df["posy"].unique().to_list())

    df_filt = df.filter((pl.col("ff") == ff) & (pl.col("vel0") == vel0))

    if df_filt.is_empty():
        pivot = pl.DataFrame(0, index=global_y, columns=global_x)
    else:
        max_i = df_filt["i"].max()
        df_validas = df_filt.filter(
            (pl.col("i") == max_i)
            & (
                (pl.col("type") == 2)
                | ((pl.col("type") == 1) & (pl.col("z") > 1e-4))
            )
        )

        pivot = (
            df_validas.group_by(["posy", "posx"])
            .agg(pl.len())
            .collect()
            .pivot(
                index="posy",
                columns="posx",
                values="len",
               agg_func="sum",
            )
        )

    pivot = pivot.reindex(index=global_y, columns=global_x, fill_value=0)

    pivot.index = (pivot.index / 1.52).round(2)
    pivot.columns = (pivot.columns / 1.52).round(2)

    return pivot